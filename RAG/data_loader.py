"""
Data loader for KGQA datasets (WebQSP, CWQ).
Handles loading parquet files and preprocessing paths for retrieval.
"""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor

from .config import RAGConfig


@dataclass
class KGSample:
    """A single KGQA sample."""
    question_id: str
    question: str
    ground_truth_paths: List[List[str]]  # List of paths, each path is list of relations
    topic_entity: Optional[str] = None
    answer_entities: Optional[List[str]] = None
    graph: Optional[List[List[str]]] = None  # KG triples: [[subj, rel, obj], ...]
    raw_paths: Optional[List[Any]] = None    # Raw path objects (dicts/strings) preserving entities
    
    def get_path_strings(self) -> List[str]:
        """Convert paths to string format for retrieval (Entity-Enhanced if possible)."""
        paths = []
        if self.raw_paths:
            for p in self.raw_paths:
                if isinstance(p, dict):
                    parts = []
                    ents = p.get('entities', [])
                    rels = p.get('relations', [])
                    for i in range(len(rels)):
                        if i < len(ents): parts.append(ents[i])
                        parts.append(rels[i])
                    if len(ents) > len(rels): parts.append(ents[-1])
                    chain = " -> ".join(parts)
                    if chain: paths.append(chain)
                elif isinstance(p, str):
                    paths.append(p)
        
        if not paths:
            return [" -> ".join(path) for path in self.ground_truth_paths if path]
        return paths
    
    def extract_entities_for_path(self, path: str) -> List[str]:
        """
        Extract entities by traversing graph following the given path.
        
        Args:
            path: A relation path like "location.country.languages_spoken" 
                  or "rel1 -> rel2"
        
        Returns:
            List of entities reachable via this path from topic entity
        """
        if not self.graph or not self.topic_entity:
            return []
        
        # Parse path into relation(s)
        if ' -> ' in path:
            relations = [r.strip() for r in path.split(' -> ')]
        else:
            # Single relation
            relations = [path.strip()]
        
        # Build graph index
        graph_by_subject = {}
        for triple in self.graph:
            if len(triple) >= 3:
                subj, rel, obj = triple[0], triple[1], triple[2]
                if subj not in graph_by_subject:
                    graph_by_subject[subj] = []
                graph_by_subject[subj].append((rel, obj))
        
        def relation_matches(path_rel: str, graph_rel: str) -> bool:
            """Check if path relation matches graph relation (flexible matching)."""
            # Direct match
            if path_rel == graph_rel or path_rel in graph_rel or graph_rel in path_rel:
                return True
            
            # Suffix matching - extract last part after dot
            path_suffix = path_rel.split('.')[-1] if '.' in path_rel else path_rel
            graph_suffix = graph_rel.split('.')[-1] if '.' in graph_rel else graph_rel
            
            if path_suffix == graph_suffix:
                return True
            
            # Key word matching (e.g., 'languages_spoken' matches anything with 'languages')
            key_words = ['language', 'birthplace', 'birth', 'death', 'religion', 
                        'country', 'capital', 'president', 'founder', 'location',
                        'spouse', 'parent', 'child', 'team', 'school', 'university']
            for kw in key_words:
                if kw in path_suffix.lower() and kw in graph_suffix.lower():
                    return True
            
            return False
        
        # Traverse graph following relations
        current_entities = [self.topic_entity]
        
        for rel in relations:
            next_entities = []
            for entity in current_entities:
                if entity in graph_by_subject:
                    for edge_rel, target in graph_by_subject[entity]:
                        # Check if relation matches (flexible matching)
                        if relation_matches(rel, edge_rel):
                            next_entities.append(target)
            current_entities = next_entities
            if not current_entities:
                break
        
        # Filter out non-entity values (like m.XXX or g.XXX)
        valid_entities = [e for e in current_entities 
                        if not e.startswith('m.') and not e.startswith('g.')]
        
        return valid_entities


class KGDataLoader:
    """Loader for Knowledge Graph QA datasets."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.data_path = config.dataset_path
        self._train_data: Optional[List[KGSample]] = None
        self._val_data: Optional[List[KGSample]] = None
        self._test_data: Optional[List[KGSample]] = None
        self._all_paths: Optional[List[str]] = None
        self._path_to_idx: Optional[Dict[str, int]] = None
    
    def _parse_row(self, row: pd.Series) -> KGSample:
        """Parse a single row from the parquet file."""
        # Handle different possible column names
        question_id = str(row.get('id', row.get('question_id', row.name)))
        question = row.get('question', row.get('query', ''))
        
        # Parse ground truth paths - prefer shortest_gt_paths over paths
        gt_paths_raw = row.get('shortest_gt_paths', row.get('paths', row.get('ground_truth_paths', [])))
        
        if isinstance(gt_paths_raw, str):
            try:
                gt_paths_raw = json.loads(gt_paths_raw)
            except json.JSONDecodeError:
                gt_paths_raw = []
        
        if gt_paths_raw is None:
            gt_paths_raw = []
        
        # Normalize paths to List[List[str]]
        ground_truth_paths = []
        if gt_paths_raw:
            for path in gt_paths_raw:
                if isinstance(path, str):
                    # Path is a string like "rel1 -> rel2 -> rel3"
                    relations = [r.strip() for r in path.split("->")]
                    ground_truth_paths.append(relations)
                elif isinstance(path, dict):
                    # Path is a dict with relation_chain or relations field
                    # Format: {"full_path": "...", "relation_chain": "rel1.rel2", "relations": ["rel1", "rel2"]}
                    if 'relations' in path and path['relations']:
                        ground_truth_paths.append(path['relations'])
                    elif 'relation_chain' in path and path['relation_chain']:
                        # Split relation_chain by dots or arrows
                        chain = path['relation_chain']
                        if ' -> ' in chain:
                            relations = [r.strip() for r in chain.split(' -> ')]
                        else:
                            # Relations are dot-separated like "location.country.official_language"
                            # Treat the whole thing as a single relation path
                            relations = [chain]
                        ground_truth_paths.append(relations)
                    elif 'full_path' in path:
                        # Extract relations from full_path like "(Entity) --[rel]--> (Entity)"
                        full_path = path['full_path']
                        import re
                        rels = re.findall(r'--\[([^\]]+)\]-->', full_path)
                        if rels:
                            ground_truth_paths.append(rels)
                elif isinstance(path, list):
                    # Path is already a list of relations
                    ground_truth_paths.append(path)
        
        # Get topic entity
        q_entity = row.get('q_entity', row.get('topic_entity', row.get('seed_entity', None)))
        if isinstance(q_entity, str):
            try:
                q_entity = json.loads(q_entity)
                if isinstance(q_entity, list) and q_entity:
                    q_entity = q_entity[0]
            except json.JSONDecodeError:
                pass
        elif isinstance(q_entity, list) and q_entity:
            q_entity = q_entity[0]
        
        # Get answer entities
        answer_entities = row.get('a_entity', row.get('answer_entities', row.get('answer', None)))
        if isinstance(answer_entities, str):
            try:
                answer_entities = json.loads(answer_entities)
            except json.JSONDecodeError:
                answer_entities = [answer_entities]
        
        # Parse graph (KG triples)
        graph = None
        graph_raw = row.get('graph', None)
        if graph_raw is not None:
            if isinstance(graph_raw, str):
                try:
                    import ast
                    graph = ast.literal_eval(graph_raw)
                except (ValueError, SyntaxError):
                    try:
                        graph = json.loads(graph_raw)
                    except json.JSONDecodeError:
                        graph = None
            elif isinstance(graph_raw, list):
                graph = graph_raw
        
        return KGSample(
            question_id=question_id,
            question=question,
            ground_truth_paths=ground_truth_paths,
            topic_entity=q_entity,
            answer_entities=answer_entities,
            graph=graph,
            raw_paths=gt_paths_raw
        )
    
    def _load_split(self, split: str) -> List[KGSample]:
        """Load a specific data split."""
        file_path = self.data_path / f"{split}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        samples = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {split}"):
            sample = self._parse_row(row)
            if sample.question and sample.ground_truth_paths:
                samples.append(sample)
        
        return samples
    
    @property
    def train_data(self) -> List[KGSample]:
        if self._train_data is None:
            self._train_data = self._load_split("train")
        return self._train_data
    
    @property
    def val_data(self) -> List[KGSample]:
        if self._val_data is None:
            self._val_data = self._load_split("val")
        return self._val_data
    
    @property
    def test_data(self) -> List[KGSample]:
        if self._test_data is None:
            self._test_data = self._load_split("test")
        return self._test_data
    
    def build_path_corpus(self, include_test: bool = None) -> Tuple[List[str], Dict[str, int]]:
        """
        Build a corpus of all unique paths from the training data.
        This serves as the retrieval corpus.
        
        Args:
            include_test: If True, include paths from test set (Simulates Full Schema). 
                          If None, uses config setting.
        
        Returns:
            all_paths: List of unique path strings
            path_to_idx: Mapping from path string to index
        """
        if self._all_paths is not None and self._path_to_idx is not None:
            return self._all_paths, self._path_to_idx
            
        if include_test is None:
             include_test = getattr(self.config, 'include_test_in_corpus', False)

        path_set = set()
        
        # Collect paths from train (and optionally val)
        for sample in self.train_data:
            for path_str in sample.get_path_strings():
                path_set.add(path_str)
        
        for sample in self.val_data:
            for path_str in sample.get_path_strings():
                path_set.add(path_str)
        
        if include_test:
            for sample in self.test_data:
                for path_str in sample.get_path_strings():
                    path_set.add(path_str)
        
        self._all_paths = sorted(list(path_set))  # Sort for reproducibility
        self._path_to_idx = {path: idx for idx, path in enumerate(self._all_paths)}
        
        print(f"Built path corpus with {len(self._all_paths)} unique paths")
        return self._all_paths, self._path_to_idx
    
    def build_path_corpus_fast(
        self, 
        include_test: bool = True,
        num_workers: int = 8,
        cache_dir: Optional[Path] = None
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Build path corpus with parallel processing and caching.
        
        Optimizations:
        1. Parallel path extraction using ThreadPoolExecutor
        2. Disk caching to speed up repeated loads
        3. Batch set operations for deduplication
        
        Args:
            include_test: Include test set paths in corpus
            num_workers: Number of parallel workers
            cache_dir: Directory for caching (None = no caching)
            
        Returns:
            all_paths: List of unique path strings
            path_to_idx: Mapping from path string to index
        """
        import time
        start_time = time.time()
        
        # Check cache first
        cache_key = f"corpus_test={include_test}"
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"path_corpus_{hashlib.md5(cache_key.encode()).hexdigest()[:8]}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached = pickle.load(f)
                    self._all_paths = cached['paths']
                    self._path_to_idx = cached['path_to_idx']
                    print(f"Loaded cached corpus: {len(self._all_paths)} paths in {time.time()-start_time:.2f}s")
                    return self._all_paths, self._path_to_idx
                except Exception as e:
                    print(f"Cache load failed: {e}")
        
        # Return cached if already built
        if self._all_paths is not None and self._path_to_idx is not None:
            return self._all_paths, self._path_to_idx
        
        print("Building corpus (optimized parallel processing)...")
        
        # Collect all samples
        all_samples = []
        all_samples.extend(self.train_data)
        all_samples.extend(self.val_data)
        if include_test:
            all_samples.extend(self.test_data)
        
        # Parallel path extraction
        def extract_paths(sample: 'KGSample') -> List[str]:
            return sample.get_path_strings()
        
        path_set = set()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            path_lists = list(tqdm(
                executor.map(extract_paths, all_samples),
                total=len(all_samples),
                desc="Extracting paths"
            ))
            
            for paths in path_lists:
                path_set.update(paths)
        
        self._all_paths = sorted(list(path_set))
        self._path_to_idx = {path: idx for idx, path in enumerate(self._all_paths)}
        
        elapsed = time.time() - start_time
        print(f"Built corpus: {len(self._all_paths)} unique paths in {elapsed:.2f}s")
        
        # Save to cache
        if cache_dir:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'paths': self._all_paths,
                        'path_to_idx': self._path_to_idx
                    }, f)
                print(f"  Cached to {cache_file}")
            except Exception as e:
                print(f"  Cache save failed: {e}")
        
        return self._all_paths, self._path_to_idx
    
    def get_candidate_pool_for_sample(
        self, 
        sample: KGSample, 
        all_paths: List[str],
        num_negatives: int = 99
    ) -> Tuple[List[str], List[int]]:
        """
        Get a candidate pool for a sample (ground truth + negatives).
        
        Returns:
            candidates: List of candidate path strings
            labels: Binary labels (1 = ground truth, 0 = negative)
        """
        import random
        
        gt_paths = set(sample.get_path_strings())
        negatives = [p for p in all_paths if p not in gt_paths]
        
        # Sample negatives
        if len(negatives) > num_negatives:
            negatives = random.sample(negatives, num_negatives)
        
        candidates = list(gt_paths) + negatives
        labels = [1] * len(gt_paths) + [0] * len(negatives)
        
        # Shuffle
        combined = list(zip(candidates, labels))
        random.shuffle(combined)
        candidates, labels = zip(*combined) if combined else ([], [])
        
        return list(candidates), list(labels)
