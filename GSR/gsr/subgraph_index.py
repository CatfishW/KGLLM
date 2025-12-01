"""
Subgraph Index Builder and Lookup System for GSR-style Subgraph Retrieval.

Builds an index of relation patterns (subgraph IDs) from training data paths.
Each subgraph ID represents a relation sequence pattern that can retrieve
relevant subgraphs for answering questions.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class SubgraphPattern:
    """Represents a subgraph pattern with its relation sequence."""
    subgraph_id: str
    relations: List[str]  # Ordered list of relations
    relation_pattern: str  # Human-readable pattern (e.g., "r1|r2|r3")
    example_count: int  # Number of examples using this pattern
    example_triples: List[Tuple[str, str, str]]  # Sample triples following this pattern
    answer_types: Set[str]  # Types of answers this pattern produces


class SubgraphIndex:
    """Index of subgraph patterns for efficient retrieval."""
    
    def __init__(self):
        self.patterns: Dict[str, SubgraphPattern] = {}
        self.relation_to_patterns: Dict[str, List[str]] = defaultdict(list)
        self.pattern_counter: Counter = Counter()
    
    def add_path(
        self,
        relations: List[str],
        triples: List[Tuple[str, str, str]],
        answer_entity: Optional[str] = None
    ) -> str:
        """
        Add a path to the index and return its subgraph ID.
        
        Args:
            relations: List of relation strings (e.g., ["people.person.sibling_s"])
            triples: List of (subject, relation, object) triples
            answer_entity: The answer entity in the path (optional)
        
        Returns:
            subgraph_id: The ID for this pattern
        """
        # Create subgraph ID from relation sequence
        subgraph_id = self._create_subgraph_id(relations)
        
        # Create or update pattern
        if subgraph_id not in self.patterns:
            self.patterns[subgraph_id] = SubgraphPattern(
                subgraph_id=subgraph_id,
                relations=relations,
                relation_pattern="|".join(relations),
                example_count=0,
                example_triples=[],
                answer_types=set()
            )
        
        # Update pattern
        pattern = self.patterns[subgraph_id]
        pattern.example_count += 1
        
        # Store example triples (keep up to 10 examples)
        if len(pattern.example_triples) < 10:
            pattern.example_triples.append(triples[0] if triples else None)
        
        # Index by individual relations for fast lookup
        for rel in relations:
            if subgraph_id not in self.relation_to_patterns[rel]:
                self.relation_to_patterns[rel].append(subgraph_id)
        
        self.pattern_counter[subgraph_id] += 1
        
        return subgraph_id
    
    def _create_subgraph_id(self, relations: List[str]) -> str:
        """
        Create a subgraph ID from a relation sequence.
        
        Format: "path_" + hash of relation pattern (for uniqueness)
        Or: "path_" + sanitized relation names (human-readable)
        """
        if not relations:
            return "path_empty"
        
        # Option 1: Human-readable format (relation names)
        sanitized = [rel.replace(".", "_") for rel in relations]
        readable_id = "path_" + "_".join(sanitized)
        
        # Option 2: Hash-based format (for very long patterns)
        if len(readable_id) > 100:
            pattern_str = "|".join(relations)
            hash_obj = hashlib.md5(pattern_str.encode())
            hash_id = hash_obj.hexdigest()[:8]
            return f"path_{hash_id}"
        
        return readable_id
    
    def get_pattern(self, subgraph_id: str) -> Optional[SubgraphPattern]:
        """Retrieve a pattern by its ID."""
        return self.patterns.get(subgraph_id)
    
    def search_by_relations(self, relations: List[str], top_k: int = 10) -> List[SubgraphPattern]:
        """
        Search for patterns containing the given relations.
        
        Returns patterns sorted by relevance (number of matching relations).
        """
        pattern_scores = defaultdict(int)
        
        for rel in relations:
            for pattern_id in self.relation_to_patterns.get(rel, []):
                pattern_scores[pattern_id] += 1
        
        # Sort by score (number of matching relations)
        sorted_ids = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for pattern_id, score in sorted_ids[:top_k]:
            if pattern_id in self.patterns:
                results.append(self.patterns[pattern_id])
        
        return results
    
    def get_top_patterns(self, top_k: int = 100) -> List[SubgraphPattern]:
        """Get top-k most frequent patterns."""
        top_ids = [pattern_id for pattern_id, _ in self.pattern_counter.most_common(top_k)]
        return [self.patterns[pid] for pid in top_ids if pid in self.patterns]
    
    def save(self, path: str):
        """Save index to JSON file."""
        data = {
            "patterns": {
                pid: {
                    "subgraph_id": p.subgraph_id,
                    "relations": p.relations,
                    "relation_pattern": p.relation_pattern,
                    "example_count": p.example_count,
                    "example_triples": p.example_triples,
                    "answer_types": list(p.answer_types)
                }
                for pid, p in self.patterns.items()
            },
            "statistics": {
                "total_patterns": len(self.patterns),
                "total_examples": sum(p.example_count for p in self.patterns.values())
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved subgraph index to {path}")
        print(f"  Total patterns: {len(self.patterns)}")
        print(f"  Total examples: {sum(p.example_count for p in self.patterns.values())}")
    
    @classmethod
    def load(cls, path: str) -> 'SubgraphIndex':
        """Load index from JSON file."""
        index = cls()
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for pid, p_data in data["patterns"].items():
            pattern = SubgraphPattern(
                subgraph_id=p_data["subgraph_id"],
                relations=p_data["relations"],
                relation_pattern=p_data["relation_pattern"],
                example_count=p_data["example_count"],
                example_triples=[tuple(t) if isinstance(t, list) else t for t in p_data["example_triples"]],
                answer_types=set(p_data.get("answer_types", []))
            )
            index.patterns[pid] = pattern
            
            # Rebuild relation index
            for rel in pattern.relations:
                if pid not in index.relation_to_patterns[rel]:
                    index.relation_to_patterns[rel].append(pid)
            
            index.pattern_counter[pid] = pattern.example_count
        
        print(f"Loaded subgraph index from {path}")
        print(f"  Total patterns: {len(index.patterns)}")
        
        return index


def build_subgraph_index_from_dataset(
    data_path: str,
    output_path: str,
    min_pattern_frequency: int = 1
) -> SubgraphIndex:
    """
    Build subgraph index from dataset with paths.
    
    Args:
        data_path: Path to dataset (parquet or jsonl)
        output_path: Path to save the index
        min_pattern_frequency: Minimum number of examples for a pattern to be included
    
    Returns:
        SubgraphIndex: The built index
    """
    from pathlib import Path
    import pandas as pd
    import json
    
    index = SubgraphIndex()
    path = Path(data_path)
    
    # Load data
    if path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
        samples = df.to_dict('records')
        # Handle JSON-encoded fields
        for sample in samples:
            for key in ['graph', 'paths', 'answer', 'q_entity', 'a_entity']:
                if key in sample and isinstance(sample[key], str):
                    try:
                        sample[key] = json.loads(sample[key])
                    except:
                        pass
    elif path.suffix in ['.jsonl', '.json']:
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            if path.suffix == '.jsonl':
                for line in f:
                    samples.append(json.loads(line))
            else:
                samples = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    print(f"Building subgraph index from {len(samples)} samples...")
    
    # Process each sample
    for sample in samples:
        paths = sample.get('paths', [])
        graph = sample.get('graph', [])
        a_entities = sample.get('a_entity', sample.get('a_entities', []))
        
        if isinstance(a_entities, str):
            a_entities = [a_entities]
        
        # Process each path
        for path in paths:
            if isinstance(path, dict):
                relations = path.get('relations', [])
                entities = path.get('entities', [])
                
                if not relations:
                    continue
                
                # Extract triples from graph that match this path
                triples = []
                for i in range(len(entities) - 1):
                    if i < len(relations):
                        # Find matching triple in graph
                        for triple in graph:
                            if len(triple) == 3:
                                subj, rel, obj = triple[0], triple[1], triple[2]
                                if (str(subj) == str(entities[i]) and 
                                    str(rel) == str(relations[i]) and
                                    str(obj) == str(entities[i+1])):
                                    triples.append(triple)
                                    break
                
                # Add to index
                answer_entity = entities[-1] if entities else None
                subgraph_id = index.add_path(
                    relations=[str(r) for r in relations],
                    triples=triples,
                    answer_entity=str(answer_entity) if answer_entity else None
                )
    
    # Filter patterns by frequency
    if min_pattern_frequency > 1:
        filtered_patterns = {}
        for pid, pattern in index.patterns.items():
            if pattern.example_count >= min_pattern_frequency:
                filtered_patterns[pid] = pattern
        index.patterns = filtered_patterns
        print(f"Filtered to {len(filtered_patterns)} patterns (min frequency: {min_pattern_frequency})")
    
    # Save index
    if output_path:
        # Create output directory if it doesn't exist
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        index.save(output_path)
    
    return index


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build subgraph index from dataset')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset (parquet or jsonl)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save subgraph index')
    parser.add_argument('--min_frequency', type=int, default=1,
                       help='Minimum pattern frequency to include')
    
    args = parser.parse_args()
    
    index = build_subgraph_index_from_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        min_pattern_frequency=args.min_frequency
    )
    
    # Print statistics
    print("\n" + "="*60)
    print("SUBGRAPH INDEX STATISTICS")
    print("="*60)
    print(f"Total patterns: {len(index.patterns)}")
    
    # Show top patterns
    top_patterns = index.get_top_patterns(top_k=10)
    print(f"\nTop 10 patterns:")
    for i, pattern in enumerate(top_patterns, 1):
        print(f"  {i}. {pattern.subgraph_id}")
        print(f"     Relations: {pattern.relation_pattern}")
        print(f"     Examples: {pattern.example_count}")

