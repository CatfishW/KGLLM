"""
Combined QA Pipeline for KGQA.

Combines diffusion path generation with reranker-based RAG for state-of-the-art
question answering on knowledge graphs.

4-Stage Pipeline:
1. RAG Retrieval: BM25+Dense hybrid -> top-K candidate paths from corpus
2. Diffusion Generation: Novel paths from trained diffusion model  
3. Reranking: Qwen3-Reranker scores all candidate paths
4. LLM Answer: Generate final answer using top paths + question
"""
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
from tqdm import tqdm

from .config import RAGConfig
from .data_loader import KGDataLoader, KGSample
from .retriever import HybridRetriever
from .reranker import Qwen3Reranker
from .llm_client import LLMClient, LLMConfig
from .diffusion_generator import DiffusionPathGenerator, DiffusionConfig


@dataclass
class CombinedConfig(RAGConfig):
    """Extended configuration for the combined pipeline."""
    
    # Diffusion settings
    diffusion_checkpoint: str = "/data/Yanlai/KGLLM/Core/diffusion_100M/checkpoints/last.ckpt"
    diffusion_vocab: str = "/data/Yanlai/KGLLM/Core/diffusion_100M/vocab.json"
    num_diffusion_paths: int = 5
    diffusion_temperature: float = 1.0
    diffusion_path_length: int = 6
    
    # LLM settings
    llm_api_url: str = "https://game.agaii.org/llm/v1"
    llm_model: str = ""  # Auto-detect
    llm_temperature: float = 0.7
    llm_max_tokens: int = 256
    
    # Combined pipeline settings
    use_diffusion: bool = True
    use_rag: bool = True
    use_llm_answer: bool = True
    combine_paths: bool = True  # Merge retrieved + generated paths
    max_paths_for_rerank: int = 100  # Max paths to send to reranker
    max_paths_for_llm: int = 5  # Top paths to send to LLM
    
    # Entity retrieval (optional)
    use_entity_retrieval: bool = False
    graph_path: Optional[str] = None


@dataclass
class CombinedResult:
    """Result from the combined QA pipeline."""
    question_id: str
    question: str
    topic_entity: Optional[str]
    
    # Path retrieval results
    retrieved_paths: List[str]
    generated_paths: List[str]
    combined_paths: List[str]
    reranked_paths: List[str]
    reranked_scores: List[float]
    
    # Ground truth
    ground_truth_paths: List[str]
    ground_truth_answers: List[str]
    
    # Answer generation
    generated_answer: str
    
    # Timing
    retrieval_time_ms: float
    generation_time_ms: float
    rerank_time_ms: float
    llm_time_ms: float
    
    @property
    def total_time_ms(self) -> float:
        return self.retrieval_time_ms + self.generation_time_ms + self.rerank_time_ms + self.llm_time_ms


class CombinedQAPipeline:
    """
    Combined QA pipeline for Knowledge Graph Question Answering.
    
    Pipeline stages:
    1. RAG Retrieval (BM25 + Dense) -> top-K candidate paths
    2. Diffusion Generation -> diverse novel paths
    3. Path Pool -> merge retrieved + generated paths
    4. Qwen3 Reranking -> score and rank all paths
    5. LLM Answer Generation -> final answer using top paths
    """
    
    def __init__(self, config: CombinedConfig):
        self.config = config
        
        # Data loader
        self.data_loader = KGDataLoader(config)
        
        # RAG components
        if config.use_rag:
            self.retriever = HybridRetriever(config)
        else:
            self.retriever = None
        
        self.reranker = Qwen3Reranker(config)
        
        # Diffusion generator
        if config.use_diffusion:
            diffusion_config = DiffusionConfig(
                checkpoint_path=config.diffusion_checkpoint,
                vocab_path=config.diffusion_vocab,
                device=config.device,
                num_paths=config.num_diffusion_paths,
                path_length=config.diffusion_path_length,
                temperature=config.diffusion_temperature
            )
            self.diffusion_generator = DiffusionPathGenerator(diffusion_config)
        else:
            self.diffusion_generator = None
        
        # LLM client
        if config.use_llm_answer or getattr(config, 'use_hyde', False):
            llm_config = LLMConfig(
                api_url=config.llm_api_url,
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
            self.llm_client = LLMClient(llm_config)
        else:
            self.llm_client = None
        
        self._indexed = False
        self._corpus = None
    
    def build_index(self, include_test_paths: bool = None):
        """Build retrieval index from training data."""
        if self.retriever is None:
            print("[CombinedPipeline] RAG retrieval disabled, skipping index build")
            self._indexed = True
            return
        
        # Default to config if None
        if include_test_paths is None:
             include_test_paths = getattr(self.config, 'include_test_in_corpus', False)

        print(f"[CombinedPipeline] Building retrieval index (include_test={include_test_paths})...")
        self._corpus, _ = self.data_loader.build_path_corpus(include_test=include_test_paths)
        self.retriever.index(self._corpus)
        self._indexed = True
        print(f"[CombinedPipeline] Index built with {len(self._corpus)} paths")
    
    def _augment_query_with_hyde(self, question: str) -> str:
        """Generate a hypothetical path to augment the query (HyDE)."""
        if not self.llm_client:
            return question
            
        messages = [
            {'role': 'system', 'content': 'Predict the likely Knowledge Graph path (Entities and Relations) that answers the question. Be concise.'},
            {'role': 'user', 'content': f'Question: {question}\nPrediction:'}
        ]
        # Use a slightly lower temp for stability
        hypo_path = self.llm_client.chat(messages, max_tokens=32, temperature=0.3).strip()
        return f"{question} Expected Path: {hypo_path}"

    def _expand_query(self, question: str) -> str:
        """Expand query with relation keywords for better path retrieval."""
        expansion_map = {
            ('speak', 'language', 'languages'): 'language_spoken language',
            ('born', 'from', 'birthplace', 'hometown', 'where'): 'place_of_birth birthplace location',
            ('died', 'death'): 'place_of_death death',
            ('religion', 'religious'): 'religion religious_belief',
            ('president', 'leader', 'ruler', 'govern'): 'president leader government head_of_government',
            ('founded', 'founder', 'created'): 'founder organization_founded',
            ('married', 'spouse', 'wife', 'husband'): 'spouse marriage partner',
            ('capital',): 'capital administrative_capital',
            ('team', 'play'): 'team sports_team championships',
            ('wrote', 'author', 'book', 'written'): 'works_written author compositions',
            ('profession', 'job', 'occupation', 'do', 'did', 'work'): 'profession occupation employment',
            ('star', 'actor', 'film', 'movie'): 'film actor performance',
            ('country', 'nation'): 'country nationality',
        }
        
        q_lower = question.lower()
        expansions = []
        
        for keywords, relations in expansion_map.items():
            if any(kw in q_lower for kw in keywords):
                expansions.append(relations)
        
        if expansions:
            return question + ' ' + ' '.join(expansions)
        return question
    
    def _retrieve_paths(self, query: str) -> Tuple[List[str], float]:
        """Stage 1: Retrieve candidate paths using hybrid retrieval."""
        # Optimization: If corpus is small, skip retrieval and let Reranker handle everything
        # Reranker is significantly more accurate than BM25/Dense
        if self._corpus and len(self._corpus) <= 1000:
            return self._corpus, 0.0

        if self.retriever is None or not self.config.use_rag:
            return [], 0.0
        
        start_time = time.time()
        # Note: query is already augmented/expanded if needed
        _, _, retrieved_docs = self.retriever.retrieve(
            query, 
            top_k=self.config.top_k_retrieve
        )
        retrieval_time = (time.time() - start_time) * 1000
        
        return retrieved_docs, retrieval_time
    
    def _generate_paths(self, question: str) -> Tuple[List[str], float]:
        """Stage 2: Generate novel paths using diffusion model."""
        if self.diffusion_generator is None or not self.config.use_diffusion:
            return [], 0.0
        
        start_time = time.time()
        generated_paths = self.diffusion_generator.generate(
            question,
            num_paths=self.config.num_diffusion_paths,
            path_length=self.config.diffusion_path_length,
            temperature=self.config.diffusion_temperature
        )
        generation_time = (time.time() - start_time) * 1000
        
        return generated_paths, generation_time
    
    def _combine_paths(
        self, 
        retrieved: List[str], 
        generated: List[str]
    ) -> List[str]:
        """Combine and deduplicate paths from retrieval and generation."""
        if not self.config.combine_paths:
            return retrieved + generated
        
        # Deduplicate while preserving order
        seen = set()
        combined = []
        
        # Add retrieved first (usually higher quality for in-corpus)
        for path in retrieved:
            normalized = path.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                combined.append(path)
        
        # Add generated (novel paths not in corpus)
        for path in generated:
            normalized = path.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                combined.append(path)
        
        # Limit to max paths for reranker
        return combined[:self.config.max_paths_for_rerank]
    
    def _rerank_paths(
        self, 
        question: str, 
        paths: List[str]
    ) -> Tuple[List[str], List[float], float]:
        """Stage 3: Rerank all candidate paths."""
        if not paths:
            return [], [], 0.0
        
        start_time = time.time()
        _, scores, reranked_paths = self.reranker.rerank(
            question, 
            paths, 
            top_k=self.config.top_k_rerank
        )
        rerank_time = (time.time() - start_time) * 1000
        
        return reranked_paths, scores, rerank_time
    
    def _generate_answer(
        self, 
        question: str, 
        paths: List[str],
        topic_entity: Optional[str] = None
    ) -> Tuple[str, float]:
        """Stage 4: Generate final answer using LLM."""
        if self.llm_client is None or not self.config.use_llm_answer:
            return "", 0.0
        
        # Limit paths for LLM context
        top_paths = paths[:self.config.max_paths_for_llm]
        
        start_time = time.time()
        answer = self.llm_client.generate_answer(
            question=question,
            paths=top_paths,
            topic_entity=topic_entity,
            max_tokens=self.config.llm_max_tokens
        )
        llm_time = (time.time() - start_time) * 1000
        
        return answer, llm_time
    
    def process_sample(self, sample: KGSample) -> CombinedResult:
        """Process a single sample through the full pipeline."""
        # HyDE Augmentation
        search_query = sample.question
        if getattr(self.config, 'use_hyde', False):
            search_query = self._augment_query_with_hyde(sample.question)

        # Stage 1: Retrieve
        retrieved_paths, ret_time = self._retrieve_paths(search_query)
        
        # Stage 2: Generate
        generated_paths, gen_time = self._generate_paths(sample.question)
        
        # Combine paths
        combined_paths = self._combine_paths(retrieved_paths, generated_paths)
        
        # Stage 3: Rerank
        reranked_paths, scores, rerank_time = self._rerank_paths(
            search_query, 
            combined_paths
        )
        
        # Stage 4: Extract entities from graph - path-based with 1-hop fallback
        candidate_entities = []
        if sample.graph and sample.topic_entity:
            # First try: extract using ranked paths (high precision)
            for path in reranked_paths[:10]:  # Increase to top 10 paths
                entities = sample.extract_entities_for_path(path)
                candidate_entities.extend(entities)
            
            # Deduplicate
            seen = set()
            candidate_entities = [e for e in candidate_entities 
                                 if not (e in seen or seen.add(e))]
            
            # Fallback: if no entities from paths, try 1-hop with question filtering
            if len(candidate_entities) < 3:
                # Build graph index
                graph_by_subject = {}
                for triple in sample.graph:
                    if len(triple) >= 3:
                        subj, rel, obj = triple[0], triple[1], triple[2]
                        if subj not in graph_by_subject:
                            graph_by_subject[subj] = []
                        graph_by_subject[subj].append((rel, obj))
                
                # Get 1-hop entities, filter by question keywords
                question_lower = sample.question.lower()
                
                # Define question-relation mappings
                keyword_rel_map = {
                    ('language', 'speak'): 'language',
                    ('born', 'from', 'birthplace'): 'birth',
                    ('died', 'death'): 'death',
                    ('president',): 'president',
                    ('capital',): 'capital',
                    ('religion',): 'religion',
                    ('founder', 'founded'): 'founder',
                    ('team',): 'team',
                }
                
                # Find matching relation keywords
                matching_rels = []
                for keywords, rel_pattern in keyword_rel_map.items():
                    if any(kw in question_lower for kw in keywords):
                        matching_rels.append(rel_pattern)
                
                if sample.topic_entity in graph_by_subject:
                    for rel, obj in graph_by_subject[sample.topic_entity]:
                        if obj.startswith('m.') or obj.startswith('g.'):
                            continue
                        
                        rel_lower = rel.lower()
                        # Check if relation matches question
                        for pattern in matching_rels:
                            if pattern in rel_lower:
                                if obj not in seen:
                                    candidate_entities.append(obj)
                                    seen.add(obj)
                                break
                        
                        if len(candidate_entities) >= 20:
                            break
        
        # Generate answer with entity candidates if available
        start_time = time.time()
        if candidate_entities:
            answer = self.llm_client.generate_answer_with_entities(
                question=sample.question,
                paths=reranked_paths[:self.config.max_paths_for_llm],
                candidate_entities=candidate_entities[:25],
                topic_entity=sample.topic_entity,
                max_tokens=self.config.llm_max_tokens
            )
        else:
            answer = self.llm_client.generate_answer(
                question=sample.question,
                paths=reranked_paths[:self.config.max_paths_for_llm],
                topic_entity=sample.topic_entity,
                max_tokens=self.config.llm_max_tokens
            )
        llm_time = (time.time() - start_time) * 1000
        
        return CombinedResult(
            question_id=sample.question_id,
            question=sample.question,
            topic_entity=sample.topic_entity,
            retrieved_paths=retrieved_paths[:20],  # Limit for storage
            generated_paths=generated_paths,
            combined_paths=combined_paths[:20],
            reranked_paths=reranked_paths,
            reranked_scores=scores,
            ground_truth_paths=sample.get_path_strings(),
            ground_truth_answers=sample.answer_entities or [],
            generated_answer=answer,
            retrieval_time_ms=ret_time,
            generation_time_ms=gen_time,
            rerank_time_ms=rerank_time,
            llm_time_ms=llm_time
        )
    
    def process_dataset(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        show_progress: bool = True
    ) -> List[CombinedResult]:
        """
        Process an entire dataset split.
        
        Args:
            split: "train", "val", or "test"
            limit: Maximum samples to process
            show_progress: Show progress bar
            
        Returns:
            List of CombinedResult objects
        """
        # Build index if needed
        if not self._indexed:
            self.build_index()
        
        # Load reranker
        self.reranker.load()
        
        # Pre-load diffusion generator (vocab is 36MB, takes time)
        if self.diffusion_generator:
            print("[CombinedPipeline] Loading diffusion generator...")
            self.diffusion_generator.load()
            print("[CombinedPipeline] Diffusion generator loaded")
        
        # Get data
        if split == "train":
            samples = self.data_loader.train_data
        elif split == "val":
            samples = self.data_loader.val_data
        else:
            samples = self.data_loader.test_data
        
        if limit:
            samples = samples[:limit]
        
        results = []
        iterator = samples
        if show_progress:
            iterator = tqdm(samples, desc=f"Processing {split}")
        
        for sample in iterator:
            result = self.process_sample(sample)
            results.append(result)
        
        return results
    
    def unload(self):
        """Unload models to free GPU memory."""
        self.reranker.unload()
        if self.diffusion_generator:
            self.diffusion_generator.unload()
        print("[CombinedPipeline] Models unloaded")
