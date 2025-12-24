"""
RAG Pipeline: End-to-end retrieval and reranking for KGQA.
Combines hybrid retrieval with Qwen3 reranking for SOTA performance.
"""
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import time

from .config import RAGConfig
from .data_loader import KGDataLoader, KGSample
from .retriever import HybridRetriever
from .reranker import Qwen3Reranker


@dataclass
class RetrievalResult:
    """Result from the RAG pipeline for a single query."""
    question_id: str
    question: str
    retrieved_paths: List[str]  # After retrieval
    retrieved_scores: List[float]
    reranked_paths: List[str]  # After reranking
    reranked_scores: List[float]
    ground_truth_paths: List[str]
    retrieval_time_ms: float
    rerank_time_ms: float
    
    @property
    def total_time_ms(self) -> float:
        return self.retrieval_time_ms + self.rerank_time_ms


class RAGPipeline:
    """
    Complete RAG pipeline for Knowledge Graph Question Answering.
    
    Pipeline stages:
    1. Hybrid Retrieval (BM25 + Dense) -> top-K candidates
    2. Qwen3 Reranking -> final top-k results
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.data_loader = KGDataLoader(config)
        self.retriever = HybridRetriever(config)
        self.reranker = Qwen3Reranker(config)
        
        self._indexed = False
        self._corpus = None
    
    def build_index(self, include_test_paths: bool = False):
        """Build retrieval index from training data."""
        print("Building retrieval index...")
        
        # Get all unique paths from training data
        self._corpus, _ = self.data_loader.build_path_corpus(include_test=include_test_paths)
        
        # Build hybrid index
        self.retriever.index(self._corpus)
        self._indexed = True
        
        print(f"Index built with {len(self._corpus)} paths")
    
    def retrieve_and_rerank(
        self,
        question: str,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
    ) -> Tuple[List[str], List[float], float, float]:
        """
        Run full pipeline: retrieve then rerank.
        
        Returns:
            paths: Final reranked paths
            scores: Final scores
            retrieval_time_ms: Time for retrieval
            rerank_time_ms: Time for reranking
        """
        if not self._indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        if top_k_retrieve is None:
            top_k_retrieve = self.config.top_k_retrieve
        if top_k_rerank is None:
            top_k_rerank = self.config.top_k_rerank
        
        # Stage 1: Hybrid Retrieval
        start_time = time.time()
        _, _, retrieved_docs = self.retriever.retrieve(question, top_k=top_k_retrieve)
        retrieval_time_ms = (time.time() - start_time) * 1000
        
        # Stage 2: Qwen3 Reranking
        start_time = time.time()
        _, scores, reranked_docs = self.reranker.rerank(
            question, retrieved_docs, top_k=top_k_rerank
        )
        rerank_time_ms = (time.time() - start_time) * 1000
        
        return reranked_docs, scores, retrieval_time_ms, rerank_time_ms
    
    def process_sample(self, sample: KGSample) -> RetrievalResult:
        """Process a single sample through the pipeline."""
        paths, scores, ret_time, rerank_time = self.retrieve_and_rerank(sample.question)
        
        return RetrievalResult(
            question_id=sample.question_id,
            question=sample.question,
            retrieved_paths=[],  # We can add this if needed for analysis
            retrieved_scores=[],
            reranked_paths=paths,
            reranked_scores=scores,
            ground_truth_paths=sample.get_path_strings(),
            retrieval_time_ms=ret_time,
            rerank_time_ms=rerank_time,
        )
    
    def process_dataset(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[RetrievalResult]:
        """
        Process an entire dataset split.
        
        Args:
            split: "train", "val", or "test"
            limit: Maximum samples to process (for debugging)
            show_progress: Show progress bar
        
        Returns:
            List of RetrievalResult objects
        """
        if not self._indexed:
            self.build_index()
        
        # Load reranker
        self.reranker.load()
        
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
    
    def retrieval_only(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Run retrieval only (without reranking) for speed comparison.
        """
        if not self._indexed:
            self.build_index()
        
        if top_k is None:
            top_k = self.config.top_k_retrieve
        
        if split == "train":
            samples = self.data_loader.train_data
        elif split == "val":
            samples = self.data_loader.val_data
        else:
            samples = self.data_loader.test_data
        
        if limit:
            samples = samples[:limit]
        
        results = []
        for sample in tqdm(samples, desc=f"Retrieval only ({split})"):
            start_time = time.time()
            _, scores, docs = self.retriever.retrieve(sample.question, top_k=top_k)
            ret_time = (time.time() - start_time) * 1000
            
            result = RetrievalResult(
                question_id=sample.question_id,
                question=sample.question,
                retrieved_paths=docs,
                retrieved_scores=scores,
                reranked_paths=docs,  # Same as retrieved (no reranking)
                reranked_scores=scores,
                ground_truth_paths=sample.get_path_strings(),
                retrieval_time_ms=ret_time,
                rerank_time_ms=0.0,
            )
            results.append(result)
        
        return results
