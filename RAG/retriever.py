"""
Hybrid Retriever: BM25 + Dense retrieval for fast initial candidate generation.
Uses FAISS for efficient dense retrieval and rank_bm25 for sparse retrieval.
"""
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import pickle
import hashlib
from tqdm import tqdm

from .config import RAGConfig


class BM25Retriever:
    """BM25 sparse retriever using rank_bm25."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercasing tokenization."""
        return text.lower().replace("->", " ").replace("_", " ").split()
    
    def index(self, corpus: List[str]):
        """Build BM25 index over corpus."""
        from rank_bm25 import BM25Okapi
        
        self.corpus = corpus
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
    
    def retrieve(self, query: str, top_k: int = 100) -> Tuple[List[int], List[float]]:
        """Retrieve top-k documents for a query."""
        if self.bm25 is None:
            raise ValueError("Index not built. Call index() first.")
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        return top_indices.tolist(), top_scores.tolist()


class DenseRetriever:
    """Dense retriever using sentence transformers and FAISS."""
    
    def __init__(self, model_name: str, device: str = "cuda", use_cache: bool = True, cache_dir: Path = None):
        self.model_name = model_name
        self.device = device
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.model = None
        self.index = None
        self.corpus = None
        self.embeddings = None
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
    
    def _get_cache_path(self, corpus_hash: str) -> Path:
        return self.cache_dir / f"dense_index_{corpus_hash}.pkl"
    
    def _compute_corpus_hash(self, corpus: List[str]) -> str:
        content = "\n".join(corpus[:1000])  # Use first 1000 for hash
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def index_corpus(self, corpus: List[str], batch_size: int = 256):
        """Build dense index over corpus."""
        import faiss
        
        self.corpus = corpus
        corpus_hash = self._compute_corpus_hash(corpus)
        cache_path = self._get_cache_path(corpus_hash) if self.cache_dir else None
        
        # Try loading from cache
        if self.use_cache and cache_path and cache_path.exists():
            print(f"Loading dense index from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
                self.embeddings = cached["embeddings"]
                self.index = cached["index"]
                return
        
        print(f"Building dense index with {self.model_name}...")
        self._load_model()
        
        # Encode corpus in batches
        all_embeddings = []
        for i in tqdm(range(0, len(corpus), batch_size), desc="Encoding corpus"):
            batch = corpus[i:i + batch_size]
            embeddings = self.model.encode(
                batch, 
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # For cosine similarity
            )
            all_embeddings.append(embeddings)
        
        self.embeddings = np.vstack(all_embeddings).astype(np.float32)
        
        # Build FAISS index (using Inner Product for normalized vectors = cosine sim)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product
        self.index.add(self.embeddings)
        
        # Cache the index
        if self.use_cache and cache_path:
            print(f"Caching dense index to: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump({"embeddings": self.embeddings, "index": self.index}, f)
    
    def retrieve(self, query: str, top_k: int = 100) -> Tuple[List[int], List[float]]:
        """Retrieve top-k documents for a query."""
        if self.index is None:
            raise ValueError("Index not built. Call index_corpus() first.")
        
        self._load_model()
        
        # Encode query
        query_embedding = self.model.encode(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        return indices[0].tolist(), scores[0].tolist()
    
    def retrieve_batch(self, queries: List[str], top_k: int = 100) -> List[Tuple[List[int], List[float]]]:
        """Batch retrieve for multiple queries."""
        if self.index is None:
            raise ValueError("Index not built. Call index_corpus() first.")
        
        self._load_model()
        
        # Encode all queries
        query_embeddings = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype(np.float32)
        
        # Batch search
        all_scores, all_indices = self.index.search(query_embeddings, top_k)
        
        results = []
        for indices, scores in zip(all_indices, all_scores):
            results.append((indices.tolist(), scores.tolist()))
        
        return results


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and Dense retrieval.
    Uses Reciprocal Rank Fusion (RRF) for score combination.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.bm25_retriever = BM25Retriever(k1=config.bm25_k1, b=config.bm25_b)
        self.dense_retriever = DenseRetriever(
            model_name=config.dense_model_name,
            device=config.device,
            use_cache=config.use_cache,
            cache_dir=config.cache_dir
        )
        self.corpus = None
        self.alpha = config.hybrid_alpha  # Weight for dense (1-alpha for BM25)
    
    def index(self, corpus: List[str]):
        """Build both sparse and dense indices."""
        self.corpus = corpus
        print("Building BM25 index...")
        self.bm25_retriever.index(corpus)
        print("Building dense index...")
        self.dense_retriever.index_corpus(corpus)
        print("Hybrid index ready!")
    
    def _reciprocal_rank_fusion(
        self, 
        results_list: List[Tuple[List[int], List[float]]], 
        k: int = 60
    ) -> Tuple[List[int], List[float]]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion.
        
        Args:
            results_list: List of (indices, scores) tuples from different retrievers
            k: Constant for RRF (default 60 as per original paper)
        
        Returns:
            Combined (indices, scores) sorted by fused score
        """
        doc_scores: Dict[int, float] = {}
        
        for indices, _ in results_list:
            for rank, doc_id in enumerate(indices):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += 1.0 / (k + rank + 1)
        
        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        indices = [doc_id for doc_id, _ in sorted_docs]
        scores = [score for _, score in sorted_docs]
        
        return indices, scores
    
    def _weighted_fusion(
        self,
        bm25_results: Tuple[List[int], List[float]],
        dense_results: Tuple[List[int], List[float]],
    ) -> Tuple[List[int], List[float]]:
        """
        Combine results using weighted score fusion.
        Normalizes scores to [0, 1] range before combining.
        """
        def normalize_scores(scores: List[float]) -> List[float]:
            if not scores:
                return scores
            min_s, max_s = min(scores), max(scores)
            if max_s - min_s < 1e-9:
                return [1.0] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]
        
        bm25_indices, bm25_scores = bm25_results
        dense_indices, dense_scores = dense_results
        
        # Normalize
        bm25_scores_norm = normalize_scores(bm25_scores)
        dense_scores_norm = normalize_scores(dense_scores)
        
        # Create score maps
        doc_scores: Dict[int, float] = {}
        
        for idx, score in zip(bm25_indices, bm25_scores_norm):
            doc_scores[idx] = (1 - self.alpha) * score
        
        for idx, score in zip(dense_indices, dense_scores_norm):
            if idx in doc_scores:
                doc_scores[idx] += self.alpha * score
            else:
                doc_scores[idx] = self.alpha * score
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        indices = [doc_id for doc_id, _ in sorted_docs]
        scores = [score for _, score in sorted_docs]
        
        return indices, scores
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = None,
        fusion_method: str = "rrf"  # "rrf" or "weighted"
    ) -> Tuple[List[int], List[float], List[str]]:
        """
        Retrieve top-k candidates using hybrid retrieval.
        
        Returns:
            indices: Document indices
            scores: Fusion scores
            documents: Actual document strings
        """
        if top_k is None:
            top_k = self.config.top_k_retrieve
        
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)  # Over-retrieve for fusion
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        
        # Fuse results
        if fusion_method == "rrf":
            indices, scores = self._reciprocal_rank_fusion([bm25_results, dense_results])
        else:
            indices, scores = self._weighted_fusion(bm25_results, dense_results)
        
        # Trim to top_k
        indices = indices[:top_k]
        scores = scores[:top_k]
        documents = [self.corpus[i] for i in indices]
        
        return indices, scores, documents
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = None,
    ) -> List[Tuple[List[int], List[float], List[str]]]:
        """Batch retrieve for multiple queries."""
        results = []
        for query in tqdm(queries, desc="Hybrid retrieval"):
            results.append(self.retrieve(query, top_k))
        return results
