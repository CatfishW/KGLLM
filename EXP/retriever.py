"""
KG Path Retriever: RAG-style retrieval for relation paths.

This module implements the retrieval component of the KG-RAG system,
using FAISS for similarity search and optional re-ranking.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import RAGConfig, DEFAULT_CONFIG
from .path_indexer import PathIndex, PathMetadata


@dataclass
class RetrievedPath:
    """A retrieved relation path with metadata."""
    rank: int
    score: float
    relation_chain: str
    relations: List[str]
    natural_language: str
    example_questions: List[str]
    example_entities: List[List[str]]
    frequency: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rank': self.rank,
            'score': self.score,
            'relation_chain': self.relation_chain,
            'relations': self.relations,
            'natural_language': self.natural_language,
            'example_questions': self.example_questions,
            'example_entities': self.example_entities,
            'frequency': self.frequency
        }
    
    def __repr__(self) -> str:
        return f"RetrievedPath(rank={self.rank}, score={self.score:.4f}, chain='{self.relation_chain}')"


class KGPathRetriever:
    """
    RAG-style retriever for Knowledge Graph relation paths.
    
    Given a natural language question, retrieves the most relevant
    relation paths that could lead to the answer.
    
    Example:
        >>> retriever = KGPathRetriever("EXP/index")
        >>> paths = retriever.retrieve("who is obama's wife", top_k=5)
        >>> for path in paths:
        ...     print(f"{path.score:.3f}: {path.relation_chain}")
    """
    
    def __init__(
        self,
        index_path: str,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            index_path: Path to the FAISS index directory
            config: Optional configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.index_path = Path(index_path)
        
        # Load embedding model
        print(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        
        # Load FAISS index
        print(f"Loading index from: {self.index_path}")
        self.path_index = PathIndex(self.config)
        self.path_index.load(self.index_path)
        
        # Optional: Load re-ranker
        self.reranker = None
        if self.config.use_reranker:
            self._load_reranker()
    
    def _load_reranker(self):
        """Load cross-encoder for re-ranking."""
        try:
            from sentence_transformers import CrossEncoder
            print(f"Loading re-ranker: {self.config.reranker_model}")
            self.reranker = CrossEncoder(self.config.reranker_model)
        except Exception as e:
            print(f"Warning: Could not load re-ranker: {e}")
            self.reranker = None
    
    def encode_question(self, question: str) -> np.ndarray:
        """
        Encode a question to a dense vector.
        
        Args:
            question: Natural language question
            
        Returns:
            Normalized embedding vector
        """
        embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding
    
    def encode_questions(self, questions: List[str]) -> np.ndarray:
        """
        Encode multiple questions.
        
        Args:
            questions: List of questions
            
        Returns:
            Array of normalized embedding vectors
        """
        embeddings = self.embedding_model.encode(
            questions,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(questions) > 10
        )
        return embeddings
    
    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        relation_filter: Optional[List[str]] = None
    ) -> List[RetrievedPath]:
        """
        Retrieve relevant relation paths for a question.
        
        Args:
            question: Natural language question
            top_k: Number of paths to retrieve (default from config)
            min_score: Minimum similarity score (default from config)
            relation_filter: Optional list of relations to filter by
            
        Returns:
            List of RetrievedPath objects sorted by relevance
        """
        top_k = top_k or self.config.default_top_k
        min_score = min_score if min_score is not None else self.config.similarity_threshold
        
        # Get more candidates if using re-ranker
        search_k = self.config.rerank_top_k if self.reranker else top_k
        
        # Encode question
        query_embedding = self.encode_question(question)
        
        # Search FAISS index
        results = self.path_index.search(query_embedding, search_k)
        
        # Filter by score
        results = [(meta, score) for meta, score in results if score >= min_score]
        
        # Filter by relations if specified
        if relation_filter:
            relation_set = set(relation_filter)
            results = [
                (meta, score) for meta, score in results
                if any(rel in relation_set for rel in meta.relations)
            ]
        
        # Re-rank if enabled
        if self.reranker and len(results) > 0:
            results = self._rerank(question, results, top_k)
        else:
            results = results[:top_k]
        
        # Convert to RetrievedPath objects
        retrieved_paths = []
        for rank, (meta, score) in enumerate(results, 1):
            path = RetrievedPath(
                rank=rank,
                score=score,
                relation_chain=meta.relation_chain,
                relations=meta.relations,
                natural_language=meta.natural_language,
                example_questions=meta.example_questions,
                example_entities=meta.example_entities,
                frequency=meta.frequency
            )
            retrieved_paths.append(path)
        
        return retrieved_paths
    
    def _rerank(
        self,
        question: str,
        candidates: List[Tuple[PathMetadata, float]],
        top_k: int
    ) -> List[Tuple[PathMetadata, float]]:
        """
        Re-rank candidates using cross-encoder.
        
        Args:
            question: Original question
            candidates: List of (metadata, score) tuples
            top_k: Number of results to return
            
        Returns:
            Re-ranked list of (metadata, score) tuples
        """
        if not candidates or not self.reranker:
            return candidates[:top_k]
        
        # Create pairs for cross-encoder
        pairs = [
            (question, meta.natural_language)
            for meta, _ in candidates
        ]
        
        # Get cross-encoder scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine with original scores (optional: weighted average)
        combined = []
        for i, (meta, orig_score) in enumerate(candidates):
            # Use re-ranker score primarily, but keep high-frequency paths slightly boosted
            freq_boost = min(0.1, meta.frequency / 1000)  # Small boost for common paths
            final_score = float(rerank_scores[i]) + freq_boost
            combined.append((meta, final_score))
        
        # Sort by combined score
        combined.sort(key=lambda x: -x[1])
        
        return combined[:top_k]
    
    def batch_retrieve(
        self,
        questions: List[str],
        top_k: Optional[int] = None
    ) -> List[List[RetrievedPath]]:
        """
        Retrieve paths for multiple questions.
        
        Args:
            questions: List of questions
            top_k: Number of paths per question
            
        Returns:
            List of lists of RetrievedPath objects
        """
        top_k = top_k or self.config.default_top_k
        
        # Encode all questions
        query_embeddings = self.encode_questions(questions)
        
        all_results = []
        for i, embedding in enumerate(query_embeddings):
            results = self.path_index.search(embedding, top_k)
            
            retrieved_paths = []
            for rank, (meta, score) in enumerate(results, 1):
                if score >= self.config.similarity_threshold:
                    path = RetrievedPath(
                        rank=rank,
                        score=score,
                        relation_chain=meta.relation_chain,
                        relations=meta.relations,
                        natural_language=meta.natural_language,
                        example_questions=meta.example_questions,
                        example_entities=meta.example_entities,
                        frequency=meta.frequency
                    )
                    retrieved_paths.append(path)
            
            all_results.append(retrieved_paths)
        
        return all_results
    
    def retrieve_with_keywords(
        self,
        question: str,
        top_k: Optional[int] = None,
        boost_keywords: Optional[List[str]] = None
    ) -> List[RetrievedPath]:
        """
        Retrieve with keyword boosting for hybrid search.
        
        Args:
            question: Natural language question
            top_k: Number of paths to retrieve
            boost_keywords: Keywords to boost (e.g., entity types like "person", "film")
            
        Returns:
            List of RetrievedPath objects
        """
        top_k = top_k or self.config.default_top_k
        
        # Get semantic results
        results = self.retrieve(question, top_k=top_k * 2)
        
        if not boost_keywords:
            return results[:top_k]
        
        # Boost paths that contain keywords
        boosted = []
        for path in results:
            boost = 0.0
            chain_lower = path.relation_chain.lower()
            for kw in boost_keywords:
                if kw.lower() in chain_lower:
                    boost += 0.1
            
            boosted.append((path, path.score + boost))
        
        # Sort by boosted score
        boosted.sort(key=lambda x: -x[1])
        
        # Update scores and ranks
        final_results = []
        for rank, (path, new_score) in enumerate(boosted[:top_k], 1):
            path.rank = rank
            path.score = new_score
            final_results.append(path)
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_paths': len(self.path_index.metadata),
            'index_vectors': self.path_index.index.ntotal,
            'embedding_model': self.config.embedding_model,
            'index_type': self.config.index_type,
            'use_reranker': self.reranker is not None
        }
