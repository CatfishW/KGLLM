"""
KG-RAG Pipeline: End-to-end retrieval pipeline for Knowledge Graph Question Answering.

This module provides a high-level interface for the KG-RAG system,
handling question processing, path retrieval, and output formatting.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .config import RAGConfig, DEFAULT_CONFIG
from .retriever import KGPathRetriever, RetrievedPath


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    question: str
    retrieved_paths: List[RetrievedPath]
    top_relation_chain: Optional[str] = None
    top_score: Optional[float] = None
    
    def __post_init__(self):
        if self.retrieved_paths:
            self.top_relation_chain = self.retrieved_paths[0].relation_chain
            self.top_score = self.retrieved_paths[0].score
    
    @property
    def paths(self) -> List[RetrievedPath]:
        """Alias for retrieved_paths."""
        return self.retrieved_paths
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'question': self.question,
            'top_relation_chain': self.top_relation_chain,
            'top_score': self.top_score,
            'num_paths': len(self.retrieved_paths),
            'paths': [p.to_dict() for p in self.retrieved_paths]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def format_readable(self, max_examples: int = 2) -> str:
        """Format response for human reading."""
        lines = [
            f"Question: {self.question}",
            f"Retrieved {len(self.retrieved_paths)} paths:",
            ""
        ]
        
        for path in self.retrieved_paths:
            lines.append(f"  [{path.rank}] Score: {path.score:.4f}")
            lines.append(f"      Chain: {path.relation_chain}")
            lines.append(f"      Natural: {path.natural_language}")
            lines.append(f"      Frequency: {path.frequency} occurrences in training data")
            
            if path.example_questions and max_examples > 0:
                lines.append(f"      Example questions:")
                for eq in path.example_questions[:max_examples]:
                    lines.append(f"        - {eq}")
            lines.append("")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"RAGResponse(question='{self.question[:50]}...', paths={len(self.retrieved_paths)})"


class KGRAGPipeline:
    """
    End-to-end RAG pipeline for Knowledge Graph path retrieval.
    
    Example:
        >>> pipeline = KGRAGPipeline("EXP/index")
        >>> response = pipeline.ask("who is obama's wife")
        >>> print(response.format_readable())
    """
    
    def __init__(
        self,
        index_path: str = "EXP/index",
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            index_path: Path to the FAISS index directory
            config: Optional configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.index_path = Path(index_path)
        
        # Lazy loading of retriever
        self._retriever: Optional[KGPathRetriever] = None
    
    @property
    def retriever(self) -> KGPathRetriever:
        """Lazy load the retriever."""
        if self._retriever is None:
            self._retriever = KGPathRetriever(
                str(self.index_path),
                self.config
            )
        return self._retriever
    
    def preprocess_question(self, question: str) -> str:
        """
        Preprocess a question for retrieval.
        
        Args:
            question: Raw question string
            
        Returns:
            Cleaned question
        """
        # Lowercase
        question = question.lower().strip()
        
        # Remove extra whitespace
        question = re.sub(r'\s+', ' ', question)
        
        # Remove trailing punctuation (but keep question marks)
        question = re.sub(r'[.!]+$', '', question)
        
        return question
    
    def extract_keywords(self, question: str) -> List[str]:
        """
        Extract potential keywords for hybrid retrieval.
        
        Args:
            question: Question string
            
        Returns:
            List of keywords
        """
        keywords = []
        
        # Domain-specific keywords
        domain_patterns = {
            'person': ['who', 'person', 'people', 'born', 'married', 'wife', 'husband', 'father', 'mother', 'brother', 'sister', 'child', 'children', 'parent'],
            'location': ['where', 'country', 'city', 'state', 'located', 'capital', 'continent', 'island'],
            'film': ['movie', 'film', 'actor', 'actress', 'directed', 'starring', 'character', 'play'],
            'music': ['song', 'album', 'singer', 'band', 'music', 'wrote', 'composed'],
            'time': ['when', 'year', 'date', 'born', 'died', 'founded'],
            'government': ['president', 'leader', 'government', 'political', 'party', 'election'],
        }
        
        question_lower = question.lower()
        
        for domain, patterns in domain_patterns.items():
            if any(p in question_lower for p in patterns):
                keywords.append(domain)
        
        return keywords
    
    def ask(
        self,
        question: str,
        top_k: int = 5,
        use_keywords: bool = False,
        min_score: Optional[float] = None
    ) -> RAGResponse:
        """
        Ask a question and retrieve relevant paths.
        
        Args:
            question: Natural language question
            top_k: Number of paths to retrieve
            use_keywords: Use hybrid keyword-boosted retrieval
            min_score: Minimum similarity score threshold
            
        Returns:
            RAGResponse with retrieved paths
        """
        # Preprocess
        processed_question = self.preprocess_question(question)
        
        # Retrieve
        if use_keywords:
            keywords = self.extract_keywords(processed_question)
            paths = self.retriever.retrieve_with_keywords(
                processed_question,
                top_k=top_k,
                boost_keywords=keywords
            )
        else:
            paths = self.retriever.retrieve(
                processed_question,
                top_k=top_k,
                min_score=min_score
            )
        
        return RAGResponse(
            question=question,
            retrieved_paths=paths
        )
    
    def batch_ask(
        self,
        questions: List[str],
        top_k: int = 5
    ) -> List[RAGResponse]:
        """
        Ask multiple questions.
        
        Args:
            questions: List of questions
            top_k: Number of paths per question
            
        Returns:
            List of RAGResponse objects
        """
        # Preprocess all questions
        processed = [self.preprocess_question(q) for q in questions]
        
        # Batch retrieve
        all_paths = self.retriever.batch_retrieve(processed, top_k)
        
        # Create responses
        responses = []
        for q, paths in zip(questions, all_paths):
            responses.append(RAGResponse(question=q, retrieved_paths=paths))
        
        return responses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'index_path': str(self.index_path),
            **self.retriever.get_stats()
        }
    
    def interactive_demo(self):
        """Run interactive demo in terminal."""
        print("\n" + "="*60)
        print("  KG-RAG: Knowledge Graph Path Retrieval Demo")
        print("="*60)
        print(f"\nIndex loaded with {self.retriever.get_stats()['total_paths']} unique paths")
        print("\nEnter a question to retrieve relevant relation paths.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                question = input("Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                response = self.ask(question, top_k=5)
                print("\n" + response.format_readable())
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def create_pipeline(
    index_path: str = "EXP/index",
    **config_kwargs
) -> KGRAGPipeline:
    """
    Factory function to create a pipeline.
    
    Args:
        index_path: Path to the index
        **config_kwargs: Additional config options
        
    Returns:
        Configured KGRAGPipeline
    """
    config = RAGConfig(**config_kwargs) if config_kwargs else None
    return KGRAGPipeline(index_path, config)


# === Enhanced Pipeline with Toggleable Modules ===

@dataclass
class EnhancedRAGResponse(RAGResponse):
    """Enhanced response with additional metadata."""
    question_type: Optional[str] = None
    classification_confidence: Optional[float] = None
    filter_applied: bool = False
    diffusion_reranked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'question_type': self.question_type,
            'classification_confidence': self.classification_confidence,
            'filter_applied': self.filter_applied,
            'diffusion_reranked': self.diffusion_reranked
        })
        return base
    
    def format_readable(self, max_examples: int = 2) -> str:
        lines = []
        
        # Add enhanced metadata
        if self.question_type:
            lines.append(f"Question Type: {self.question_type} (confidence: {self.classification_confidence:.2f})")
        if self.filter_applied:
            lines.append("Subgraph Filter: Applied")
        if self.diffusion_reranked:
            lines.append("Diffusion Reranking: Applied")
        if lines:
            lines.append("")
        
        # Add base response
        lines.append(super().format_readable(max_examples))
        
        return "\n".join(lines)


class EnhancedKGRAGPipeline:
    """
    Enhanced RAG pipeline with toggleable modules:
    - Question Classifier: Classify questions by complexity type
    - Subgraph Filter: Filter paths by predicted hop count
    - Diffusion Ranker: Rerank paths using diffusion model
    
    Example:
        >>> config = RAGConfig(
        ...     use_question_classifier=True,
        ...     use_diffusion_ranker=True
        ... )
        >>> pipeline = EnhancedKGRAGPipeline("EXP/index", config)
        >>> response = pipeline.ask("who is obama's wife")
    """
    
    def __init__(
        self,
        index_path: str = "EXP/index",
        config: Optional[RAGConfig] = None
    ):
        self.config = config or DEFAULT_CONFIG
        self.index_path = Path(index_path)
        
        # Base retriever
        self._retriever: Optional[KGPathRetriever] = None
        
        # Enhanced modules (lazy loaded)
        self._question_classifier = None
        self._subgraph_filter = None
        self._diffusion_ranker = None
    
    @property
    def retriever(self) -> KGPathRetriever:
        """Lazy load retriever."""
        if self._retriever is None:
            self._retriever = KGPathRetriever(str(self.index_path), self.config)
        return self._retriever
    
    @property
    def question_classifier(self):
        """Lazy load question classifier."""
        if self._question_classifier is None:
            from .question_classifier import QuestionClassifier
            self._question_classifier = QuestionClassifier(
                mode=self.config.classifier_mode,
                model_path=self.config.classifier_model_path,
                enable=self.config.use_question_classifier
            )
        return self._question_classifier
    
    @property
    def subgraph_filter(self):
        """Lazy load subgraph filter."""
        if self._subgraph_filter is None:
            from .subgraph_filter import SubgraphFilter
            self._subgraph_filter = SubgraphFilter(
                classifier=self.question_classifier,
                enable=self.config.use_question_classifier,
                min_results=self.config.classifier_min_results,
                strict=self.config.classifier_strict_filter
            )
        return self._subgraph_filter
    
    @property
    def diffusion_ranker(self):
        """Lazy load diffusion ranker."""
        if self._diffusion_ranker is None:
            from .diffusion_ranker import DiffusionRanker
            self._diffusion_ranker = DiffusionRanker(
                model_path=self.config.diffusion_model_path,
                enable=self.config.use_diffusion_ranker,
                mode=self.config.diffusion_mode,
                num_diffusion_steps=self.config.diffusion_num_steps,
                score_weight=self.config.diffusion_score_weight,
                vocab_path=self.config.vocab_path
            )
        return self._diffusion_ranker
    
    def preprocess_question(self, question: str) -> str:
        """Preprocess question."""
        question = question.lower().strip()
        question = re.sub(r'\s+', ' ', question)
        question = re.sub(r'[.!]+$', '', question)
        return question
    
    def ask(
        self,
        question: str,
        top_k: int = 5,
        use_keywords: bool = False
    ) -> EnhancedRAGResponse:
        """
        Ask a question with enhanced processing.
        
        Pipeline:
        1. Retrieve candidate paths (more than top_k for filtering)
        2. Apply question classifier + subgraph filter (if enabled)
        3. Apply diffusion reranking (if enabled)
        4. Return top_k results
        """
        processed = self.preprocess_question(question)
        
        # Get extra candidates for filtering/reranking
        retrieve_k = top_k * 3 if (self.config.use_question_classifier or 
                                    self.config.use_diffusion_ranker) else top_k
        
        # Step 1: Initial retrieval
        paths = self.retriever.retrieve(processed, top_k=retrieve_k)
        
        # Track what modules were applied
        question_type = None
        classification_confidence = None
        filter_applied = False
        diffusion_reranked = False
        
        # Step 2: Question classification + subgraph filtering
        if self.config.use_question_classifier:
            from .question_classifier import ClassificationResult
            
            classification = self.question_classifier.classify(processed)
            question_type = classification.question_type.value
            classification_confidence = classification.confidence
            
            paths, filter_result = self.subgraph_filter.filter_paths(
                processed, paths
            )
            filter_applied = filter_result.applied_filter
        
        # Step 3: Diffusion reranking
        if self.config.use_diffusion_ranker:
            paths = self.diffusion_ranker.rerank(processed, paths)
            diffusion_reranked = True
        
        # Limit to top_k
        paths = paths[:top_k]
        
        # Update ranks
        for i, path in enumerate(paths):
            path.rank = i + 1
        
        return EnhancedRAGResponse(
            question=question,
            retrieved_paths=paths,
            question_type=question_type,
            classification_confidence=classification_confidence,
            filter_applied=filter_applied,
            diffusion_reranked=diffusion_reranked
        )
    
    def batch_ask(
        self,
        questions: List[str],
        top_k: int = 5
    ) -> List[EnhancedRAGResponse]:
        """Ask multiple questions."""
        return [self.ask(q, top_k) for q in questions]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            'index_path': str(self.index_path),
            **self.retriever.get_stats(),
            'modules': {
                'question_classifier': self.config.use_question_classifier,
                'classifier_mode': self.config.classifier_mode,
                'diffusion_ranker': self.config.use_diffusion_ranker,
                'diffusion_mode': self.config.diffusion_mode
            }
        }
        return stats
    
    def interactive_demo(self):
        """Run interactive demo."""
        print("\n" + "="*60)
        print("  Enhanced KG-RAG Demo")
        print("="*60)
        
        # Show active modules
        modules = []
        if self.config.use_question_classifier:
            modules.append(f"Question Classifier ({self.config.classifier_mode})")
        if self.config.use_diffusion_ranker:
            modules.append(f"Diffusion Ranker ({self.config.diffusion_mode})")
        
        if modules:
            print(f"Active modules: {', '.join(modules)}")
        else:
            print("No enhanced modules enabled (baseline mode)")
        
        print(f"\nIndex: {self.retriever.get_stats()['total_paths']} paths")
        print("\nEnter a question. Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                response = self.ask(question, top_k=5)
                print("\n" + response.format_readable())
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()


def create_enhanced_pipeline(
    index_path: str = "EXP/index",
    use_classifier: bool = False,
    use_diffusion: bool = False,
    classifier_mode: str = "rule",
    diffusion_mode: str = "embedding",
    **config_kwargs
) -> EnhancedKGRAGPipeline:
    """
    Factory function for enhanced pipeline.
    
    Args:
        index_path: Path to FAISS index
        use_classifier: Enable question classifier
        use_diffusion: Enable diffusion ranker
        classifier_mode: "rule", "neural", or "hybrid"
        diffusion_mode: "likelihood" or "embedding"
        
    Returns:
        Configured EnhancedKGRAGPipeline
    """
    config = RAGConfig(
        use_question_classifier=use_classifier,
        classifier_mode=classifier_mode,
        use_diffusion_ranker=use_diffusion,
        diffusion_mode=diffusion_mode,
        **config_kwargs
    )
    return EnhancedKGRAGPipeline(index_path, config)

