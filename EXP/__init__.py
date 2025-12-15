"""
KG-RAG: Knowledge Graph Path Retrieval for Question Answering

Modules:
- config: Configuration settings
- path_indexer: Build FAISS index from parquet data
- retriever: Semantic path retrieval
- pipeline: End-to-end RAG pipeline
- question_classifier: Classify questions by complexity type
- subgraph_filter: Filter paths by question type
- diffusion_ranker: Rerank paths using diffusion model
"""

from .config import RAGConfig
from .retriever import KGPathRetriever, RetrievedPath
from .pipeline import KGRAGPipeline, EnhancedKGRAGPipeline, create_enhanced_pipeline
from .question_classifier import QuestionClassifier, QuestionType
from .subgraph_filter import SubgraphFilter
from .diffusion_ranker import DiffusionRanker

__all__ = [
    # Config
    "RAGConfig",
    # Core retrieval
    "KGPathRetriever", 
    "RetrievedPath",
    # Pipelines
    "KGRAGPipeline",
    "EnhancedKGRAGPipeline",
    "create_enhanced_pipeline",
    # Enhanced modules
    "QuestionClassifier",
    "QuestionType",
    "SubgraphFilter",
    "DiffusionRanker",
]

__version__ = "1.1.0"

