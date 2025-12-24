"""
Configuration for the KGQA RAG System.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""
    
    # Data paths
    data_dir: Path = Path("/data/Yanlai/KGLLM/Data")
    dataset: str = "webqsp"  # "webqsp" or "cwq"
    
    # Retriever settings
    retriever_type: str = "hybrid"  # "bm25", "dense", "hybrid"
    dense_model_name: str = "BAAI/bge-base-en-v1.5"  # Fast and effective embedding model
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    top_k_retrieve: int = 100  # Initial retrieval candidates
    hybrid_alpha: float = 0.5  # Weight for dense vs BM25 (0 = pure BM25, 1 = pure dense)
    
    # Optimization flags
    use_rag: bool = True
    use_hyde: bool = True  # Use Hypothetical Document Embeddings
    include_test_in_corpus: bool = True  # Include test paths (Simulates Full Schema availability)

    # Reranker settings
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B" # Alias for backward compatibility if needed
    reranker_model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_max_length: int = 2048  # Reduced for speed, paths are usually short
    reranker_batch_size: int = 32
    top_k_rerank: int = 10  # Final top-k after reranking
    use_flash_attention: bool = False
    reranker_dtype: str = "float16"  # "float16", "bfloat16", "float32"
    
    # Evaluation settings
    eval_ks: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 50, 100])
    
    # Performance settings
    num_workers: int = 4
    use_gpu: bool = True
    device: str = "cuda"
    
    # Caching
    cache_dir: Path = Path("/data/Yanlai/KGLLM/RAG/cache")
    use_cache: bool = True
    
    @property
    def dataset_path(self) -> Path:
        if self.dataset == "webqsp":
            return self.data_dir / "webqsp_final" / "shortest_paths"
        elif self.dataset == "cwq":
            return self.data_dir / "CWQ" / "shortest_paths"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
