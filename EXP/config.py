"""
Configuration settings for KG-RAG system.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class RAGConfig:
    """Configuration for the KG-RAG retrieval system."""
    
    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384  # Dimension for all-MiniLM-L6-v2
    
    # FAISS index settings
    index_type: str = "flat"  # "flat" for accuracy, "ivf" for speed with large data
    nlist: int = 100  # Number of clusters for IVF index
    nprobe: int = 10  # Number of clusters to search for IVF
    use_gpu: bool = False  # Use GPU for FAISS (requires faiss-gpu)
    
    # Retrieval settings
    default_top_k: int = 10
    similarity_threshold: float = 0.0  # Minimum similarity score
    
    # Data paths
    data_dir: str = "Data/webqsp_final"
    index_dir: str = "EXP/index"
    vocab_path: str = "Data/webqsp_final/vocab.json"
    
    # Processing settings
    batch_size: int = 64
    max_path_length: int = 5  # Maximum number of relations in a path
    
    # Relation chain formatting
    relation_separator: str = " -> "
    
    # Re-ranking (optional)
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 20  # Get more candidates before re-ranking
    
    # === Enhanced Modules ===
    
    # Question Classifier
    use_question_classifier: bool = False
    classifier_mode: str = "rule"  # "rule", "neural", or "hybrid"
    classifier_model_path: Optional[str] = None
    classifier_strict_filter: bool = False
    classifier_confidence_threshold: float = 0.7
    classifier_min_results: int = 3  # Minimum results after filtering
    
    # Diffusion Ranker
    use_diffusion_ranker: bool = False
    diffusion_mode: str = "embedding"  # "likelihood" or "embedding"
    diffusion_model_path: Optional[str] = None
    diffusion_num_steps: int = 10  # Steps for likelihood scoring
    diffusion_score_weight: float = 0.5  # Weight vs retrieval score
    
    @property
    def train_parquet_path(self) -> Path:
        return Path(self.data_dir) / "train.parquet"
    
    @property
    def val_parquet_path(self) -> Path:
        return Path(self.data_dir) / "val.parquet"
    
    @property
    def test_parquet_path(self) -> Path:
        return Path(self.data_dir) / "test.parquet"
    
    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)
    
    def validate(self) -> bool:
        """Validate configuration."""
        assert self.index_type in ["flat", "ivf"], f"Invalid index type: {self.index_type}"
        assert self.default_top_k > 0, "top_k must be positive"
        assert self.classifier_mode in ["rule", "neural", "hybrid"], f"Invalid classifier mode"
        assert self.diffusion_mode in ["likelihood", "embedding"], f"Invalid diffusion mode"
        return True
    
    def with_classifier(self, enable: bool = True, mode: str = "rule") -> 'RAGConfig':
        """Return config with classifier enabled/disabled."""
        self.use_question_classifier = enable
        self.classifier_mode = mode
        return self
    
    def with_diffusion(self, enable: bool = True, mode: str = "embedding") -> 'RAGConfig':
        """Return config with diffusion ranker enabled/disabled."""
        self.use_diffusion_ranker = enable
        self.diffusion_mode = mode
        return self


# Default configuration
DEFAULT_CONFIG = RAGConfig()


def get_relation_natural_language(relation: str) -> str:
    """
    Convert a Freebase relation to natural language description.
    
    Example:
        "people.person.parents" -> "person's parents"
        "film.film.directed_by" -> "film directed by"
    """
    # Remove common prefixes
    parts = relation.split(".")
    
    if len(parts) >= 3:
        domain = parts[0]  # e.g., "people", "film"
        entity_type = parts[1]  # e.g., "person", "film"  
        property_name = parts[2]  # e.g., "parents", "directed_by"
        
        # Convert underscores to spaces
        property_name = property_name.replace("_", " ")
        
        return f"{entity_type}'s {property_name}"
    elif len(parts) == 2:
        return parts[1].replace("_", " ")
    else:
        return relation.replace("_", " ").replace(".", " ")


def relation_chain_to_natural_language(relations: List[str], separator: str = " -> ") -> str:
    """
    Convert a relation chain to natural language.
    
    Example:
        ["people.person.parents", "people.person.children"] 
        -> "person's parents -> person's children"
    """
    nl_relations = [get_relation_natural_language(r) for r in relations]
    return separator.join(nl_relations)
