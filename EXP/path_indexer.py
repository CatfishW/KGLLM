"""
Path Indexer: Build FAISS vector index from relation paths in WebQSP dataset.

This module extracts unique relation chains from the parquet data,
creates embeddings using sentence-transformers, and builds a FAISS index
for fast similarity search.
"""

import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .config import (
    RAGConfig, 
    DEFAULT_CONFIG,
    relation_chain_to_natural_language,
    get_relation_natural_language
)


@dataclass
class PathMetadata:
    """Metadata for an indexed path."""
    path_id: int
    relation_chain: str  # e.g., "people.person.parents -> people.person.children"
    relations: List[str]  # e.g., ["people.person.parents", "people.person.children"]
    natural_language: str  # e.g., "person's parents -> person's children"
    example_questions: List[str]  # Sample questions that use this path
    example_entities: List[List[str]]  # Sample entity chains
    frequency: int  # How often this path appears in training data


class PathIndex:
    """
    FAISS-based index for relation paths.
    
    Stores embeddings of relation chains and supports fast similarity search.
    """
    
    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.config = config
        self.embedding_model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.metadata: List[PathMetadata] = []
        self.relation_to_ids: Dict[str, List[int]] = defaultdict(list)
        
    def _load_embedding_model(self):
        """Load the sentence transformer model."""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            
    def _create_faiss_index(self, dim: int) -> faiss.Index:
        """Create FAISS index based on configuration."""
        if self.config.index_type == "flat":
            # Exact search with inner product (cosine similarity with normalized vectors)
            index = faiss.IndexFlatIP(dim)
        elif self.config.index_type == "ivf":
            # Approximate search with inverted file index
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, self.config.nlist)
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
            
        if self.config.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("Using GPU-accelerated FAISS")
            except Exception as e:
                print(f"GPU not available, using CPU: {e}")
                
        return index
    
    def build_from_parquet(
        self,
        parquet_paths: List[Path],
        output_dir: Path,
        include_question_pairs: bool = True
    ):
        """
        Build the index from parquet files.
        
        Args:
            parquet_paths: List of paths to parquet files
            output_dir: Directory to save the index
            include_question_pairs: Also index question-path pairs for better retrieval
        """
        self._load_embedding_model()
        
        # Step 1: Extract all unique paths and their metadata
        print("Step 1: Extracting paths from parquet files...")
        path_data = self._extract_paths_from_parquet(parquet_paths)
        
        print(f"Found {len(path_data)} unique relation chains")
        
        # Step 2: Create embeddings
        print("Step 2: Creating embeddings...")
        embeddings, metadata = self._create_embeddings(
            path_data, 
            include_question_pairs
        )
        
        # Step 3: Build FAISS index
        print("Step 3: Building FAISS index...")
        self._build_index(embeddings)
        self.metadata = metadata
        
        # Build relation lookup
        for i, m in enumerate(metadata):
            for rel in m.relations:
                self.relation_to_ids[rel].append(i)
        
        # Step 4: Save index and metadata
        print("Step 4: Saving index...")
        self.save(output_dir)
        
        print(f"Index built successfully with {len(metadata)} entries")
        return self
    
    def _extract_paths_from_parquet(
        self, 
        parquet_paths: List[Path]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract unique relation chains from parquet files.
        
        Returns:
            Dictionary mapping relation chain string to metadata
        """
        path_data = {}  # relation_chain_str -> metadata
        
        for parquet_path in parquet_paths:
            if not parquet_path.exists():
                print(f"Warning: {parquet_path} not found, skipping")
                continue
                
            print(f"Processing {parquet_path.name}...")
            df = pd.read_parquet(parquet_path)
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting paths"):
                question = row['question']
                paths_str = row['paths']
                
                # Parse paths JSON
                if isinstance(paths_str, str):
                    try:
                        paths = json.loads(paths_str)
                    except json.JSONDecodeError:
                        continue
                else:
                    paths = paths_str if paths_str else []
                
                for path in paths:
                    if not path:
                        continue
                        
                    relations = path.get('relations', [])
                    if not relations:
                        continue
                    
                    # Skip paths that are too long
                    if len(relations) > self.config.max_path_length:
                        continue
                    
                    relation_chain = self.config.relation_separator.join(relations)
                    entities = path.get('entities', [])
                    
                    if relation_chain not in path_data:
                        path_data[relation_chain] = {
                            'relations': relations,
                            'questions': [],
                            'entities': [],
                            'frequency': 0
                        }
                    
                    path_data[relation_chain]['questions'].append(question)
                    if entities:
                        path_data[relation_chain]['entities'].append(entities)
                    path_data[relation_chain]['frequency'] += 1
        
        return path_data
    
    def _create_embeddings(
        self,
        path_data: Dict[str, Dict[str, Any]],
        include_question_pairs: bool = True
    ) -> Tuple[np.ndarray, List[PathMetadata]]:
        """
        Create embeddings for all paths.
        
        Uses two strategies:
        1. Embed the natural language description of the relation chain
        2. (Optional) Also embed question-path pairs for better matching
        """
        texts_to_embed = []
        metadata = []
        
        for path_id, (relation_chain, data) in enumerate(path_data.items()):
            relations = data['relations']
            questions = data['questions']
            entities = data['entities']
            frequency = data['frequency']
            
            # Create natural language description
            nl_description = relation_chain_to_natural_language(relations)
            
            # Sample example questions and entities
            example_questions = questions[:5]  # Keep up to 5 examples
            example_entities = entities[:5]
            
            meta = PathMetadata(
                path_id=path_id,
                relation_chain=relation_chain,
                relations=relations,
                natural_language=nl_description,
                example_questions=example_questions,
                example_entities=example_entities,
                frequency=frequency
            )
            metadata.append(meta)
            
            # Create combined text for embedding:
            # Include both the relation chain description AND example questions
            # This helps align the embedding space between questions and paths
            if include_question_pairs and example_questions:
                # Combine natural language description with sample questions
                combined_texts = [nl_description]
                # Add up to 3 example questions to create a richer representation
                for q in example_questions[:3]:
                    combined_texts.append(q)
                # Join with separator
                combined_text = " | ".join(combined_texts)
                texts_to_embed.append(combined_text)
            else:
                # Just use natural language description
                texts_to_embed.append(nl_description)
        
        # Create embeddings in batches
        print(f"Embedding {len(texts_to_embed)} texts...")
        embeddings = self.embedding_model.encode(
            texts_to_embed,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embeddings, metadata
    
    def _build_index(self, embeddings: np.ndarray):
        """Build the FAISS index from embeddings."""
        dim = embeddings.shape[1]
        self.index = self._create_faiss_index(dim)
        
        # Train if IVF index
        if self.config.index_type == "ivf":
            print("Training IVF index...")
            self.index.train(embeddings)
        
        # Add embeddings
        self.index.add(embeddings)
        
        print(f"Index contains {self.index.ntotal} vectors of dimension {dim}")
    
    def save(self, output_dir: Path):
        """Save index and metadata to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = output_dir / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = output_dir / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump({
                'metadata': [asdict(m) for m in self.metadata],
                'relation_to_ids': dict(self.relation_to_ids),
                'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config)
            }, f)
        
        # Save human-readable summary
        summary_path = output_dir / "index_summary.json"
        summary = {
            'total_paths': len(self.metadata),
            'index_type': self.config.index_type,
            'embedding_model': self.config.embedding_model,
            'embedding_dim': self.config.embedding_dim,
            'top_paths_by_frequency': [
                {
                    'relation_chain': m.relation_chain,
                    'frequency': m.frequency,
                    'example_question': m.example_questions[0] if m.example_questions else None
                }
                for m in sorted(self.metadata, key=lambda x: -x.frequency)[:20]
            ]
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Index saved to {output_dir}")
    
    def load(self, index_dir: Path) -> 'PathIndex':
        """Load index and metadata from disk."""
        index_dir = Path(index_dir)
        
        # Load FAISS index
        index_path = index_dir / "faiss.index"
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = index_dir / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
            self.metadata = [PathMetadata(**m) for m in data['metadata']]
            self.relation_to_ids = defaultdict(list, data['relation_to_ids'])
        
        print(f"Loaded index with {len(self.metadata)} paths")
        return self
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10
    ) -> List[Tuple[PathMetadata, float]]:
        """
        Search for similar paths.
        
        Args:
            query_embedding: Query vector (normalized)
            top_k: Number of results to return
            
        Returns:
            List of (PathMetadata, score) tuples
        """
        if self.config.index_type == "ivf":
            self.index.nprobe = self.config.nprobe
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
        
        return results


def build_index(
    data_dir: str = "Data/webqsp_final",
    output_dir: str = "EXP/index",
    config: Optional[RAGConfig] = None
) -> PathIndex:
    """
    Build the path index from WebQSP data.
    
    Args:
        data_dir: Directory containing parquet files
        output_dir: Directory to save the index
        config: Optional configuration
        
    Returns:
        Built PathIndex
    """
    if config is None:
        config = RAGConfig(data_dir=data_dir, index_dir=output_dir)
    
    data_path = Path(data_dir)
    parquet_files = [
        data_path / "train.parquet",
        data_path / "val.parquet",
        # Optionally include test.parquet
    ]
    
    index = PathIndex(config)
    index.build_from_parquet(parquet_files, Path(output_dir))
    
    return index


def main():
    """Command-line interface for building the index."""
    parser = argparse.ArgumentParser(description="Build FAISS index for KG path retrieval")
    parser.add_argument("--data_dir", type=str, default="Data/webqsp_final",
                       help="Directory containing parquet files")
    parser.add_argument("--output_dir", type=str, default="EXP/index",
                       help="Directory to save the index")
    parser.add_argument("--embedding_model", type=str, 
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Sentence transformer model for embeddings")
    parser.add_argument("--index_type", type=str, default="flat",
                       choices=["flat", "ivf"],
                       help="FAISS index type")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for embedding")
    
    args = parser.parse_args()
    
    config = RAGConfig(
        data_dir=args.data_dir,
        index_dir=args.output_dir,
        embedding_model=args.embedding_model,
        index_type=args.index_type,
        batch_size=args.batch_size
    )
    
    build_index(args.data_dir, args.output_dir, config)


if __name__ == "__main__":
    main()
