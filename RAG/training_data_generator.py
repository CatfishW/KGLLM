"""
Training Data Generator for Path Retriever Fine-Tuning.

Generates contrastive learning data:
- Positive: (question, ground_truth_path)
- Hard Negatives: top-k retrieved non-ground-truth paths
- Random Negatives: sampled from corpus
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd

from .config import RAGConfig
from .data_loader import KGDataLoader, KGSample
from .retriever import HybridRetriever


@dataclass
class TrainingExample:
    """Single training example for contrastive learning."""
    query: str
    positive: str
    hard_negatives: List[str]
    random_negatives: List[str]
    question_id: str
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "positive": self.positive,
            "hard_negatives": self.hard_negatives,
            "random_negatives": self.random_negatives,
            "question_id": self.question_id
        }


class TrainingDataGenerator:
    """
    Generate training data for dense retriever fine-tuning.
    
    Uses hard negative mining from the current retriever to create
    challenging contrastive examples.
    """
    
    def __init__(
        self,
        config: RAGConfig,
        num_hard_negatives: int = 7,
        num_random_negatives: int = 3,
        retrieval_pool_size: int = 100
    ):
        self.config = config
        self.num_hard_negatives = num_hard_negatives
        self.num_random_negatives = num_random_negatives
        self.retrieval_pool_size = retrieval_pool_size
        
        self.loader = KGDataLoader(config)
        self.retriever = HybridRetriever(config)
        
        # Build corpus
        print("Building path corpus...")
        self.corpus, self.path_to_idx = self.loader.build_path_corpus(include_test=False)
        print(f"  Corpus size: {len(self.corpus)} unique paths")
        
        # Index for retrieval
        print("Indexing corpus for retrieval...")
        self.retriever.index(self.corpus)
        
    def _normalize_path(self, path: str) -> str:
        """Normalize path for comparison."""
        return " -> ".join([
            r.strip().lower()
            for r in path.replace("->", " -> ").split("->")
        ])
    
    def _get_path_strings(self, sample: KGSample) -> List[str]:
        """Get normalized ground truth path strings."""
        return sample.get_path_strings()
    
    def _mine_hard_negatives(
        self,
        question: str,
        gt_paths: List[str],
        k: int
    ) -> List[str]:
        """
        Mine hard negatives using retrieval.
        
        Returns top-k retrieved paths that are NOT ground truth.
        """
        # Retrieve candidates
        _, _, docs = self.retriever.retrieve(
            question, 
            top_k=self.retrieval_pool_size
        )
        
        # Normalize ground truth for comparison
        gt_normalized = set(self._normalize_path(p) for p in gt_paths)
        
        # Filter out ground truth paths
        hard_negatives = []
        for doc in docs:
            if self._normalize_path(doc) not in gt_normalized:
                hard_negatives.append(doc)
                if len(hard_negatives) >= k:
                    break
        
        return hard_negatives
    
    def _sample_random_negatives(
        self,
        gt_paths: List[str],
        hard_negatives: List[str],
        k: int
    ) -> List[str]:
        """Sample random negatives from corpus."""
        # Build exclusion set
        gt_normalized = set(self._normalize_path(p) for p in gt_paths)
        hard_normalized = set(self._normalize_path(p) for p in hard_negatives)
        exclude = gt_normalized | hard_normalized
        
        # Sample from corpus
        candidates = [
            p for p in self.corpus 
            if self._normalize_path(p) not in exclude
        ]
        
        random_negatives = random.sample(
            candidates, 
            min(k, len(candidates))
        )
        
        return random_negatives
    
    def generate_example(self, sample: KGSample) -> Optional[TrainingExample]:
        """Generate training example from a single sample."""
        gt_paths = self._get_path_strings(sample)
        
        if not gt_paths:
            return None
        
        # Use first ground truth path as positive
        positive = gt_paths[0]
        
        # Mine hard negatives
        hard_negatives = self._mine_hard_negatives(
            sample.question,
            gt_paths,
            self.num_hard_negatives
        )
        
        # Sample random negatives
        random_negatives = self._sample_random_negatives(
            gt_paths,
            hard_negatives,
            self.num_random_negatives
        )
        
        return TrainingExample(
            query=sample.question,
            positive=positive,
            hard_negatives=hard_negatives,
            random_negatives=random_negatives,
            question_id=sample.question_id
        )
    
    def generate_dataset(
        self,
        split: str = "train",
        output_path: Optional[Path] = None
    ) -> List[TrainingExample]:
        """
        Generate full training dataset.
        
        Args:
            split: Data split ("train", "val")
            output_path: Path to save JSON (optional)
            
        Returns:
            List of TrainingExample objects
        """
        if split == "train":
            data = self.loader.train_data
        elif split == "val":
            data = self.loader.val_data
        else:
            raise ValueError(f"Invalid split: {split}")
        
        print(f"Generating training data from {split} split ({len(data)} samples)...")
        
        examples = []
        for sample in tqdm(data, desc=f"Mining {split} negatives"):
            example = self.generate_example(sample)
            if example:
                examples.append(example)
        
        print(f"  Generated {len(examples)} training examples")
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(
                    [e.to_dict() for e in examples],
                    f,
                    indent=2
                )
            print(f"  Saved to: {output_path}")
        
        return examples
    
    def generate_sentence_transformers_format(
        self,
        examples: List[TrainingExample],
        output_path: Path
    ) -> None:
        """
        Generate data in Sentence Transformers format.
        
        Creates JSONL with:
        - anchor: query
        - positive: ground truth path
        - negative: combined hard + random negatives
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for ex in examples:
                # Combine all negatives
                all_negatives = ex.hard_negatives + ex.random_negatives
                
                # Write one line per example
                record = {
                    "anchor": ex.query,
                    "positive": ex.positive,
                    "negative": all_negatives[0] if all_negatives else ""
                }
                f.write(json.dumps(record) + "\n")
                
                # Also write examples with other negatives for diversity
                for neg in all_negatives[1:]:
                    record = {
                        "anchor": ex.query,
                        "positive": ex.positive,
                        "negative": neg
                    }
                    f.write(json.dumps(record) + "\n")
        
        print(f"Saved Sentence Transformers format to: {output_path}")


def main():
    """Generate training data for retriever fine-tuning."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="webqsp", help="Dataset name")
    parser.add_argument("--output-dir", default="./training_data", help="Output directory")
    parser.add_argument("--num-hard", type=int, default=7, help="Number of hard negatives")
    parser.add_argument("--num-random", type=int, default=3, help="Number of random negatives")
    args = parser.parse_args()
    
    config = RAGConfig(dataset=args.dataset)
    output_dir = Path(args.output_dir)
    
    generator = TrainingDataGenerator(
        config,
        num_hard_negatives=args.num_hard,
        num_random_negatives=args.num_random
    )
    
    # Generate train set
    train_examples = generator.generate_dataset(
        "train",
        output_dir / "train_examples.json"
    )
    
    # Generate val set
    val_examples = generator.generate_dataset(
        "val",
        output_dir / "val_examples.json"
    )
    
    # Generate Sentence Transformers format
    generator.generate_sentence_transformers_format(
        train_examples,
        output_dir / "train_st.jsonl"
    )
    generator.generate_sentence_transformers_format(
        val_examples,
        output_dir / "val_st.jsonl"
    )
    
    print("\n=== Training Data Generation Complete ===")
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples: {len(val_examples)}")


if __name__ == "__main__":
    main()
