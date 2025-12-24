"""
Reranker Fine-Tuning with Margin Ranking Loss.

Uses sentence-transformers CrossEncoder with proper ranking loss
instead of causal LM loss.
"""
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch
import numpy as np
from tqdm import tqdm

from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader


@dataclass  
class RerankerConfig:
    """Configuration for reranker training."""
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    output_dir: str = "./models/finetuned_reranker_ce"
    
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    
    train_data_path: str = "./training_data/train_examples.json"
    val_data_path: str = "./training_data/val_examples.json"
    
    max_length: int = 256
    seed: int = 42


def load_reranker_data(path: str) -> List[InputExample]:
    """Load data for cross-encoder training."""
    examples = []
    
    with open(path, "r") as f:
        data = json.load(f)
    
    for item in data:
        query = item["query"]
        positive = item["positive"]
        hard_negs = item.get("hard_negatives", [])
        random_negs = item.get("random_negatives", [])
        
        # Positive example (label=1)
        examples.append(InputExample(texts=[query, positive], label=1.0))
        
        # Negative examples (label=0)
        all_negs = hard_negs + random_negs
        for neg in all_negs[:3]:  # Limit negatives
            examples.append(InputExample(texts=[query, neg], label=0.0))
    
    return examples


def create_evaluation_data(path: str) -> Dict:
    """Create data for CERerankingEvaluator."""
    with open(path, "r") as f:
        data = json.load(f)
    
    # Format: {query_id: {'query': str, 'positive': list, 'negative': list}}
    eval_data = {}
    
    for i, item in enumerate(data):
        query = item["query"]
        positive = item["positive"]
        hard_negs = item.get("hard_negatives", [])
        random_negs = item.get("random_negatives", [])
        
        eval_data[f"q{i}"] = {
            'query': query,
            'positive': [positive],  # Use list
            'negative': hard_negs[:3] + random_negs[:2]  # Use list
        }
    
    return eval_data


def train_reranker(config: RerankerConfig):
    """Train cross-encoder reranker with ranking loss."""
    print("=" * 60)
    print("Cross-Encoder Reranker Training")
    print("=" * 60)
    print(f"Base model: {config.base_model}")
    print(f"Output: {config.output_dir}")
    
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Load model
    print("\nLoading cross-encoder...")
    model = CrossEncoder(
        config.base_model,
        max_length=config.max_length,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load training data
    print(f"\nLoading training data from {config.train_data_path}...")
    train_examples = load_reranker_data(config.train_data_path)
    print(f"  Loaded {len(train_examples)} training pairs")
    
    # Create data loader
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=config.batch_size
    )
    
    # Create evaluator
    evaluator = None
    if Path(config.val_data_path).exists():
        print(f"Loading validation data from {config.val_data_path}...")
        eval_data = create_evaluation_data(config.val_data_path)
        evaluator = CERerankingEvaluator(eval_data, name="val")
        print(f"  Created evaluator with {len(eval_data)} queries")
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=config.epochs,
        warmup_steps=config.warmup_steps,
        output_path=config.output_dir,
        save_best_model=True,
        show_progress_bar=True
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--output-dir", default="./models/finetuned_reranker_ce")
    parser.add_argument("--train-data", default="./training_data/train_examples.json")
    parser.add_argument("--val-data", default="./training_data/val_examples.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    
    config = RerankerConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    train_reranker(config)


if __name__ == "__main__":
    main()
