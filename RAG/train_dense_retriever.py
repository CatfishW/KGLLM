"""
Dense Retriever Fine-Tuning with Sentence Transformers.

Fine-tunes a bi-encoder using Multiple Negatives Ranking Loss (MNRL)
for path retrieval in KGQA.

Base Model: dunzhang/stella_en_400M_v5 (SOTA efficient embedding model)
"""
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import torch
from tqdm import tqdm

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    models,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from datasets import Dataset


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    base_model: str = "dunzhang/stella_en_400M_v5"
    output_dir: str = "./models/finetuned_retriever"
    
    # Training
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Data
    train_data_path: str = "./training_data/train_st.jsonl"
    val_data_path: str = "./training_data/val_st.jsonl"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Misc
    seed: int = 42


def load_training_data(path: str) -> List[InputExample]:
    """Load training data from JSONL file."""
    examples = []
    
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            # MNRL format: (anchor, positive, negative)
            examples.append(InputExample(
                texts=[
                    record["anchor"],
                    record["positive"],
                    record["negative"]
                ]
            ))
    
    return examples


def load_training_data_as_dataset(path: str) -> Dataset:
    """Load training data as HuggingFace Dataset for new API."""
    data = {"anchor": [], "positive": [], "negative": []}
    
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            data["anchor"].append(record["anchor"])
            data["positive"].append(record["positive"])
            data["negative"].append(record["negative"])
    
    return Dataset.from_dict(data)


def create_evaluator(
    val_data_path: str,
    corpus_path: Optional[str] = None
) -> evaluation.TripletEvaluator:
    """Create evaluator for validation."""
    anchors = []
    positives = []
    negatives = []
    
    with open(val_data_path, "r") as f:
        for line in f:
            record = json.loads(line)
            anchors.append(record["anchor"])
            positives.append(record["positive"])
            negatives.append(record["negative"])
    
    return evaluation.TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name="val_triplet",
        show_progress_bar=True
    )


def train_retriever(config: TrainConfig):
    """
    Fine-tune dense retriever using MNRL loss.
    
    MNRL uses in-batch negatives for efficient contrastive learning.
    """
    print("=" * 60)
    print("Dense Retriever Fine-Tuning")
    print("=" * 60)
    print(f"Base model: {config.base_model}")
    print(f"Output dir: {config.output_dir}")
    print(f"Device: {config.device}")
    print()
    
    # Set seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Check if training data exists
    if not Path(config.train_data_path).exists():
        print(f"ERROR: Training data not found at {config.train_data_path}")
        print("Please run training_data_generator.py first:")
        print("  python -m RAG.training_data_generator --dataset webqsp")
        return
    
    # Load model
    print("Loading base model...")
    model = SentenceTransformer(config.base_model, device=config.device, trust_remote_code=True)
    
    # Check model info
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"  Max sequence length: {model.max_seq_length}")
    
    # Load training data
    print(f"\nLoading training data from {config.train_data_path}...")
    train_dataset = load_training_data_as_dataset(config.train_data_path)
    print(f"  Loaded {len(train_dataset)} training examples")
    
    # Load validation data
    val_dataset = None
    evaluator = None
    if Path(config.val_data_path).exists():
        print(f"Loading validation data from {config.val_data_path}...")
        evaluator = create_evaluator(config.val_data_path)
        print(f"  Created triplet evaluator")
    
    # Create loss function
    # MNRL uses in-batch negatives automatically
    loss = losses.MultipleNegativesRankingLoss(model)
    
    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        seed=config.seed,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch" if evaluator else "no",
        load_best_model_at_end=False,  # Custom evaluator doesn't provide eval_loss
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )
    
    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # We use custom evaluator
        loss=loss,
        evaluator=evaluator,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {config.output_dir}...")
    model.save(config.output_dir)
    
    # Final evaluation
    if evaluator:
        print("\nFinal evaluation:")
        results = evaluator(model)
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune dense retriever")
    parser.add_argument("--base-model", default="dunzhang/stella_en_400M_v5",
                        help="Base embedding model")
    parser.add_argument("--output-dir", default="./models/finetuned_retriever",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--train-data", default="./training_data/train_st.jsonl",
                        help="Training data path")
    parser.add_argument("--val-data", default="./training_data/val_st.jsonl",
                        help="Validation data path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    config = TrainConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    train_retriever(config)


if __name__ == "__main__":
    main()
