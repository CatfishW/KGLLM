"""
Training script for GSR Subgraph ID Generator.

Trains a T5 model to generate subgraph IDs (relation patterns) from questions.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsr.subgraph_id_generator import SubgraphIDGenerator


class GSRDataset(Dataset):
    """Dataset for GSR training: (question, subgraph_id) pairs."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: T5Tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # Load data
        self.samples = self._load_data(data_path)
        
        print(f"Loaded {len(self.samples)} training samples")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load training data."""
        path = Path(data_path)
        
        if path.suffix == '.jsonl':
            samples = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    samples.append(json.loads(line))
            return samples
        elif path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        question = sample['question']
        subgraph_id = sample['subgraph_id']
        
        # Format input
        input_text = f"Question: {question}"
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            subgraph_id,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0)
        }


def main():
    parser = argparse.ArgumentParser(description='Train GSR Subgraph ID Generator')
    
    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data (jsonl)')
    parser.add_argument('--val_data', type=str, default=None,
                       help='Path to validation data (jsonl)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='t5-small',
                       help='T5 model name (t5-small, t5-base, etc.)')
    parser.add_argument('--output_dir', type=str, default='outputs_gsr',
                       help='Output directory for model checkpoints')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Warmup steps')
    parser.add_argument('--max_input_length', type=int, default=512,
                       help='Maximum input sequence length')
    parser.add_argument('--max_target_length', type=int, default=128,
                       help='Maximum target sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
    # Create datasets
    print("Loading training data...")
    train_dataset = GSRDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length
    )
    
    val_dataset = None
    if args.val_data:
        print("Loading validation data...")
        val_dataset = GSRDataset(
            data_path=args.val_data,
            tokenizer=tokenizer,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length
        )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000 if val_dataset else None,
        evaluation_strategy='steps' if val_dataset else 'no',
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model='eval_loss' if val_dataset else None,
        greater_is_better=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training complete!")


if __name__ == '__main__':
    main()

