"""
LoRA Fine-Tuning for Qwen3 Reranker.

Uses PEFT/LoRA for efficient fine-tuning of the cross-encoder reranker
on WebQSP path relevance data.
"""
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)


@dataclass
class RerankerTrainConfig:
    """Training configuration for LoRA fine-tuning."""
    # Model
    base_model: str = "Qwen/Qwen3-Reranker-0.6B"
    output_dir: str = "./models/finetuned_reranker"
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None  # Auto-detect
    
    # Training
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 256
    
    # Quantization for memory efficiency
    use_4bit: bool = True
    
    # Data
    train_data_path: str = "./training_data/train_examples.json"
    val_data_path: str = "./training_data/val_examples.json"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for Qwen models
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class RerankerDataset(Dataset):
    """Dataset for reranker fine-tuning."""
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_length: int = 256,
        instruction: str = "Given a question, determine if the following Knowledge Graph path is relevant for answering it."
    ):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        
        # Create positive and negative pairs
        for ex in examples:
            query = ex["query"]
            positive = ex["positive"]
            hard_negatives = ex.get("hard_negatives", [])
            random_negatives = ex.get("random_negatives", [])
            
            # Positive example
            self.examples.append({
                "query": query,
                "document": positive,
                "label": 1
            })
            
            # Negative examples (sample to balance)
            all_negatives = hard_negatives + random_negatives
            for neg in all_negatives[:3]:  # Limit negatives per query
                self.examples.append({
                    "query": query,
                    "document": neg,
                    "label": 0
                })
        
        print(f"  Created {len(self.examples)} training pairs")
    
    def __len__(self):
        return len(self.examples)
    
    def _format_input(self, query: str, document: str) -> str:
        """Format input for the reranker using Qwen's expected format."""
        # Use instruct format
        prompt = f"<|im_start|>system\n{self.instruction}<|im_end|>\n"
        prompt += f"<|im_start|>user\nQuestion: {query}\nPath: {document}\n\nIs this path relevant? Answer 'yes' or 'no'.<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    def __getitem__(self, idx) -> Dict:
        ex = self.examples[idx]
        
        # Format input
        text = self._format_input(ex["query"], ex["document"])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get label (for ranking, we use the yes/no token position)
        label = ex["label"]
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class RerankerTrainer:
    """LoRA trainer for reranker."""
    
    def __init__(self, config: RerankerTrainConfig):
        self.config = config
        self.device = config.device
        
        # Set seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if enabled
        print("Loading base model...")
        if config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Apply LoRA
        print("Applying LoRA...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_data(self) -> Tuple[Dataset, Dataset]:
        """Load training and validation datasets."""
        print(f"Loading training data from {self.config.train_data_path}...")
        with open(self.config.train_data_path, "r") as f:
            train_examples = json.load(f)
        
        train_dataset = RerankerDataset(
            train_examples,
            self.tokenizer,
            max_length=self.config.max_length
        )
        
        val_dataset = None
        if Path(self.config.val_data_path).exists():
            print(f"Loading validation data from {self.config.val_data_path}...")
            with open(self.config.val_data_path, "r") as f:
                val_examples = json.load(f)
            val_dataset = RerankerDataset(
                val_examples,
                self.tokenizer,
                max_length=self.config.max_length
            )
        
        return train_dataset, val_dataset
    
    def train(self):
        """Run LoRA fine-tuning."""
        print("=" * 60)
        print("Reranker LoRA Fine-Tuning")
        print("=" * 60)
        
        # Load data
        train_dataset, val_dataset = self.load_data()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="epoch" if val_dataset else "no",
            load_best_model_at_end=True if val_dataset else False,
            fp16=False,  # Use bf16 instead
            bf16=torch.cuda.is_available(),
            report_to="none",
            seed=self.config.seed,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(self.tokenizer)
        
        # Custom Trainer with ranking loss
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Save LoRA weights
        print(f"\nSaving LoRA weights to {self.config.output_dir}...")
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"LoRA weights saved to: {self.config.output_dir}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3 reranker")
    parser.add_argument("--base-model", default="Qwen/Qwen3-Reranker-0.6B",
                        help="Base reranker model")
    parser.add_argument("--output-dir", default="./models/finetuned_reranker",
                        help="Output directory")
    parser.add_argument("--train-data", default="./training_data/train_examples.json",
                        help="Training data path")
    parser.add_argument("--val-data", default="./training_data/val_examples.json",
                        help="Validation data path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    config = RerankerTrainConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        learning_rate=args.lr,
        use_4bit=not args.no_4bit
    )
    
    trainer = RerankerTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
