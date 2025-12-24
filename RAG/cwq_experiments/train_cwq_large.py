"""
CWQ with bge-large-en-v1.5 - Train a larger retriever for CWQ.
Hypothesis: Larger model (335M vs 109M params) will capture complex paths better.
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.losses import TripletLoss
from datasets import Dataset


def parse_cwq_paths(shortest_gt_paths_str):
    if not shortest_gt_paths_str or shortest_gt_paths_str == '[]':
        return []
    try:
        fixed = shortest_gt_paths_str.replace('"s ', "'s ").replace('"', '"').replace('"', '"')
        paths = json.loads(fixed)
        result = []
        for p in paths:
            if 'entities' in p and 'relations' in p:
                entities = p['entities']
                relations = p['relations']
                parts = []
                for i in range(len(relations)):
                    if i < len(entities):
                        parts.append(entities[i])
                    parts.append(relations[i])
                if len(entities) > len(relations):
                    parts.append(entities[-1])
                full_path = " -> ".join(parts)
                result.append(full_path)
        return result
    except:
        return []


def train_large_retriever():
    print("=" * 60)
    print("Training bge-large-en-v1.5 on CWQ")
    print("=" * 60)
    
    # Check if training data exists
    train_file = '../../RAG/training_data/cwq_train.jsonl'
    
    # Load training data
    print("Loading training data...")
    data = []
    with open(train_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Training samples: {len(data)}")
    
    # Convert to Dataset
    dataset = Dataset.from_list(data)
    
    # Load large model
    print("\nLoading bge-large-en-v1.5...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')
    
    # Training args - use smaller batch for large model
    training_args = SentenceTransformerTrainingArguments(
        output_dir='../models/finetuned_retriever_cwq_large',
        num_train_epochs=3,  # Fewer epochs for large model
        per_device_train_batch_size=8,  # Smaller batch for large model
        learning_rate=1e-5,
        warmup_ratio=0.1,
        logging_steps=50,
        save_total_limit=2,
        fp16=True,
    )
    
    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=TripletLoss(model),
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save
    model.save_pretrained('../models/finetuned_retriever_cwq_large')
    print("\nModel saved to: ../models/finetuned_retriever_cwq_large")


if __name__ == "__main__":
    train_large_retriever()
