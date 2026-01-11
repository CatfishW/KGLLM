"""
Dataset for Entity Linking Training.

Creates training data for the bi-encoder entity linker:
- Positive pairs: (question mention, topic entity)
- Hard negatives: Entities from the same graph that are NOT the answer

Usage:
    dataset = EntityLinkingDataset(
        data_path="train.parquet",
        tokenizer_name="BAAI/bge-small-en-v1.5",
    )
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import random
import re
from pathlib import Path
from transformers import AutoTokenizer
import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)


class EntityLinkingDataset(Dataset):
    """
    Dataset for training the bi-encoder entity linker.
    
    Each sample contains:
    - Mention text with context (from question)
    - Positive entity (topic entity from annotations)
    - Hard negative entities (other entities from the KG)
    """
    
    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer_name: str = "BAAI/bge-small-en-v1.5",
        max_seq_length: int = 128,
        num_negatives: int = 7,
        training: bool = True,
    ):
        """
        Initialize the entity linking dataset.
        
        Args:
            data_path: Path or list of paths to parquet files with QA data
            tokenizer_name: Name of tokenizer to use
            max_seq_length: Maximum sequence length
            num_negatives: Number of hard negatives per positive
            training: Whether in training mode (enables negative sampling)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.num_negatives = num_negatives
        self.training = training
        
        # Load data from one or multiple files
        self.data = self._load_data(data_path)
        if isinstance(data_path, list):
            logger.info(f"Loaded {len(self.data)} samples from {len(data_path)} files")
        else:
            logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _load_data(self, data_path: Union[str, List[str]]) -> List[Dict]:
        """Load and parse data from parquet file(s)."""
        # Handle multiple files
        if isinstance(data_path, str):
            paths = [data_path]
        else:
            paths = data_path
        
        samples = []
        for path in paths:
            if not Path(path).exists():
                logger.warning(f"File not found: {path}, skipping")
                continue
            
            try:
                df = pd.read_parquet(path)
                file_samples = self._parse_dataframe(df)
                samples.extend(file_samples)
                logger.info(f"  Loaded {len(file_samples)} samples from {Path(path).name}")
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
        
        return samples
    
    def _parse_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        """Parse dataframe into samples using vectorized operations."""
        samples = []
        
        # Find the entity column (q_entity, topic_mid, topic_entity)
        entity_col = None
        for col in ["q_entity", "topic_mid", "topic_entity"]:
            if col in df.columns:
                entity_col = col
                break
        
        if entity_col is None or "question" not in df.columns:
            logger.warning(f"Dataframe missing required columns: {df.columns.tolist()}")
            return samples
        
        # Vectorized extraction - process all rows at once
        questions = df["question"].tolist()
        entities_raw = df[entity_col].tolist()
        
        # Parse entities (can be list or JSON string)
        for i, (question, entity_val) in enumerate(zip(questions, entities_raw)):
            if not question or pd.isna(question):
                continue
                
            # Parse topic entities
            topic_entities = []
            if isinstance(entity_val, str):
                try:
                    topic_entities = json.loads(entity_val)
                except:
                    topic_entities = [entity_val] if entity_val else []
            elif isinstance(entity_val, list):
                topic_entities = entity_val
            elif entity_val is not None and not pd.isna(entity_val):
                topic_entities = [str(entity_val)]
            
            if not topic_entities:
                continue
            
            # For entity linking, we just need question -> entity pairs
            # Skip expensive graph parsing - use simple negative sampling later
            samples.append({
                "question": question,
                "topic_entities": topic_entities,
                "topic_names": topic_entities,  # Use entity as name
                "negative_entities": [],  # Will use in-batch negatives instead
                "graph": [],
            })
        
        return samples
    
    def _extract_mention(self, question: str, entity_name: str) -> str:
        """
        Create mention context for encoding.
        
        Format: "entity_name. question_context"
        """
        # Check if entity name appears in question
        question_lower = question.lower()
        entity_lower = entity_name.lower()
        
        if entity_lower in question_lower:
            # Use the actual mention from question
            return f"{entity_name}. {question}"
        else:
            # Entity might be referenced by demonym or other form
            return f"{entity_name}. {question}"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary containing:
            - mention_input_ids: [seq_len]
            - mention_attention_mask: [seq_len]
            - entity_input_ids: [num_entities, seq_len]
            - entity_attention_mask: [num_entities, seq_len]
            - labels: Index of positive entity (always 0)
        """
        sample = self.data[idx]
        
        question = sample["question"]
        topic_names = sample["topic_names"]
        negative_entities = sample["negative_entities"]
        
        # Select one topic entity as positive
        positive_entity = random.choice(topic_names) if topic_names else "unknown"
        
        # Create mention context
        mention_text = self._extract_mention(question, positive_entity)
        
        # Encode mention
        mention_encoded = self.tokenizer(
            mention_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Build entity list: positive first, then negatives
        entities = [positive_entity]
        
        if self.training and negative_entities:
            # Sample hard negatives
            num_neg = min(self.num_negatives, len(negative_entities))
            neg_sample = random.sample(negative_entities, num_neg)
            entities.extend(neg_sample)
            
            # Pad with random negatives if needed
            while len(entities) < self.num_negatives + 1:
                entities.append(random.choice(negative_entities) if negative_entities else "unknown")
        
        # Encode entities
        entity_encoded = self.tokenizer(
            entities,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "mention_input_ids": mention_encoded["input_ids"].squeeze(0),
            "mention_attention_mask": mention_encoded["attention_mask"].squeeze(0),
            "entity_input_ids": entity_encoded["input_ids"],
            "entity_attention_mask": entity_encoded["attention_mask"],
            "labels": torch.tensor(0),  # Positive is always first
            "question": question,
            "positive_entity": positive_entity,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for entity linking batches."""
    # Stack mention encodings
    mention_input_ids = torch.stack([b["mention_input_ids"] for b in batch])
    mention_attention_mask = torch.stack([b["mention_attention_mask"] for b in batch])
    
    # For in-batch negatives, we use all entities from the batch
    # Flatten entity encodings
    all_entity_input_ids = []
    all_entity_attention_mask = []
    
    for b in batch:
        # Take only the positive entity (index 0)
        all_entity_input_ids.append(b["entity_input_ids"][0])
        all_entity_attention_mask.append(b["entity_attention_mask"][0])
    
    entity_input_ids = torch.stack(all_entity_input_ids)
    entity_attention_mask = torch.stack(all_entity_attention_mask)
    
    # Labels: diagonal (each mention matches its own entity)
    labels = torch.arange(len(batch))
    
    return {
        "mention_input_ids": mention_input_ids,
        "mention_attention_mask": mention_attention_mask,
        "entity_input_ids": entity_input_ids,
        "entity_attention_mask": entity_attention_mask,
        "labels": labels,
    }


class EntityLinkingDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for entity linking."""
    
    def __init__(
        self,
        train_path: Union[str, List[str]],
        val_path: Optional[Union[str, List[str]]] = None,
        test_path: Optional[Union[str, List[str]]] = None,
        tokenizer_name: str = "BAAI/bge-small-en-v1.5",
        max_seq_length: int = 128,
        num_negatives: int = 7,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = EntityLinkingDataset(
                data_path=self.train_path,
                tokenizer_name=self.tokenizer_name,
                max_seq_length=self.max_seq_length,
                num_negatives=self.num_negatives,
                training=True,
            )
            
            if self.val_path:
                self.val_dataset = EntityLinkingDataset(
                    data_path=self.val_path,
                    tokenizer_name=self.tokenizer_name,
                    max_seq_length=self.max_seq_length,
                    num_negatives=self.num_negatives,
                    training=False,
                )
        
        if stage == "test" or stage is None:
            if self.test_path:
                self.test_dataset = EntityLinkingDataset(
                    data_path=self.test_path,
                    tokenizer_name=self.tokenizer_name,
                    max_seq_length=self.max_seq_length,
                    num_negatives=self.num_negatives,
                    training=False,
                )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )


if __name__ == "__main__":
    # Test the dataset
    import os
    
    train_path = "/data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/train.parquet"
    
    if os.path.exists(train_path):
        dataset = EntityLinkingDataset(
            data_path=train_path,
            max_seq_length=64,
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"\nSample:")
        print(f"  Question: {sample['question']}")
        print(f"  Positive entity: {sample['positive_entity']}")
        print(f"  Mention input shape: {sample['mention_input_ids'].shape}")
        print(f"  Entity input shape: {sample['entity_input_ids'].shape}")
    else:
        print(f"Train path not found: {train_path}")
