"""
Dataset for Path Ranker Model.

Loads preprocessed data with candidate paths and creates training batches.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import random
import ast
from pathlib import Path
from transformers import AutoTokenizer
import pytorch_lightning as pl


class PathRankerDataset(Dataset):
    """
    Dataset for path ranking model.
    
    Each sample contains:
    - Question text
    - Candidate relation paths (as text)
    - Ground truth path index
    """
    
    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer_name: str = "BAAI/bge-base-en-v1.5",
        max_question_length: int = 128,
        max_path_length: int = 64,
        max_candidates: int = 100,
        training: bool = True,
        negative_sampling: bool = True,
        num_negatives: int = 99,  # Number of negative candidates
    ):
        super().__init__()
        self.max_question_length = max_question_length
        self.max_path_length = max_path_length
        self.max_candidates = max_candidates
        self.training = training
        self.negative_sampling = negative_sampling
        self.num_negatives = num_negatives
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load data
        self.data = self._load_data(data_path)
        
        print(f"Loaded {len(self.data)} samples for {'training' if training else 'evaluation'}")
    
    def _load_data(self, data_path: Union[str, List[str]]) -> List[Dict]:
        """Load preprocessed data."""
        if isinstance(data_path, str):
            data_path = [data_path]
        
        all_data = []
        for path in data_path:
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                sample = self._parse_row(row)
                if sample is not None:
                    all_data.append(sample)
        
        return all_data
    
    def _parse_row(self, row) -> Optional[Dict]:
        """Parse a single data row."""
        try:
            # Parse candidate paths
            candidates = row.get('candidate_paths', [])
            if isinstance(candidates, str):
                candidates = json.loads(candidates)
            elif isinstance(candidates, np.ndarray):
                candidates = candidates.tolist()
            
            # Parse ground truth paths
            gt_paths = row.get('gt_paths', [])
            if isinstance(gt_paths, str):
                gt_paths = json.loads(gt_paths)
            elif isinstance(gt_paths, np.ndarray):
                gt_paths = gt_paths.tolist()
            
            if not candidates or not gt_paths:
                # print(f"Skipping {row.get('id')}: candidates={len(candidates) if candidates is not None else 0}, gt={len(gt_paths) if gt_paths is not None else 0}")
                return None
            
            # Convert numpy arrays in paths if any
            candidates = [list(p) if isinstance(p, (np.ndarray, list)) else p for p in candidates]
            gt_paths = [list(p) if isinstance(p, (np.ndarray, list)) else p for p in gt_paths]
            
            # Find GT path index in candidates
            gt_idx = None
            for i, cand in enumerate(candidates):
                if cand in gt_paths:
                    gt_idx = i
                    break
            
            if gt_idx is None:
                # GT not in candidates - add it
                candidates.append(gt_paths[0])
                gt_idx = len(candidates) - 1
            
            return {
                'id': row['id'],
                'question': row['question'],
                'candidate_paths': candidates,
                'gt_paths': gt_paths,
                'gt_idx': gt_idx,
            }
        except Exception as e:
            return None
    
    def path_to_text(self, path: List[str]) -> str:
        """Convert relation path to text for encoding."""
        if not path:
            return "[EMPTY]"
        # Replace underscores and dots with spaces for better tokenization
        relations = [r.replace('.', ' ').replace('_', ' ') for r in path]
        return " -> ".join(relations)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        # Get candidates
        all_candidates = sample['candidate_paths']
        gt_idx = sample['gt_idx']
        
        # Sample candidates for training
        if self.training and self.negative_sampling:
            # Always include ground truth
            selected_indices = [gt_idx]
            
            # Sample negatives
            other_indices = [i for i in range(len(all_candidates)) if i != gt_idx]
            if len(other_indices) > self.num_negatives:
                other_indices = random.sample(other_indices, self.num_negatives)
            selected_indices.extend(other_indices)
            
            # Shuffle and find new GT index
            random.shuffle(selected_indices)
            candidates = [all_candidates[i] for i in selected_indices]
            new_gt_idx = selected_indices.index(gt_idx)
        else:
            # Use all candidates up to max
            # If GT is outside max_candidates, force it in so we validate correctly
            if gt_idx >= self.max_candidates:
                # Take top (max-1) candidates
                candidates = all_candidates[:self.max_candidates - 1]
                # Append GT
                candidates.append(all_candidates[gt_idx])
                new_gt_idx = self.max_candidates - 1
            else:
                candidates = all_candidates[:self.max_candidates]
                new_gt_idx = gt_idx
        
        num_candidates = len(candidates)
        
        # Tokenize question
        q_enc = self.tokenizer(
            sample['question'],
            max_length=self.max_question_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Tokenize all candidate paths
        path_texts = [self.path_to_text(p) for p in candidates]
        
        # Pad to max_candidates
        while len(path_texts) < self.max_candidates:
            path_texts.append("[PAD]")
        path_texts = path_texts[:self.max_candidates]
        
        p_enc = self.tokenizer(
            path_texts,
            max_length=self.max_path_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Create candidate mask
        candidate_mask = torch.zeros(self.max_candidates)
        candidate_mask[:num_candidates] = 1.0
        
        return {
            'id': sample['id'],
            'question_input_ids': q_enc['input_ids'].squeeze(0),
            'question_attention_mask': q_enc['attention_mask'].squeeze(0),
            'path_input_ids': p_enc['input_ids'],  # [max_candidates, max_path_length]
            'path_attention_mask': p_enc['attention_mask'],
            'candidate_mask': candidate_mask,
            'labels': torch.tensor(new_gt_idx, dtype=torch.long),
            'question_text': sample['question'],
            'candidate_text': candidates,  # Raw text list
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch for path ranker."""
    return {
        'id': [b['id'] for b in batch],
        'question_input_ids': torch.stack([b['question_input_ids'] for b in batch]),
        'question_attention_mask': torch.stack([b['question_attention_mask'] for b in batch]),
        'path_input_ids': torch.stack([b['path_input_ids'] for b in batch]),
        'path_attention_mask': torch.stack([b['path_attention_mask'] for b in batch]),
        'candidate_mask': torch.stack([b['candidate_mask'] for b in batch]),
        'candidate_mask': torch.stack([b['candidate_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'question_text': [b['question_text'] for b in batch],
        'candidate_text': [b['candidate_text'] for b in batch],
    }


class PathRankerDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for Path Ranker."""
    
    def __init__(
        self,
        train_path: Union[str, List[str]],
        val_path: Optional[Union[str, List[str]]] = None,
        test_path: Optional[Union[str, List[str]]] = None,
        tokenizer_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 16,
        num_workers: int = 4,
        max_question_length: int = 128,
        max_path_length: int = 64,
        max_candidates: int = 100,
        num_negatives: int = 99,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_question_length = max_question_length
        self.max_path_length = max_path_length
        self.max_candidates = max_candidates
        self.num_negatives = num_negatives
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = PathRankerDataset(
                data_path=self.train_path,
                tokenizer_name=self.tokenizer_name,
                max_question_length=self.max_question_length,
                max_path_length=self.max_path_length,
                max_candidates=self.max_candidates,
                training=True,
                num_negatives=self.num_negatives,
            )
            
            if self.val_path:
                self.val_dataset = PathRankerDataset(
                    data_path=self.val_path,
                    tokenizer_name=self.tokenizer_name,
                    max_question_length=self.max_question_length,
                    max_path_length=self.max_path_length,
                    max_candidates=self.max_candidates,
                    training=False,
                )
        
        if stage == 'test' or stage is None:
            if self.test_path:
                self.test_dataset = PathRankerDataset(
                    data_path=self.test_path,
                    tokenizer_name=self.tokenizer_name,
                    max_question_length=self.max_question_length,
                    max_path_length=self.max_path_length,
                    max_candidates=self.max_candidates,
                    training=False,
                )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


if __name__ == '__main__':
    # Test the dataset
    print("Testing PathRankerDataset...")
    
    dataset = PathRankerDataset(
        data_path='/data/Yanlai/KGLLM/Data/preprocessed_paths/webqsp_train.parquet',
        tokenizer_name='sentence-transformers/all-MiniLM-L6-v2',  # Smaller for testing
        max_candidates=50,
        num_negatives=49,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"Question shape: {sample['question_input_ids'].shape}")
    print(f"Path input shape: {sample['path_input_ids'].shape}")
    print(f"Candidate mask shape: {sample['candidate_mask'].shape}")
    print(f"Label: {sample['labels'].item()}")
