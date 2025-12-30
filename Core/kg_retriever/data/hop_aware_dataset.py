"""
Hop-Aware Dataset for Path Ranking.

Extends PathRankerDataset to provide hop count labels for auxiliary training objectives.
"""

import torch
from typing import Dict, List, Optional, Union
from .path_ranker_dataset import PathRankerDataset, PathRankerDataModule, collate_fn as base_collate_fn

class HopAwarePathRankerDataset(PathRankerDataset):
    """
    Dataset that provides hop count labels in addition to ranking targets.
    """
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get base item
        item = super().__getitem__(idx)
        
        # Calculate hop count for the ground truth path
        # In the base class, 'candidate_text' contains all candidates including GT
        # 'labels' is the index of the GT path in 'candidate_text'
        gt_idx = item['labels'].item()
        gt_path = item['candidate_text'][gt_idx]
        
        # Hop count is the number of relations in the path
        # Note: gt_path is a list of relation strings
        hop_count = len(gt_path)
        
        # Clamp to 4 hops (since most are 1-2, max observed ~4-7)
        # We classify into bins: 1, 2, 3, 4+ (mapped to 0, 1, 2, 3)
        # Actually in the model we assumed 0-4 (5 classes)
        # 0-hop is rare but possible (empty path?). Let's map directly.
        
        hop_label = min(hop_count, 4)
        
        item['hop_labels'] = torch.tensor(hop_label, dtype=torch.long)
        
        return item

def hop_aware_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function that includes hop labels."""
    batch_dict = base_collate_fn(batch)
    
    # Add hop labels
    if 'hop_labels' in batch[0]:
        batch_dict['hop_labels'] = torch.stack([b['hop_labels'] for b in batch])
        
    return batch_dict

class HopAwarePathRankerDataModule(PathRankerDataModule):
    """DataModule that uses HopAwarePathRankerDataset."""
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = HopAwarePathRankerDataset(
                data_path=self.train_path,
                tokenizer_name=self.tokenizer_name,
                max_question_length=self.max_question_length,
                max_path_length=self.max_path_length,
                max_candidates=self.max_candidates,
                training=True,
                num_negatives=self.num_negatives,
            )
            
            if self.val_path:
                self.val_dataset = HopAwarePathRankerDataset(
                    data_path=self.val_path,
                    tokenizer_name=self.tokenizer_name,
                    max_question_length=self.max_question_length,
                    max_path_length=self.max_path_length,
                    max_candidates=self.max_candidates,
                    training=False,
                )
        
        if stage == 'test' or stage is None:
            if self.test_path:
                self.test_dataset = HopAwarePathRankerDataset(
                    data_path=self.test_path,
                    tokenizer_name=self.tokenizer_name,
                    max_question_length=self.max_question_length,
                    max_path_length=self.max_path_length,
                    max_candidates=self.max_candidates,
                    training=False,
                )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=hop_aware_collate_fn,
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=hop_aware_collate_fn,
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=hop_aware_collate_fn,
        )
