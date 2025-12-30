"""
Training Script for HopColBERT Reranker

State-of-the-art path reranking with:
- Late Interaction (MaxSim) scoring
- Hop-wise auxiliary losses
- Multi-task training with Plackett-Luce + margin loss

Usage:
    python train_hop_colbert.py --config configs/hop_colbert.yaml
    python train_hop_colbert.py --config configs/hop_colbert.yaml --fast_dev_run
"""

import os
import sys
import json
import argparse
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    LearningRateMonitor, 
    EarlyStopping,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from models.hop_colbert_reranker import HopColBERTReranker
from data.path_ranker_dataset import PathRankerDataModule, PathRankerDataset


class PathRankingDataset(Dataset):
    """
    Dataset for path ranking with hop boundary detection.
    
    Expected data format:
    {
        "question": "What is...",
        "candidate_paths": [
            ["relation1", "relation2"],
            ["relation1"],
            ...
        ],
        "labels": 0,  # Index of correct path
        "answer": ["entity1", "entity2"]  # Optional, for evaluation
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_question_length: int = 128,
        max_path_length: int = 64,
        max_candidates: int = 100,
        max_hops: int = 4,
    ):
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length
        self.max_path_length = max_path_length
        self.max_candidates = max_candidates
        self.max_hops = max_hops
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def path_to_text(self, path: List[str]) -> str:
        """Convert relation path to text."""
        return " -> ".join(path)
    
    def detect_hop_boundaries(
        self, 
        path_tokens: torch.Tensor,  # [T]
        path: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect token boundaries for each hop in the path.
        
        Returns:
            hop_boundaries: [max_hops, 2] - (start, end) for each hop
            hop_mask: [max_hops] - which hops are valid
        """
        hop_boundaries = torch.zeros(self.max_hops, 2, dtype=torch.long)
        hop_mask = torch.zeros(self.max_hops, dtype=torch.bool)
        
        if len(path) == 0:
            return hop_boundaries, hop_mask
        
        # Tokenize each relation separately to find boundaries
        path_text = self.path_to_text(path)
        current_pos = 0
        
        for h, relation in enumerate(path):
            if h >= self.max_hops:
                break
            
            # Find where this relation starts in the full tokenization
            # This is approximate - we split on " -> "
            rel_tokens = self.tokenizer.encode(relation, add_special_tokens=False)
            rel_len = len(rel_tokens)
            
            # Skip CLS token position
            start_pos = max(1, current_pos)
            end_pos = min(start_pos + rel_len, self.max_path_length - 1)
            
            hop_boundaries[h, 0] = start_pos
            hop_boundaries[h, 1] = end_pos
            hop_mask[h] = True
            
            # Move to next hop (account for " -> " separator)
            separator_len = len(self.tokenizer.encode(" -> ", add_special_tokens=False))
            current_pos = end_pos + separator_len
        
        return hop_boundaries, hop_mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        question = sample['question']
        candidate_paths = sample['candidate_paths'][:self.max_candidates]
        label = sample['labels']
        
        # Ensure label is valid
        if label >= len(candidate_paths):
            label = 0  # Fallback
        
        num_candidates = len(candidate_paths)
        
        # Tokenize question
        q_enc = self.tokenizer(
            question,
            max_length=self.max_question_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Tokenize candidate paths
        path_texts = [self.path_to_text(p) for p in candidate_paths]
        
        # Pad to max_candidates
        while len(path_texts) < self.max_candidates:
            path_texts.append("")
        
        p_enc = self.tokenizer(
            path_texts,
            max_length=self.max_path_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Create candidate mask
        candidate_mask = torch.zeros(self.max_candidates)
        candidate_mask[:num_candidates] = 1
        
        # Detect hop boundaries for each candidate
        all_hop_boundaries = []
        all_hop_masks = []
        
        for i, path in enumerate(candidate_paths):
            boundaries, mask = self.detect_hop_boundaries(p_enc['input_ids'][i], path)
            all_hop_boundaries.append(boundaries)
            all_hop_masks.append(mask)
        
        # Pad hop boundaries
        while len(all_hop_boundaries) < self.max_candidates:
            all_hop_boundaries.append(torch.zeros(self.max_hops, 2, dtype=torch.long))
            all_hop_masks.append(torch.zeros(self.max_hops, dtype=torch.bool))
        
        hop_boundaries = torch.stack(all_hop_boundaries)  # [C, max_hops, 2]
        hop_mask = torch.stack(all_hop_masks)  # [C, max_hops]
        
        return {
            'question_input_ids': q_enc['input_ids'].squeeze(0),
            'question_attention_mask': q_enc['attention_mask'].squeeze(0),
            'path_input_ids': p_enc['input_ids'],
            'path_attention_mask': p_enc['attention_mask'],
            'candidate_mask': candidate_mask,
            'hop_boundaries': hop_boundaries,
            'hop_mask': hop_mask,
            'labels': torch.tensor(label, dtype=torch.long),
        }


class PathRankingDataModule(pl.LightningDataModule):
    """Data module for path ranking training."""
    
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: Optional[str] = None,
        tokenizer_name: str = "BAAI/bge-base-en-v1.5",
        max_question_length: int = 128,
        max_path_length: int = 64,
        max_candidates: int = 100,
        max_hops: int = 4,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.tokenizer_name = tokenizer_name
        self.max_question_length = max_question_length
        self.max_path_length = max_path_length
        self.max_candidates = max_candidates
        self.max_hops = max_hops
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.tokenizer = None
    
    def setup(self, stage: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        if stage == "fit" or stage is None:
            self.train_dataset = PathRankingDataset(
                self.train_path,
                self.tokenizer,
                self.max_question_length,
                self.max_path_length,
                self.max_candidates,
                self.max_hops,
            )
            self.val_dataset = PathRankingDataset(
                self.val_path,
                self.tokenizer,
                self.max_question_length,
                self.max_path_length,
                self.max_candidates,
                self.max_hops,
            )
        
        if stage == "test" or stage is None:
            if self.test_path:
                self.test_dataset = PathRankingDataset(
                    self.test_path,
                    self.tokenizer,
                    self.max_question_length,
                    self.max_path_length,
                    self.max_candidates,
                    self.max_hops,
                )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config: Dict[str, Any]) -> HopColBERTReranker:
    """Create model from configuration dictionary."""
    model_cfg = config.get('model', {})
    late_int_cfg = config.get('late_interaction', {})
    hop_aux_cfg = config.get('hop_auxiliary', {})
    loss_cfg = config.get('loss', {})
    train_cfg = config.get('training', {})
    
    model = HopColBERTReranker(
        encoder_name=model_cfg.get('encoder_name', 'BAAI/bge-base-en-v1.5'),
        hidden_dim=model_cfg.get('hidden_dim', 768),
        projection_dim=model_cfg.get('projection_dim', 128),
        max_hops=model_cfg.get('max_hops', 4),
        max_question_length=model_cfg.get('max_question_length', 128),
        max_path_length=model_cfg.get('max_path_length', 64),
        dropout=model_cfg.get('dropout', 0.1),
        learning_rate=train_cfg.get('learning_rate', 2e-5),
        weight_decay=train_cfg.get('weight_decay', 0.01),
        warmup_ratio=train_cfg.get('warmup_ratio', 0.1),
        max_steps=train_cfg.get('max_steps', 50000),
        freeze_encoder=model_cfg.get('freeze_encoder', False),
        late_interaction_temp=late_int_cfg.get('temperature', 0.02),
        normalize_embeds=late_int_cfg.get('normalize', True),
        hop_aux_weight=hop_aux_cfg.get('weight', 0.3) if hop_aux_cfg.get('enabled', True) else 0.0,
        hop_loss_weights=hop_aux_cfg.get('loss_weights'),
        hop_aux_cumulative=hop_aux_cfg.get('cumulative', True),
        primary_loss_temp=loss_cfg.get('primary_temperature', 1.0),
        margin_weight=loss_cfg.get('margin_weight', 0.1),
        margin=loss_cfg.get('margin', 1.0),
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train HopColBERT Reranker')
    parser.add_argument('--config', type=str, default='configs/hop_colbert.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--weights_only', action='store_true',
                        help='Load only model weights from checkpoint, not optimizer state')
    parser.add_argument('--fast_dev_run', action='store_true',
                        help='Run quick test')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Override max steps')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir or f"outputs_hop_colbert_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    print(f"Output directory: {output_dir}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Create model
    model = create_model_from_config(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data module - using PathRankerDataModule which handles parquet files
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})
    
    # Support both single path and multiple paths
    train_paths = data_cfg.get('train_paths', data_cfg.get('train_path', []))
    val_paths = data_cfg.get('val_paths', data_cfg.get('val_path', None))
    test_paths = data_cfg.get('test_paths', data_cfg.get('test_path', None))
    
    data_module = PathRankerDataModule(
        train_path=train_paths,
        val_path=val_paths,
        test_path=test_paths,
        tokenizer_name=config['model']['encoder_name'],
        batch_size=train_cfg.get('batch_size', 16),
        num_workers=data_cfg.get('num_workers', 4),
        max_question_length=config['model'].get('max_question_length', 128),
        max_path_length=config['model'].get('max_path_length', 64),
        max_candidates=data_cfg.get('max_candidates', 100),
        num_negatives=data_cfg.get('num_negatives', 99),
    )
    
    # Callbacks
    ckpt_cfg = config.get('checkpointing', {})
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='hop_colbert-{epoch:02d}-{val/acc:.4f}',
            save_top_k=ckpt_cfg.get('save_top_k', 3),
            monitor=ckpt_cfg.get('monitor', 'val/acc'),
            mode=ckpt_cfg.get('mode', 'max'),
            save_last=ckpt_cfg.get('save_last', True),
        ),
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(),
    ]
    
    # Logger
    logger = TensorBoardLogger(output_dir, name='logs')
    
    # Trainer
    log_cfg = config.get('logging', {})
    
    # Handle max_steps=-1 (use epochs only) - PyTorch Lightning uses -1 for unlimited
    max_steps_val = train_cfg.get('max_steps', -1)
    if max_steps_val is None or max_steps_val == -1:
        max_steps_val = -1  # PyTorch Lightning uses -1 for unlimited steps
    
    # Multi-GPU strategy - need find_unused_parameters for hop auxiliary heads
    strategy = 'ddp_find_unused_parameters_true' if args.gpus > 1 else 'auto'
    
    trainer = pl.Trainer(
        max_steps=max_steps_val,
        max_epochs=train_cfg.get('max_epochs', 10),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus if torch.cuda.is_available() else 1,
        strategy=strategy,
        precision=train_cfg.get('precision', '16-mixed'),
        accumulate_grad_batches=train_cfg.get('gradient_accumulation_steps', 1),
        gradient_clip_val=train_cfg.get('gradient_clip_val', 1.0),
        log_every_n_steps=log_cfg.get('log_every_n_steps', 50),
        val_check_interval=log_cfg.get('val_check_interval', 0.25),
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
    )
    
    # Train
    print("\nStarting training...")
    
    # Handle weights_only checkpoint loading
    if args.checkpoint and args.weights_only:
        print(f"Loading model weights only from: {args.checkpoint}")
        import torch as torch_load
        ckpt = torch_load.load(args.checkpoint, map_location='cpu')
        # Load only the model state dict, not optimizer/scheduler
        # Use strict=False to allow architecture changes (e.g., removing final_scorer)
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['state_dict'], strict=False)
        if missing_keys:
            print(f"  Missing keys (will be randomly initialized): {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys (will be ignored): {unexpected_keys}")
        print("Model weights loaded. Starting fresh training with new optimizer.")
        trainer.fit(model, data_module)
    else:
        trainer.fit(model, data_module, ckpt_path=args.checkpoint)
    
    # Test if test data available
    if data_cfg.get('test_path'):
        print("\nRunning test evaluation...")
        trainer.test(model, data_module, ckpt_path='best')
    
    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
