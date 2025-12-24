"""
Training script for KG Retriever models (Diffusion or GNN).

Usage:
    python train.py --config configs/default.yaml        # Diffusion
    python train.py --config configs/gnn_retriever.yaml  # GNN
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.diffusion_retriever import KGDiffusionRetriever
from models.gnn_retriever import GNNRetriever
from data.dataset import KGRetrieverDataModule


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Train KG Diffusion Retriever')
    
    # Config file
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    
    # Override options
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
    if args.gpus is not None:
        config['gpus'] = args.gpus
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.debug:
        config['debug'] = True
    
    # Set precision for Tensor Cores
    torch.set_float32_matmul_precision('medium')
    
    # Set seed
    pl.seed_everything(config.get('seed', 42), workers=True)
    
    # Create output directory
    output_dir = config.get('output_dir', 'kg_diffusion_retriever')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("KG Diffusion Retriever Training")
    print("=" * 60)
    print(f"Train data: {config['train_data']}")
    print(f"Val data: {config.get('val_data', 'None')}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Hidden dim: {config['hidden_dim']}")
    print(f"Max triples: {config['max_triples']}")
    print(f"Max path length: {config['max_path_length']}")
    print("=" * 60)
    
    # Create data module
    print("\nSetting up data module...")
    data_module = KGRetrieverDataModule(
        train_path=config['train_data'],
        val_path=config.get('val_data'),
        test_path=config.get('test_data'),
        vocab_path=config.get('vocab_path'),
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
        tokenizer_name=config.get('tokenizer_name', 'sentence-transformers/all-MiniLM-L6-v2'),
        max_question_length=config.get('max_question_length', 64),
        max_triples=config['max_triples'],
        max_path_length=config['max_path_length'],
        num_entity_buckets=config.get('num_entity_buckets', 10000),
    )
    
    # Setup data
    data_module.setup('fit')
    
    num_relations = data_module.num_relations
    print(f"Vocabulary: {num_relations} relations")
    
    # Save vocabulary
    vocab_save_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_save_path, 'w') as f:
        json.dump({
            'relation_to_idx': data_module.relation_to_idx,
            'num_relations': num_relations,
        }, f, indent=2)
    print(f"Saved vocabulary to {vocab_save_path}")
    
    # Create model based on type
    model_type = config.get('model_type', 'kg_diffusion_retriever')
    print(f"\nCreating model: {model_type}")
    
    if model_type == 'gnn_retriever':
        model = GNNRetriever(
            num_relations=num_relations,
            hidden_dim=config['hidden_dim'],
            num_gnn_layers=config.get('num_gnn_layers', 3),
            num_heads=config['num_heads'],
            max_path_length=config['max_path_length'],
            num_entity_buckets=config.get('num_entity_buckets', 50000),
            dropout=config.get('dropout', 0.1),
            learning_rate=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01),
            warmup_steps=config.get('warmup_steps', 2000),
            max_steps=config.get('max_steps', 100000),
            question_encoder_name=config.get('question_encoder', 'sentence-transformers/all-MiniLM-L6-v2'),
            freeze_question_encoder=config.get('freeze_question_encoder', True),
        )
    else:  # Default: diffusion retriever
        model = KGDiffusionRetriever(
            num_relations=num_relations,
            hidden_dim=config['hidden_dim'],
            kg_feature_dim=config.get('kg_feature_dim', config['hidden_dim']),
            num_layers=config.get('num_layers', 4),
            num_heads=config['num_heads'],
            num_diffusion_steps=config.get('num_diffusion_steps', 20),
            max_path_length=config['max_path_length'],
            num_entity_buckets=config.get('num_entity_buckets', 50000),
            dropout=config.get('dropout', 0.1),
            learning_rate=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01),
            warmup_steps=config.get('warmup_steps', 2000),
            max_steps=config.get('max_steps', 100000),
            question_encoder_name=config.get('question_encoder', 'sentence-transformers/all-MiniLM-L6-v2'),
            freeze_question_encoder=config.get('freeze_question_encoder', True),
            label_smoothing=config.get('label_smoothing', 0.0),
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
    ]
    
    if not config.get('debug', False):
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(output_dir, 'checkpoints'),
                filename='kg_retriever-{epoch:02d}-{val/loss:.4f}',
                save_top_k=3,
                monitor='val/loss',
                mode='min',
                save_last=True,
            )
        )
    
    early_stopping_patience = config.get('early_stopping_patience', 20)
    if early_stopping_patience > 0 and config.get('val_data'):
        callbacks.append(
            EarlyStopping(
                monitor='val/loss',
                patience=early_stopping_patience,
                mode='min',
            )
        )
    
    # Setup logger
    if config.get('wandb', False):
        logger = WandbLogger(
            project=config.get('wandb_project', 'kg-diffusion-retriever'),
            name=config.get('experiment_name', 'kg_diffusion_retriever'),
            save_dir=output_dir,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=output_dir,
            name=config.get('experiment_name', 'kg_diffusion_retriever'),
        )
    
    # Determine GPUs
    # GPU configuration - handle both int and list formats
    gpus_config = config.get('gpus', -1)
    if isinstance(gpus_config, list):
        # List format: [0] or [0, 1]
        devices = gpus_config
        num_gpus = len(gpus_config)
    elif gpus_config == -1:
        # Auto-detect
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_gpus = devices if isinstance(devices, int) else len(devices)
    else:
        # Int format: 2
        devices = gpus_config
        num_gpus = gpus_config
    
    print(f"\nUsing {num_gpus} GPU(s): {devices}")
    
    # Strategy
    strategy = config.get('strategy', 'auto')
    if strategy == 'ddp_find_unused_parameters_true' and num_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    elif strategy == 'ddp' and num_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    elif num_gpus <= 1:
        strategy = 'auto'
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 200),
        max_steps=config.get('max_steps', 100000),
        accelerator='gpu' if num_gpus > 0 else 'cpu',
        devices=devices if num_gpus > 0 else 1,
        strategy=strategy if num_gpus > 1 else 'auto',
        precision=config.get('precision', '16-mixed'),
        gradient_clip_val=config.get('gradient_clip', 1.0),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 4),
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=config.get('check_val_every_n_epoch', 5),
        fast_dev_run=config.get('debug', False),
        deterministic=True,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume,
    )
    
    # Test
    if config.get('test_data'):
        print("\nRunning test evaluation...")
        data_module.setup('test')
        trainer.test(model, datamodule=data_module)
    
    print("\nTraining complete!")
    for cb in callbacks:
        if hasattr(cb, 'best_model_path') and cb.best_model_path:
            print(f"Best model checkpoint: {cb.best_model_path}")
            break


if __name__ == '__main__':
    main()
