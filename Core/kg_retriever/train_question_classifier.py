"""
Training script for Question Type Classifier.

Trains a TinyBERT-based classifier for 3-class question type classification:
- one_hop: Single relation path questions
- multi_hop: Multi-relation path questions
- numeric: Count/quantity questions

Usage:
    python train_question_classifier.py --config configs/question_classifier.yaml
    python train_question_classifier.py --config configs/question_classifier.yaml --max_epochs 5
"""

import os
import sys
import yaml
import argparse
from datetime import datetime
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.question_classifier_dataset import QuestionClassifierDataModule, LABEL_NAMES
from models.question_classifier import QuestionTypeClassifier


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Question Type Classifier')
    parser.add_argument('--config', type=str, default='configs/question_classifier.yaml',
                        help='Path to config file')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Override max epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run testing on checkpoint')
    parser.add_argument('--limit_batches', type=int, default=None,
                        help='Limit batches for quick testing')
    parser.add_argument('--use_llm_corrected', action='store_true',
                        help='Use LLM corrected training data')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.max_epochs is not None:
        config['training']['max_epochs'] = args.max_epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.output_dir is not None:
        config['output']['output_dir'] = args.output_dir
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Question Type Classifier Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Encoder: {config['model']['encoder_name']}")
    print(f"Max epochs: {config['training']['max_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Use LLM corrected data: {args.use_llm_corrected}")
    print("=" * 60)
    
    # Create data module
    data_module = QuestionClassifierDataModule(
        data_dir=config['data']['data_dir'],
        tokenizer_name=config['model']['encoder_name'],
        max_length=config['data']['max_length'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        max_hops=config['data']['max_hops'],
        include_augmented=config['data']['include_augmented'],
        use_weighted_sampling=config['data']['use_weighted_sampling'],
        use_llm_corrected=args.use_llm_corrected,
    )
    
    # Setup data to get class weights
    data_module.setup('fit')
    
    # Calculate max steps for scheduler
    steps_per_epoch = len(data_module.train_dataset) // config['training']['batch_size']
    max_steps = steps_per_epoch * config['training']['max_epochs']
    
    # Create model
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = QuestionTypeClassifier.load_from_checkpoint(args.resume)
    else:
        model = QuestionTypeClassifier(
            encoder_name=config['model']['encoder_name'],
            num_classes=config['model']['num_classes'],
            hidden_dim=config['model']['hidden_dim'],
            dropout=config['model']['dropout'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            warmup_steps=config['training']['warmup_steps'],
            max_steps=max_steps,
            class_weights=data_module.class_weights,
            freeze_encoder_layers=config['model']['freeze_encoder_layers'],
        )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename='best_model_{epoch:02d}_{val_acc:.4f}',
            monitor='val_acc',
            mode='max',
            save_top_k=config['output']['save_top_k'],
        ),
        EarlyStopping(
            monitor='val_acc',
            mode='max',
            patience=config['training']['early_stopping_patience'],
            min_delta=config['training']['early_stopping_min_delta'],
        ),
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(),
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='logs',
    )
    
    # Trainer
    trainer_kwargs = {
        'max_epochs': config['training']['max_epochs'],
        'accelerator': config['hardware']['accelerator'],
        'devices': config['hardware']['devices'],
        'precision': config['hardware']['precision'],
        'accumulate_grad_batches': config['training']['accumulate_grad_batches'],
        'gradient_clip_val': config['training']['gradient_clip_val'],
        'callbacks': callbacks,
        'logger': logger,
        'log_every_n_steps': 50,
        'enable_progress_bar': True,
    }
    
    if args.limit_batches:
        trainer_kwargs['limit_train_batches'] = args.limit_batches
        trainer_kwargs['limit_val_batches'] = args.limit_batches
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    if args.test_only:
        # Test only
        data_module.setup('test')
        trainer.test(model, datamodule=data_module)
    else:
        # Train
        trainer.fit(model, datamodule=data_module)
        
        # Test with best checkpoint
        print("\n" + "=" * 60)
        print("Testing with best checkpoint...")
        print("=" * 60)
        
        best_model_path = callbacks[0].best_model_path
        print(f"Best model: {best_model_path}")
        
        data_module.setup('test')
        trainer.test(model, datamodule=data_module, ckpt_path=best_model_path)
        
        # Export to ONNX if requested
        if config['output']['export_onnx']:
            print("\n" + "=" * 60)
            print("Exporting to ONNX...")
            print("=" * 60)
            
            # Load best model
            best_model = QuestionTypeClassifier.load_from_checkpoint(best_model_path)
            tokenizer = AutoTokenizer.from_pretrained(config['model']['encoder_name'])
            
            onnx_path = output_dir / 'best_model.onnx'
            best_model.export_onnx(str(onnx_path), tokenizer)
            
            # Also save tokenizer
            tokenizer.save_pretrained(output_dir / 'tokenizer')
            print(f"Saved tokenizer to {output_dir / 'tokenizer'}")
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best validation accuracy: {callbacks[0].best_model_score:.4f}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)


if __name__ == '__main__':
    main()
