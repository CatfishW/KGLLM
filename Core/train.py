"""
Training script for KG Path Diffusion Model.

Features:
- Multi-GPU training with DDP
- Mixed precision (AMP) for faster training
- Gradient accumulation
- Checkpointing and logging
- Easy configuration via command line or config file
"""
# python train.py --train_data ../Data/webqsp_combined/train_combined.parquet --val_data ../Data/webqsp_combined/val.jsonl --batch_size 4 --hidden_dim 128 --num_graph_layers 2 --num_diffusion_layers 2 --num_diffusion_steps 100 --max_path_length 20 --gpus 1 --output_dir outputs_1
import os
import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import KGPathDataModule
from kg_path_diffusion import KGPathDiffusionLightning


def parse_args():
    parser = argparse.ArgumentParser(description='Train KG Path Diffusion Model')
    
    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data (parquet or jsonl)')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data')
    parser.add_argument('--vocab_path', type=str, default=None,
                        help='Path to pre-built vocabulary (recommended for avoiding UNK tokens)')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension size')
    parser.add_argument('--num_graph_layers', type=int, default=3,
                        help='Number of graph encoder layers')
    parser.add_argument('--num_diffusion_layers', type=int, default=6,
                        help='Number of diffusion transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_diffusion_steps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--max_path_length', type=int, default=20,
                        help='Maximum path length')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--graph_encoder', type=str, default='hybrid',
                        choices=['rgcn', 'transformer', 'hybrid'],
                        help='Type of graph encoder')
    parser.add_argument('--question_encoder', type=str, 
                        default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Pretrained question encoder model')
    parser.add_argument('--freeze_question_encoder', action='store_true',
                        help='Freeze question encoder weights')
    parser.add_argument('--max_vocab_size', type=int, default=50000,
                        help='Maximum vocabulary size (to limit memory usage)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum training steps')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Gradient accumulation steps')
    
    # Hardware arguments
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs (-1 for all available)')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        choices=['16-mixed', 'bf16-mixed', '32'],
                        help='Training precision')
    parser.add_argument('--strategy', type=str, default='auto',
                        choices=['auto', 'ddp', 'ddp_find_unused_parameters_true', 'deepspeed_stage_2'],
                        help='Training strategy')
    
    # Logging arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='kg_path_diffusion',
                        help='Experiment name')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='kg-path-diffusion',
                        help='W&B project name')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (fast dev run)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("KG Path Diffusion Model Training")
    print("="*60)
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data}")
    print(f"Vocab path: {args.vocab_path if args.vocab_path else '(will build from train data)'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Precision: {args.precision}")
    print("="*60)
    
    # Create data module
    print("\nSetting up data module...")
    data_module = KGPathDataModule(
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data,
        vocab_path=args.vocab_path,  # Use pre-built vocabulary if provided
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_path_length=args.max_path_length,
        max_graph_nodes=200  # Limit graph size for memory
    )
    
    # Setup data to build vocabulary
    data_module.setup('fit')
    
    num_entities = data_module.vocab.num_entities
    num_relations = data_module.vocab.num_relations
    
    print(f"Vocabulary: {num_entities} entities, {num_relations} relations")
    
    # Save vocabulary (useful for inference and debugging)
    output_vocab_path = os.path.join(args.output_dir, 'vocab.json')
    data_module.vocab.save(output_vocab_path)
    print(f"Saved vocabulary to {output_vocab_path}")
    
    # Create model
    print("\nCreating model...")
    model = KGPathDiffusionLightning(
        num_entities=num_entities,
        num_relations=num_relations,
        hidden_dim=args.hidden_dim,
        question_encoder_name=args.question_encoder,
        freeze_question_encoder=args.freeze_question_encoder,
        graph_encoder_type=args.graph_encoder,
        num_graph_layers=args.num_graph_layers,
        num_diffusion_layers=args.num_diffusion_layers,
        num_heads=args.num_heads,
        num_diffusion_steps=args.num_diffusion_steps,
        max_path_length=args.max_path_length,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps
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
    
    # Only add checkpointing if not in debug mode (saves memory)
    if not args.debug:
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(args.output_dir, 'checkpoints'),
                filename='kg_path_diffusion-{epoch:02d}-{val_loss:.4f}',
                save_top_k=3,
                monitor='val/loss',
                mode='min',
                save_last=True
            )
        )
    
    if args.val_data:
        callbacks.append(
            EarlyStopping(
                monitor='val/loss',
                patience=10,
                mode='min'
            )
        )
    
    # Setup logger
    if args.wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.experiment_name,
            save_dir=args.output_dir
        )
    else:
        logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name=args.experiment_name
        )
    
    # Determine number of GPUs
    if args.gpus == -1:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = args.gpus
    
    print(f"\nUsing {num_gpus} GPU(s)")
    
    # Setup strategy
    if args.strategy == 'ddp' and num_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    elif args.strategy == 'ddp_find_unused_parameters_true' and num_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = args.strategy
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator='gpu' if num_gpus > 0 else 'cpu',
        devices=num_gpus if num_gpus > 0 else 1,
        strategy=strategy if num_gpus > 1 else 'auto',
        precision=args.precision,
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        val_check_interval=0.25,  # Validate 4 times per epoch
        fast_dev_run=args.debug,
        deterministic=True
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume
    )
    
    # Test if test data is provided
    if args.test_data:
        print("\nRunning test evaluation...")
        trainer.test(model, datamodule=data_module)
    
    print("\nTraining complete!")
    # Find checkpoint callback if it exists
    for cb in callbacks:
        if hasattr(cb, 'best_model_path') and cb.best_model_path:
            print(f"Best model checkpoint: {cb.best_model_path}")
            break


if __name__ == '__main__':
    main()

