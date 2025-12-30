"""
Training script for Discrete Diffusion Reranking (DCDR-style).
"""

import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from models.discrete_diffusion_reranker import DiscreteRankDiffusion
from data.path_ranker_dataset import PathRankerDataModule


def main(args):
    pl.seed_everything(args.seed)
    
    # Initialize model
    model = DiscreteRankDiffusion(
        encoder_name=args.encoder_name,
        hidden_dim=args.hidden_dim,
        num_diffusion_steps=args.num_diffusion_steps,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        swap_rate=args.swap_rate,
        pl_temperature=args.pl_temperature,
        max_question_length=args.max_question_length,
        max_path_length=args.max_path_length,
        freeze_encoder=not args.unfreeze_encoder,
    )
    
    # Load weights if checkpoint provided
    if args.load_checkpoint:
        print(f"Loading weights from {args.load_checkpoint}...")
        checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']
        # Load with strict=False to allow for missing keys (e.g. if we add new params)
        # or unexpected keys (e.g. if architecture changed slightly)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if len(missing) > 0:
            print(f"Sample missing: {missing[:5]}")
    
    # Data module
    data_module = PathRankerDataModule(
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data,
        tokenizer_name=args.encoder_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_question_length=args.max_question_length,
        max_path_length=args.max_path_length,
        max_candidates=args.max_candidates,
        num_negatives=args.num_negatives,
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename='dcdr-{epoch:02d}-{val/acc:.4f}',
            monitor='val/acc',
            mode='max',
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(
            monitor='val/acc',
            patience=args.patience,
            mode='max',
        ),
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='logs',
        version=args.experiment_name,
    )
    
    # Strategy for multi-GPU
    strategy = None
    if args.gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 'auto',
        strategy=strategy,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_progress_bar=True,
    )
    
    # Print config
    print(f"\n{'='*60}")
    print(f"Training Discrete Rank Diffusion (DCDR-style)")
    print(f"{'='*60}")
    print(f"Encoder: {args.encoder_name}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Transformer layers: {args.num_layers}")
    print(f"Diffusion steps: {args.num_diffusion_steps}")
    print(f"Swap rate: {args.swap_rate}")
    print(f"PL temperature: {args.pl_temperature}")
    print(f"Batch size: {args.batch_size} x {args.accumulate_grad_batches} x {args.gpus} = {args.batch_size * args.accumulate_grad_batches * args.gpus}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'='*60}\n")
    
    trainer.fit(model, data_module)
    
    if args.test_data:
        trainer.test(model, data_module, ckpt_path='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Discrete Rank Diffusion')
    
    # Data
    parser.add_argument('--train_data', type=str, nargs='+',
                        default=['/data/Yanlai/KGLLM/Data/preprocessed_paths/webqsp_train.parquet'])
    parser.add_argument('--val_data', type=str, nargs='+',
                        default=['/data/Yanlai/KGLLM/Data/preprocessed_paths/webqsp_val.parquet'])
    parser.add_argument('--test_data', type=str, nargs='+', default=None)
    
    # Model
    parser.add_argument('--encoder_name', type=str, default='BAAI/bge-small-en-v1.5')
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--num_diffusion_steps', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--swap_rate', type=float, default=0.5)
    parser.add_argument('--pl_temperature', type=float, default=1.0)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate_grad_batches', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    
    # Data processing
    parser.add_argument('--max_question_length', type=int, default=128)
    parser.add_argument('--max_path_length', type=int, default=64)
    parser.add_argument('--max_candidates', type=int, default=100)
    parser.add_argument('--num_negatives', type=int, default=99)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Environment
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs_dcdr')
    parser.add_argument('--experiment_name', type=str, default='default')
    
    # Inference / Fine-tuning
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to load weights from (strict=False)')
    parser.add_argument('--unfreeze_encoder', action='store_true',
                        help='Unfreeze encoder for fine-tuning')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
