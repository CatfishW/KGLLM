"""
Training script for Path Ranker Model.
"""

import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.path_ranker import PathRankerModel
from data.path_ranker_dataset import PathRankerDataModule


def train(config_path: str):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    pl.seed_everything(config.get('seed', 42))
    
    # Setup data
    data_module = PathRankerDataModule(
        train_path=config['train_data'],
        val_path=config.get('val_data'),
        test_path=config.get('test_data'),
        tokenizer_name=config.get('encoder_name', "BAAI/bge-base-en-v1.5"),
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 4),
        max_question_length=config.get('max_question_length', 128),
        max_path_length=config.get('max_path_length', 64),
        max_candidates=config.get('max_candidates', 100),
        num_negatives=config.get('num_negatives', 99),
    )
    
    # Setup model
    model = PathRankerModel(
        encoder_name=config.get('encoder_name', "BAAI/bge-base-en-v1.5"),
        hidden_dim=config.get('hidden_dim', 768),
        dropout=config.get('dropout', 0.1),
        learning_rate=float(config.get('learning_rate', 2e-5)),
        weight_decay=config.get('weight_decay', 0.01),
        warmup_steps=config.get('warmup_steps', 1000),
        max_steps=config.get('max_steps', 50000),
        freeze_encoder=config.get('freeze_encoder', False),
        temperature=config.get('temperature', 0.05),
        max_candidates=config.get('max_candidates', 100),
        max_path_length=config.get('max_path_length', 64),
    )
    
    # Setup callbacks
    output_dir = config.get('output_dir', 'outputs')
    experiment_name = config.get('experiment_name', 'path_ranker')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename=f'{experiment_name}-{{epoch:02d}}-{{val/acc:.4f}}',
        monitor='val/acc',
        mode='max',
        save_top_k=3,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callbacks = [checkpoint_callback, lr_monitor]
    
    if config.get('early_stopping_patience', 0) > 0:
        early_stopping = EarlyStopping(
            monitor='val/acc',
            mode='max',
            patience=config['early_stopping_patience'],
        )
        callbacks.append(early_stopping)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=experiment_name,
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 20),
        max_steps=config.get('max_steps', 50000),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config.get('gpus', [0]),
        strategy=config.get('strategy', 'auto'),
        precision=config.get('precision', 16),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.get('gradient_clip', 1.0),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        val_check_interval=config.get('val_check_interval', 1.0),
        log_every_n_steps=50,
    )
    
    # Train
    print(f"Starting training: {experiment_name}")
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)
