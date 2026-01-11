"""
Training script for Entity Identifier Model.

Trains the bi-encoder entity linker with contrastive learning.

Usage:
    python train_entity_identifier.py --config configs/entity_identifier.yaml
"""

import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger

from models.entity_identifier import EntityIdentifierModel
from data.entity_linking_dataset import EntityLinkingDataModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(config: dict):
    """Run training with given configuration."""
    
    # Extract config sections
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    logging_config = config.get("logging", {})
    checkpoint_config = config.get("checkpointing", {})
    
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(logging_config.get("output_dir", "outputs_entity_identifier"))
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Initialize data module - support both singular and plural path configs
    train_path = data_config.get("train_paths") or data_config.get("train_path")
    val_path = data_config.get("val_paths") or data_config.get("val_path")
    test_path = data_config.get("test_paths") or data_config.get("test_path")
    
    data_module = EntityLinkingDataModule(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        tokenizer_name=model_config.get("encoder_name", "BAAI/bge-small-en-v1.5"),
        max_seq_length=model_config.get("max_seq_length", 128),
        num_negatives=data_config.get("num_negatives", 7),
        batch_size=training_config.get("batch_size", 32),
        num_workers=data_config.get("num_workers", 4),
    )
    
    # Initialize model
    model = EntityIdentifierModel(
        encoder_name=model_config.get("encoder_name", "BAAI/bge-small-en-v1.5"),
        hidden_size=model_config.get("hidden_size", 384),
        embedding_size=model_config.get("embedding_size", 256),
        use_gliner=model_config.get("use_gliner", True),
        gliner_model=model_config.get("gliner_model", "urchade/gliner_medium-v2.1"),
        freeze_encoder=model_config.get("freeze_encoder", False),
        learning_rate=training_config.get("learning_rate", 2e-5),
        warmup_steps=training_config.get("warmup_steps", 500),
        temperature=training_config.get("temperature", 0.05),
        max_seq_length=model_config.get("max_seq_length", 128),
    )
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="entity_identifier-{epoch:02d}-{val/accuracy:.4f}",
            save_top_k=checkpoint_config.get("save_top_k", 3),
            monitor=checkpoint_config.get("monitor", "val/accuracy"),
            mode=checkpoint_config.get("mode", "max"),
            save_last=True,
        ),
        EarlyStopping(
            monitor=checkpoint_config.get("monitor", "val/accuracy"),
            patience=5,
            mode=checkpoint_config.get("mode", "max"),
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Set up logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="tensorboard",
        version="",
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=training_config.get("max_epochs", 10),
        accelerator="auto",
        devices="auto",
        strategy="ddp_find_unused_parameters_true",  # GLiNER not used in training
        precision=training_config.get("precision", "16-mixed"),
        gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
        log_every_n_steps=logging_config.get("log_every_n_steps", 50),
        val_check_interval=logging_config.get("val_check_interval", 0.5),
        callbacks=callbacks,
        logger=tb_logger,
        enable_progress_bar=True,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    # Test if test data available
    if data_config.get("test_path"):
        logger.info("Running test evaluation...")
        trainer.test(model, data_module, ckpt_path="best")
    
    logger.info(f"Training complete! Checkpoints saved to {output_dir / 'checkpoints'}")
    
    return model, output_dir


def main():
    parser = argparse.ArgumentParser(description="Train Entity Identifier Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/entity_identifier.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
