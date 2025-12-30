#!/bin/bash
# Script to wait for current training and launch finetuning

# Config
PREV_OUTPUT_DIR="outputs_dcdr_tuned"
NEW_OUTPUT_DIR="outputs_dcdr_tuned_unfrozen"
NEW_EXP_NAME="training_20251228_finetune_unfrozen"

echo "Waiting for checkpoints in $PREV_OUTPUT_DIR..."

# Wait until a checkpoint appears (or check if done)
# We'll just look for the best checkpoint.
# If process is still running, this might pick an intermediate one.
# But user said "After this training finishes".
# So this script assumes the user runs it AFTER training finishes.

CKPT_PATH=$(ls -t $PREV_OUTPUT_DIR/*.ckpt | head -n 1)

if [ -z "$CKPT_PATH" ]; then
    echo "No checkpoint found in $PREV_OUTPUT_DIR"
    exit 1
fi

echo "Found latest checkpoint: $CKPT_PATH"

echo "Launching Fine-tuning (Unfrozen Encoder)..."
# Using smaller batch size/accumulation for unfrozen training to save memory/stability?
# Keeping same config but unfreezing.

conda activate KGLLM && PYTHONUNBUFFERED=1 python train_dcdr.py \
    --train_data /data/Yanlai/KGLLM/Data/preprocessed_paths/webqsp_train.parquet /data/Yanlai/KGLLM/Data/preprocessed_paths/cwq_train.parquet \
    --val_data /data/Yanlai/KGLLM/Data/preprocessed_paths/webqsp_val.parquet /data/Yanlai/KGLLM/Data/preprocessed_paths/cwq_val.parquet \
    --output_dir $NEW_OUTPUT_DIR \
    --experiment_name $NEW_EXP_NAME \
    --encoder_name BAAI/bge-small-en-v1.5 \
    --hidden_dim 768 \
    --num_layers 4 \
    --num_heads 8 \
    --batch_size 8 \
    --accumulate_grad_batches 8 \
    --max_epochs 100 \
    --gpus 2 \
    --learning_rate 2e-5 \
    --check_val_every_n_epoch 5 \
    --load_checkpoint "$CKPT_PATH" \
    --unfreeze_encoder \
    2>&1 | tee training_dcdr_finetune_unfrozen.log

echo "Fine-tuning launched!"
