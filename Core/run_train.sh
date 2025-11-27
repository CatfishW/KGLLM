#!/bin/bash
# ============================================================
# KG Path Diffusion Model - Training Script (Linux/Mac)
# ============================================================
# Usage: ./run_train.sh [conda_env_name]
#   Example: ./run_train.sh Wu
# ============================================================

set -e

# Default conda environment
CONDA_ENV="${1:-Wu}"

echo "============================================================"
echo "KG Path Diffusion Model - Training"
echo "============================================================"
echo "Conda Environment: $CONDA_ENV"
echo "============================================================"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Run training with recommended settings
python train.py \
    --train_data ../Data/webqsp_combined/train_combined.parquet \
    --val_data ../Data/webqsp_combined/val.jsonl \
    --batch_size 4 \
    --hidden_dim 128 \
    --num_graph_layers 2 \
    --num_diffusion_layers 2 \
    --num_diffusion_steps 100 \
    --max_path_length 20 \
    --gpus 1 \
    --output_dir outputs \
    --max_epochs 50

echo ""
echo "============================================================"
echo "Training complete!"
echo "============================================================"

