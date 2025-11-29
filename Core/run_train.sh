#!/bin/bash
# ============================================================
# KG Path Diffusion Model - Training Script (Linux/Mac)
# ============================================================
# Usage: ./run_train.sh [conda_env_name] [config_path]
#   Example: ./run_train.sh Wu ./configs/flow_matching_base.yaml
# ============================================================

set -e

# Default conda environment and config
# CONDA_ENV="${1:-Wu}"
CONFIG_PATH="${2:-./configs/diffusion.yaml}"

# echo "Config File    : $CONFIG_PATH"
# echo "============================================================"
# echo "KG Path Diffusion Model - Training"
# echo "============================================================"
# echo "Conda Environment: $CONDA_ENV"
# echo "============================================================"

# # Activate conda environment
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "$CONDA_ENV"

# Run training with config-based settings
python train.py --config "$CONFIG_PATH"

echo ""
echo "============================================================"
echo "Training complete!"
echo "============================================================"

