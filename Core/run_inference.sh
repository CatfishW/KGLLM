#!/bin/bash
# ============================================================
# KG Path Diffusion Model - Inference Script (Linux/Mac)
# ============================================================
# Usage: ./run_inference.sh [conda_env_name] [checkpoint_dir]
#   Example: ./run_inference.sh Wu outputs_1
# ============================================================

set -e

# Default values
CONDA_ENV="${1:-Wu}"
CKPT_DIR="${2:-outputs_1}"

echo "============================================================"
echo "KG Path Diffusion Model - Inference"
echo "============================================================"
echo "Conda Environment: $CONDA_ENV"
echo "Checkpoint Dir: $CKPT_DIR"
echo "============================================================"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Check if checkpoint exists
if [ ! -f "$CKPT_DIR/checkpoints/last.ckpt" ]; then
    echo "ERROR: Checkpoint not found at $CKPT_DIR/checkpoints/last.ckpt"
    echo "Please train the model first or specify the correct checkpoint directory."
    exit 1
fi

# Run inference
python inference.py \
    --checkpoint "$CKPT_DIR/checkpoints/last.ckpt" \
    --vocab "$CKPT_DIR/vocab.json" \
    --data ../Data/webqsp_combined/val.jsonl \
    --output inference_results.jsonl \
    --batch_size 8 \
    --path_length 10 \
    --temperature 0.7 \
    --show_examples 10

echo ""
echo "============================================================"
echo "Inference complete! Results saved to inference_results.jsonl"
echo "============================================================"

