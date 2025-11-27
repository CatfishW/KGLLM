#!/bin/bash
# ============================================================
# Quick Test - Run inference on a few samples (Linux/Mac)
# ============================================================
# Usage: ./run_quick_test.sh [conda_env_name]
# ============================================================

set -e

CONDA_ENV="${1:-Wu}"

echo "============================================================"
echo "Quick Test - Inference on 10 samples"
echo "============================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python inference.py \
    --checkpoint outputs_1/checkpoints/last.ckpt \
    --vocab outputs_1/vocab.json \
    --data ../Data/webqsp_combined/val.jsonl \
    --output quick_test_results.jsonl \
    --max_samples 10 \
    --batch_size 2 \
    --show_examples 10

