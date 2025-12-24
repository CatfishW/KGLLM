#!/bin/bash
# Full Training Pipeline for KGQA Optimization
# Uses conda env: KGLLM

set -e

echo "=============================================="
echo "KGQA Path Retriever Training Pipeline"
echo "Target: 90%+ F1 on WebQSP"
echo "=============================================="

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate KGLLM

cd /data/Yanlai/KGLLM/RAG

# Step 1: Generate Training Data
echo ""
echo "[Step 1/4] Generating training data with hard negatives..."
python -m RAG.training_data_generator \
    --dataset webqsp \
    --output-dir ./training_data \
    --num-hard 7 \
    --num-random 3

# Step 2: Fine-tune Dense Retriever
echo ""
echo "[Step 2/4] Fine-tuning dense retriever (stella_en_400M_v5 + MNRL)..."
python -m RAG.train_dense_retriever \
    --base-model dunzhang/stella_en_400M_v5 \
    --output-dir ./models/finetuned_retriever \
    --train-data ./training_data/train_st.jsonl \
    --val-data ./training_data/val_st.jsonl \
    --epochs 5 \
    --batch-size 32 \
    --lr 2e-5

# Step 3: Fine-tune Reranker with LoRA
echo ""
echo "[Step 3/4] Fine-tuning reranker (Qwen3-Reranker-0.6B + LoRA)..."
python -m RAG.train_reranker \
    --base-model Qwen/Qwen3-Reranker-0.6B \
    --output-dir ./models/finetuned_reranker \
    --train-data ./training_data/train_examples.json \
    --val-data ./training_data/val_examples.json \
    --epochs 3 \
    --batch-size 8 \
    --lora-r 16 \
    --lr 2e-4

# Step 4: Evaluate Optimized Pipeline
echo ""
echo "[Step 4/4] Evaluating optimized pipeline..."
python -m RAG.optimized_pipeline \
    --dataset webqsp \
    --split test \
    --limit 500 \
    --retriever ./models/finetuned_retriever \
    --reranker ./models/finetuned_reranker \
    --output-dir ./results/optimized_full

echo ""
echo "=============================================="
echo "Training Pipeline Complete!"
echo "=============================================="
echo "Results saved to: ./results/optimized_full/"
