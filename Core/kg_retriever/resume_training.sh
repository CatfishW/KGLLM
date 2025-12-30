#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="outputs_dcdr_${TIMESTAMP}_bge-base-110M_unfrozen"
echo "Training output dir: $OUTDIR"

# Ensure conda env
source /home/benwulab/anaconda3/etc/profile.d/conda.sh
conda activate KGLLM

python train_dcdr.py \
    --encoder_name BAAI/bge-base-en-v1.5 \
    --hidden_dim 768 \
    --num_layers 4 \
    --num_heads 8 \
    --train_data /data/Yanlai/KGLLM/Data/preprocessed_paths/webqsp_train.parquet /data/Yanlai/KGLLM/Data/preprocessed_paths/cwq_train.parquet \
    --val_data /data/Yanlai/KGLLM/Data/preprocessed_paths/webqsp_val.parquet \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_epochs 300 \
    --gpus 2 \
    --num_workers 8 \
    --unfreeze_encoder \
    --output_dir "$OUTDIR" \
    --experiment_name "bge-base-110M-unfrozen-joint" \
    --load_checkpoint outputs_dcdr_20251229_013640_bge-base-110M_unfrozen/last.ckpt \
    2>&1 | tee training_resume_${TIMESTAMP}.log
