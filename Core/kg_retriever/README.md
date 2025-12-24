# KG Path Retriever

Neural path retriever for Knowledge Graph Question Answering (KGQA).
Two model architectures for ablation study:
1. **Diffusion Retriever** - Discrete diffusion for path generation
2. **GNN Retriever** - Message passing + autoregressive decoding (ReaRev-inspired)

## Project Structure

```
kg_retriever/
├── models/
│   ├── diffusion_retriever.py   # KG-conditioned discrete diffusion
│   └── gnn_retriever.py         # GNN message passing + RNN decoder
├── data/
│   └── dataset.py               # KGRetrieverDataset with full KG support
├── configs/
│   ├── default.yaml             # Diffusion retriever config
│   └── gnn_retriever.yaml       # GNN retriever config
├── experiments/                  # Experiment outputs
├── train.py                      # Unified training script
├── evaluate.py                   # Full-sample evaluation
└── README.md
```

## Model Architectures

### Diffusion Retriever (~44M trainable params)
- Question encoded with frozen MiniLM
- KG encoded via linear embeddings + mean pooling
- Discrete diffusion for iterative path denoising

### GNN Retriever (~12M trainable params)
- Question encoded with frozen MiniLM  
- Relation-aware GNN message passing (3 layers)
- GRU-based autoregressive path decoder

## Usage

### Train Diffusion Retriever
```bash
cd /data/Yanlai/KGLLM/Core/kg_retriever
python train.py --config configs/default.yaml
```

### Train GNN Retriever (ablation)
```bash
python train.py --config configs/gnn_retriever.yaml
```

### Evaluate
```bash
python evaluate.py \
  --checkpoint experiments/diffusion/checkpoints/last.ckpt \
  --test_data /data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/test.parquet
```

## Data Format

Uses full KG (up to 10K+ triples per question):
- `question`: Natural language question
- `graph`: List of (head, relation, tail) triples
- `shortest_gt_paths`: Ground truth reasoning paths

## References

- ReaRev: Adaptive Reasoning for KGQA (EMNLP 2022)
