# KGLLM Project Structure

## Directory Layout

```
KGLLM/
├── Core/           # Model implementations (diffusion, autoregressive, GNN)
├── Data/           # Datasets (WebQSP, CWQ, vocabularies)
├── EXP/            # Experiment configurations and demos
├── GSR/            # GraphRAG Subgraph Reasoning module
├── RAG/            # Retrieval-Augmented Generation pipeline
├── scripts/        # All standalone scripts (timestamped)
│   ├── evaluation/     # Evaluation and experiment scripts
│   ├── data_processing/# Data preparation and checking
│   └── utilities/      # Shell scripts and one-off tools
├── logs/           # Log files
├── outputs/        # Results and output files
├── checkpoints/    # Model checkpoints
└── wandb/          # Weights & Biases logging
```

## File Naming Convention

Scripts are prefixed with timestamps: `YYYYMMDD_scriptname.py`
- Example: `20251221_evaluate_final.py` was last modified on Dec 21, 2025

## Quick Navigation

| Task | Location |
|------|----------|
| Train models | `Core/train.py` |
| Run inference | `Core/inference.py` |
| Evaluate | `scripts/evaluation/` |
| Process data | `scripts/data_processing/` |
| RAG pipeline | `RAG/pipeline.py` |
| Experiment configs | `EXP/` |
