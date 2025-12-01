# Project Structure Reorganization

## New Structure (INTP-style)

```
Core/
├── models/                    # All model implementations
│   ├── __init__.py           # Exports all models
│   ├── base.py               # Shared components (QuestionEncoder)
│   ├── diffusion.py          # Diffusion model
│   ├── autoregressive.py     # Autoregressive model
│   ├── gnn_decoder.py        # GNN+Decoder model
│   └── factory.py            # Model factory
│
├── data/                      # Data handling (existing)
│   ├── __init__.py
│   └── dataset.py
│
├── training/                  # Training scripts and utilities
│   ├── __init__.py
│   ├── train.py              # Main training script
│   └── callbacks/            # Training callbacks
│       ├── __init__.py
│       └── path_examples_logger.py
│
├── utils/                     # Utility scripts
│   ├── __init__.py
│   ├── inference.py          # Inference script
│   ├── evaluate.py           # Evaluation script
│   ├── data_prep.py          # Data preparation
│   └── test_models.py        # Model testing
│
├── modules/                   # Low-level modules (existing)
│   ├── __init__.py
│   ├── diffusion.py
│   └── graph_compression.py
│
├── configs/                   # Configuration files (existing)
│   ├── diffusion.yaml
│   ├── autoregressive.yaml
│   └── gnn_decoder.yaml
│
├── docs/                      # Documentation (existing)
│   ├── SOTA_RELATION_CHAIN_METHODS.md
│   └── ...
│
├── scripts/                   # Shell/batch scripts
│   ├── run_train.sh
│   ├── run_train.bat
│   └── ...
│
└── outputs/                   # Training outputs (optional, can stay in root)
    └── ...
```

## Key Principles

1. **Logical Separation**: Models, data, training, utils are clearly separated
2. **Single Responsibility**: Each directory has one clear purpose
3. **Easy Navigation**: Related files are grouped together
4. **Extensibility**: Easy to add new models or utilities
5. **Clean Imports**: Clear import paths

## Migration Steps

1. Create new directory structure
2. Move files to appropriate locations
3. Update all imports
4. Create __init__.py files
5. Update scripts and configs
6. Test everything works

