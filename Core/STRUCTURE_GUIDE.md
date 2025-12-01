# Project Structure Guide

## Overview

The project has been reorganized into a clean, logical structure that follows INTP-style organization principles:
- **Clear separation of concerns**
- **Logical grouping of related files**
- **Easy navigation and extensibility**
- **Clean, consistent import paths**

## Directory Structure

```
Core/
├── models/                    # 🧠 All model implementations
│   ├── __init__.py           #   Exports all models
│   ├── base.py               #   Shared components (QuestionEncoder)
│   ├── diffusion.py          #   Diffusion model
│   ├── autoregressive.py     #   Autoregressive model
│   ├── gnn_decoder.py        #   GNN+Decoder model
│   └── factory.py            #   Model factory
│
├── training/                  # 🎓 Training scripts and utilities
│   ├── __init__.py
│   ├── train.py              #   Main training script
│   └── callbacks/            #   Training callbacks
│       ├── __init__.py
│       └── path_examples_logger.py
│
├── data/                      # 📊 Data handling
│   ├── __init__.py
│   └── dataset.py
│
├── modules/                   # 🔧 Low-level modules
│   ├── __init__.py
│   ├── diffusion.py          #   Diffusion components
│   └── graph_compression.py
│
├── configs/                   # ⚙️ Configuration files
│   ├── diffusion.yaml
│   ├── autoregressive.yaml
│   └── gnn_decoder.yaml
│
├── docs/                      # 📚 Documentation
│   ├── SOTA_RELATION_CHAIN_METHODS.md
│   └── ...
│
└── [root files]               # Other files (scripts, outputs, etc.)
```

## Import Patterns

### Models
```python
# Recommended: Use package imports
from models import KGPathDiffusionLightning, create_model
from models.diffusion import KGPathDiffusionModel
from models.autoregressive import KGPathAutoregressiveLightning
from models.gnn_decoder import KGPathGNNDecoderLightning

# Or use factory
from models.factory import create_model
model = create_model('autoregressive', num_entities, num_relations, config)
```

### Training
```python
from training.callbacks import PathExamplesLogger
# or
from training import PathExamplesLogger
```

### Data
```python
from data.dataset import KGPathDataModule, KGPathDataset
```

## Key Principles

1. **Single Responsibility**: Each directory has one clear purpose
2. **DRY (Don't Repeat Yourself)**: Shared components in `models/base.py`
3. **Explicit is Better than Implicit**: Clear import paths
4. **Separation of Concerns**: Models, training, data are separate
5. **Extensibility**: Easy to add new models or utilities

## Adding New Models

1. Create model file in `models/` (e.g., `models/new_model.py`)
2. Import shared components from `models.base`
3. Add to `models/__init__.py`
4. Update `models/factory.py` to support new model type
5. Create config file in `configs/`

## File Organization Logic

- **models/**: All neural network model definitions
- **training/**: Everything related to training (scripts, callbacks)
- **data/**: Data loading and preprocessing
- **modules/**: Reusable low-level components
- **configs/**: Configuration files (YAML/JSON)
- **docs/**: Documentation and research notes
- **Root**: Scripts, outputs, and project-level files

## Benefits

✅ **Clear Navigation**: Know exactly where to find things  
✅ **Easy Maintenance**: Changes are localized  
✅ **Scalable**: Easy to add new components  
✅ **Professional**: Follows Python best practices  
✅ **INTP-Friendly**: Logical, systematic organization  

## Migration Notes

- Old import paths may still work for backward compatibility
- Gradually migrate to new import paths
- All functionality remains the same, just better organized

