# New Model Implementations

This document describes the new model implementations added to the codebase.

## Overview

Two new model architectures have been implemented as alternatives to the existing diffusion model:

1. **Autoregressive Transformer** (`kg_path_autoregressive.py`) - GPT-style autoregressive model
2. **GNN + Decoder** (`kg_path_gnn_decoder.py`) - Hybrid model with GNN encoder and autoregressive decoder

## Files Created

### Model Files
- `Core/kg_path_autoregressive.py` - Autoregressive transformer model
- `Core/kg_path_gnn_decoder.py` - GNN + Decoder hybrid model
- `Core/model_factory.py` - Factory to create models based on config

### Configuration Files
- `Core/configs/autoregressive.yaml` - Config for autoregressive model
- `Core/configs/gnn_decoder.yaml` - Config for GNN decoder model

### Test Files
- `Core/test_new_models.py` - Test script to verify implementations

## Usage

### Using Autoregressive Model

```bash
python train.py --config configs/autoregressive.yaml
```

Or via command line:
```bash
python train.py \
    --model_type autoregressive \
    --train_data ../Data/webqsp_final/train.parquet \
    --val_data ../Data/webqsp_final/val.parquet \
    --decoding_strategy beam_search \
    --beam_width 5
```

### Using GNN Decoder Model

```bash
python train.py --config configs/gnn_decoder.yaml
```

Or via command line:
```bash
python train.py \
    --model_type gnn_decoder \
    --train_data ../Data/webqsp_final/train.parquet \
    --val_data ../Data/webqsp_final/val.parquet \
    --gnn_type gat \
    --gnn_layers 3
```

### Using Original Diffusion Model

The original diffusion model still works as before:
```bash
python train.py --config configs/diffusion.yaml
```

Or explicitly:
```bash
python train.py --model_type diffusion ...
```

## Model Selection

The `model_type` parameter in config or command line selects which model to use:
- `diffusion` - Original discrete diffusion model (default)
- `autoregressive` - GPT-style autoregressive transformer
- `gnn_decoder` - GNN encoder + autoregressive decoder hybrid

## Key Features

### Autoregressive Model
- Fast inference (single forward pass per token)
- Multiple decoding strategies: greedy, beam search, nucleus, top-k, contrastive
- Supports multi-path training
- Configurable decoding parameters

### GNN Decoder Model
- Leverages graph structure with GNN encoder
- Autoregressive decoder for generation
- Can work without graph structure (falls back to zero graph features)
- Supports multi-path training

## Testing

Run the test script to verify implementations:
```bash
python test_new_models.py
```

This will test:
- Model creation
- Forward pass (single and multipath)
- Model factory

## Compatibility

- All models use the same data format
- All models support the same training interface
- All models can be used with existing callbacks and logging
- The original diffusion model is unchanged and unaffected

## Configuration Parameters

### Common Parameters
- `model_type`: Model architecture to use
- `hidden_dim`: Hidden dimension size
- `num_layers` / `num_diffusion_layers`: Number of transformer layers
- `num_heads`: Number of attention heads
- `max_path_length`: Maximum path length
- `dropout`: Dropout rate
- `learning_rate`, `weight_decay`, `warmup_steps`: Training parameters

### Autoregressive-Specific
- `decoding_strategy`: "greedy", "beam_search", "nucleus", "top_k"
- `beam_width`: Beam size for beam search
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter
- `temperature`: Sampling temperature
- `length_penalty`: Length penalty for beam search
- `use_contrastive_search`: Enable contrastive search
- `contrastive_penalty_alpha`: Repetition penalty
- `contrastive_top_k`: Top-k for contrastive search

### GNN Decoder-Specific
- `gnn_type`: "gat" or "gcn"
- `gnn_layers`: Number of GNN layers
- `gnn_heads`: Number of GNN attention heads
- `decoder_layers`: Number of decoder layers
- `decoder_heads`: Number of decoder attention heads
- `use_graph_structure`: Whether to use graph structure (default: True)

## Notes

- The new models are completely independent and don't affect the existing diffusion model
- All models share the same `QuestionEncoder` for consistency
- Models can be switched by changing `model_type` in config
- Checkpoint loading is supported for all models

