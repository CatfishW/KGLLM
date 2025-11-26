# KG Path Diffusion Model

A **Graph Diffusion Model** for generating reasoning paths from knowledge graphs, given natural language questions.

## Architecture

```
Question ──► [Pretrained Transformer Encoder] ──────────────────────────┐
                                                                         │
                                                                         ▼
KG Subgraph ──► [Hybrid RGCN + Graph Transformer] ──► [Cross-Attention] ──► [Discrete Diffusion] ──► Path
                                                                         ▲
                                                                         │
Noisy Path ──► [Path Transformer with AdaLN] ───────────────────────────┘
```

### Key Components

1. **Question Encoder**: Pretrained sentence transformer (MiniLM) for encoding natural language questions
2. **Graph Encoder**: Hybrid RGCN + Graph Transformer for encoding KG subgraphs with relation-aware message passing
3. **Discrete Diffusion**: D3PM-style discrete diffusion for generating sequences of (entity, relation) pairs
4. **Path Transformer**: Transformer with Adaptive Layer Norm (AdaLN) conditioning on timestep

## Features

- ⚡ **Fast Parallel Training**: PyTorch Lightning with DDP, multi-GPU support
- 🔥 **Mixed Precision**: AMP (FP16/BF16) for faster training and lower memory
- 📊 **Efficient Data Loading**: PyTorch Geometric batching, parallel workers
- 🎯 **Configurable**: Easy command-line configuration
- 📈 **Logging**: TensorBoard and Weights & Biases support

## Installation

```bash
pip install -r requirements.txt
```

For PyTorch Geometric, follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Data Format

The model expects data in the webqsp_rog format:

```json
{
    "id": "WebQTrn-0",
    "question": "what is the name of justin bieber brother",
    "answer": ["Jaxon Bieber"],
    "q_entity": ["Justin Bieber"],
    "a_entity": ["Jaxon Bieber"],
    "graph": [
        ["Justin Bieber", "people.person.parents", "Jeremy Bieber"],
        ["Jeremy Bieber", "people.person.children", "Jaxon Bieber"],
        ...
    ],
    "paths": [
        {
            "full_path": "(Justin Bieber) --[people.person.parents]--> (Jeremy Bieber) | ...",
            "relation_chain": "people.person.parents -> people.person.children",
            "entities": ["Justin Bieber", "Jeremy Bieber", "Jaxon Bieber"],
            "relations": ["people.person.parents", "people.person.children"]
        }
    ]
}
```

## Training

### Single GPU

```bash
python train.py \
    --train_data ../Data/webqsp_rog/train-00000-of-00002.parquet \
    --val_data ../Data/webqsp_rog/validation-00000-of-00001.parquet \
    --batch_size 32 \
    --hidden_dim 256 \
    --num_diffusion_steps 1000 \
    --learning_rate 1e-4 \
    --max_epochs 100
```

### Multi-GPU (DDP)

```bash
python train.py \
    --train_data ../Data/webqsp_rog/train-00000-of-00002.parquet \
    --gpus -1 \
    --strategy ddp \
    --batch_size 32 \
    --precision 16-mixed \
    --accumulate_grad_batches 2
```

### With Weights & Biases Logging

```bash
python train.py \
    --train_data ../Data/webqsp_rog/train-00000-of-00002.parquet \
    --wandb \
    --wandb_project kg-path-diffusion \
    --experiment_name exp1
```

## Inference

```bash
python inference.py \
    --checkpoint outputs/checkpoints/best.ckpt \
    --vocab outputs/vocab.json \
    --data ../Data/webqsp_rog/test.parquet \
    --output predictions.jsonl \
    --path_length 10 \
    --temperature 1.0
```

## Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | Hidden dimension size |
| `num_graph_layers` | 3 | Number of graph encoder layers |
| `num_diffusion_layers` | 6 | Number of diffusion transformer layers |
| `num_heads` | 8 | Number of attention heads |
| `num_diffusion_steps` | 1000 | Number of diffusion timesteps |
| `max_path_length` | 20 | Maximum generated path length |
| `graph_encoder` | hybrid | Type: `rgcn`, `transformer`, `hybrid` |

## Performance Tips

1. **Use mixed precision**: `--precision 16-mixed` for ~2x speedup
2. **Increase batch size**: Use gradient accumulation if GPU memory is limited
3. **Freeze question encoder**: `--freeze_question_encoder` to reduce trainable parameters
4. **Adjust diffusion steps**: Fewer steps (100-500) for faster training, more (1000) for better quality

## Citation

If you use this code, please cite:

```bibtex
@software{kg_path_diffusion,
    title={KG Path Diffusion: Graph Diffusion Model for Knowledge Graph Reasoning},
    year={2024}
}
```

## License

MIT License

