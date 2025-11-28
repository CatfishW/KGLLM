# KG Path Diffusion Model

A **Graph Diffusion / Flow Matching** model for generating reasoning paths from knowledge graphs, given natural language questions.

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
5. **Flow Matching Transformer (optional)**: Rectified-flow style generator that predicts velocity fields in the embedding space, inspired by [Flow Matching (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747), enabling fewer sampling steps than diffusion.

## Features

- ⚡ **Fast Parallel Training**: PyTorch Lightning with DDP, multi-GPU support
- 🔥 **Mixed Precision**: AMP (FP16/BF16) for faster training and lower memory
- 📊 **Efficient Data Loading**: PyTorch Geometric batching, parallel workers
- 🎯 **Configurable**: Easy command-line configuration
- 📈 **Logging**: TensorBoard and Weights & Biases support
- 🔀 **Path Augmentation**: Random path sampling during training for better coverage

## Quick Start

### Windows

```batch
# Training (uses configs\flow_matching_base.yaml by default)
run_train.bat Wu

# Inference
run_inference.bat Wu outputs_1

# Quick test (10 samples)
run_quick_test.bat Wu
```

### Linux/Mac

```bash
# Make scripts executable
chmod +x run_train.sh run_inference.sh run_quick_test.sh

# Training (uses configs/flow_matching_base.yaml by default)
./run_train.sh Wu

# Inference
./run_inference.sh Wu outputs_1

# Quick test (10 samples)
./run_quick_test.sh Wu
```

## Installation

```bash
pip install -r requirements.txt
```

For PyTorch Geometric, follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Data Format

The model expects data in JSONL format with the following structure:

```json
{
    "id": "WebQTrn-0",
    "question": "what is the name of justin bieber brother",
    "answer": ["Jaxon Bieber"],
    "q_entity": ["Justin Bieber"],
    "a_entity": ["Jaxon Bieber"],
    "graph": [
        ["Justin Bieber", "people.person.parents", "Jeremy Bieber"],
        ["Jeremy Bieber", "people.person.children", "Jaxon Bieber"]
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

### Using Scripts (Recommended)

**Windows:**
```batch
run_train.bat Wu [optional_config]
```

**Linux/Mac:**
```bash
./run_train.sh Wu [optional_config]
```

### Manual Command

```bash
python train.py \
    --train_data ../Data/webqsp_combined/train_combined.parquet \
    --val_data ../Data/webqsp_combined/val.jsonl \
    --batch_size 4 \
    --hidden_dim 128 \
    --num_graph_layers 2 \
    --num_diffusion_layers 2 \
    --num_diffusion_steps 100 \
    --max_path_length 20 \
    --gpus 1 \
    --output_dir outputs \
    --max_epochs 50
```

### Multi-GPU (DDP)

```bash
python train.py \
    --train_data ../Data/webqsp_combined/train_combined.parquet \
    --gpus -1 \
    --strategy ddp \
    --batch_size 32 \
    --precision 16-mixed \
    --accumulate_grad_batches 2
```

### With Weights & Biases Logging

```bash
python train.py \
    --train_data ../Data/webqsp_combined/train_combined.parquet \
    --wandb \
    --wandb_project kg-path-diffusion \
    --experiment_name exp1
```

### Configuration Files

- Pass `--config path/to/config.yaml` to `train.py` (or rely on `run_train.*`, which defaults to `configs/flow_matching_base.yaml`).
- Config files are plain YAML/JSON dictionaries whose keys match the CLI arguments. Command-line flags provided alongside `--config` continue to override the file.
- Example (`configs/flow_matching_base.yaml`):

```yaml
train_data: ../Data/webqsp_final/train.parquet
val_data: ../Data/webqsp_final/val.parquet
graph_encoder: hybrid
hidden_dim: 128
dropout: 0.1
num_graph_layers: 1
num_diffusion_layers: 1
num_heads: 8
num_diffusion_steps: 12
path_generator_type: flow_matching
flow_integration_steps: 32
flow_ce_weight: 0.1
question_encoder: sentence-transformers/all-MiniLM-L6-v2
tokenizer_name: sentence-transformers/all-MiniLM-L6-v2
freeze_question_encoder: true
max_question_length: 64
max_path_length: 12

max_graph_nodes: 180
max_vocab_size: 500000
max_entities: 500000
max_relations: 50000
batch_size: 2
num_workers: 4
learning_rate: 1e-4
weight_decay: 0.01
warmup_steps: 1000
max_steps: 100000
max_epochs: 50
gradient_clip: 1.0
accumulate_grad_batches: 1
gpus: 1
precision: 16-mixed
strategy: auto
output_dir: outputs_multipath_flowmatching
experiment_name: kg_path_diffusion_flow
```

Duplicate this file (or create a new one) to keep experiment settings under version control.

## Inference

### Using Scripts (Recommended)

**Windows:**
```batch
run_inference.bat Wu outputs_1
```

**Linux/Mac:**
```bash
./run_inference.sh Wu outputs_1
```

### Manual Command

```bash
python inference.py \
    --checkpoint outputs_1/checkpoints/last.ckpt \
    --vocab outputs_1/vocab.json \
    --data ../Data/webqsp_combined/val.jsonl \
    --output inference_results.jsonl \
    --batch_size 8 \
    --path_length 10 \
    --temperature 0.7 \
    --show_examples 10
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | required | Path to model checkpoint |
| `--vocab` | required | Path to vocabulary file |
| `--data` | required | Path to input data (jsonl/parquet) |
| `--output` | `inference_results.jsonl` | Output file path |
| `--path_length` | 10 | Maximum generated path length |
| `--temperature` | 1.0 | Sampling temperature (0=greedy, >1=more random) |
| `--num_samples` | 1 | Number of paths to generate per question |
| `--max_samples` | None | Limit number of samples to process |

## Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | Hidden dimension size |
| `num_graph_layers` | 3 | Number of graph encoder layers |
| `num_diffusion_layers` | 6 | Number of transformer blocks inside the path generator |
| `num_heads` | 8 | Number of attention heads |
| `num_diffusion_steps` | 1000 | Diffusion timesteps (when `path_generator_type=diffusion`) |
| `path_generator_type` | `diffusion` | Switch between `diffusion` and `flow_matching` |
| `flow_integration_steps` | 32 | Euler integration steps for flow-matching sampling |
| `flow_ce_weight` | 0.1 | Auxiliary CE weight that stabilizes discrete decoding |
| `max_path_length` | 20 | Maximum generated path length |

| `question_encoder` | MiniLM | Pretrained question encoder name |
| `tokenizer_name` | question encoder | Tokenizer used for questions (defaults to encoder) |
| `max_question_length` | 64 | Max tokens per question |
| `max_graph_nodes` | 200 | Max nodes retained from each local subgraph |
| `max_entities` | `max_vocab_size` (500k) | Entity vocab cap when building from scratch |
| `max_relations` | 50000 | Relation vocab cap |
| `graph_encoder` | hybrid | Type: `rgcn`, `transformer`, `hybrid` |

### Selecting the Path Generator

Both generators reuse the same question/graph encoders. Use:

```bash
python train.py ... --path_generator_type flow_matching --flow_integration_steps 32
```

Flow Matching uses a rectified-flow velocity objective [Lipman et al., 2022](https://arxiv.org/abs/2210.02747), which keeps sampling fast (no reverse diffusion loop) while maintaining state-of-the-art quality on long reasoning paths. Temperature controls continue to work for both generators.

## Performance Tips

1. **Use mixed precision**: `--precision 16-mixed` for ~2x speedup
2. **Increase batch size**: Use gradient accumulation if GPU memory is limited
3. **Freeze question encoder**: `--freeze_question_encoder` to reduce trainable parameters
4. **Adjust diffusion steps**: Fewer steps (50-100) for faster training, more (500-1000) for better quality
5. **Lower temperature**: Use `--temperature 0.5-0.8` for more focused path generation

## Output Format

Inference results are saved in JSONL format:

```json
{
    "id": "WebQTrn-2415",
    "question": "what sports are played in canada",
    "answer": ["..."],
    "generated_path": "(Canada) --[sports.sports_team.location]--> (Canada Davis Cup team)",
    "generated_entities": ["Canada", "Canada Davis Cup team"],
    "generated_relations": ["sports.sports_team.location"],
    "generated_relation_chain": "sports.sports_team.location",
    "ground_truth_paths": [...]
}
```

## Project Structure

```
Core/
├── train.py              # Training script
├── inference.py          # Inference script
├── kg_path_diffusion.py  # Main model
├── run_train.bat/.sh     # Easy training scripts
├── run_inference.bat/.sh # Easy inference scripts
├── run_quick_test.bat/.sh # Quick test scripts
├── data/
│   └── dataset.py        # Dataset and data loading
├── modules/
│   ├── diffusion.py      # Discrete diffusion module
│   ├── flow_matching.py  # Flow-matching velocity transformer
│   └── graph_encoder.py  # Graph encoding modules
├── configs/
│   └── flow_matching_base.yaml  # Example training configuration
└── outputs_1/            # Training outputs
    ├── checkpoints/      # Model checkpoints
    └── vocab.json        # Vocabulary file
```

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
