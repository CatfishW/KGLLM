## KG Path Diffusion Model

**Goal**: Generate reasoning paths over a knowledge graph for a natural-language question using diffusion.

### Install

```bash
pip install -r requirements.txt
```

(Follow the PyTorch Geometric install instructions for your CUDA / PyTorch version.)

### Train

- **Windows**: `run_train.bat Wu`
- **Linux/macOS**: `./run_train.sh Wu`

These call `train.py` with a default config in `configs/` (see the config files for all options).

### Inference

- **Windows**: `run_inference.bat Wu outputs_1`
- **Linux/macOS**: `./run_inference.sh Wu outputs_1`

This expects a trained checkpoint and vocab in an `outputs_*` directory.

### Data Format

Data lives under `../Data/` and is typically parquet or JSONL. Expected format:

```json
{
    "id": "WebQTrn-0",
    "question": "what sports are played in canada",
    "answer": ["Ice Hockey", "Lacrosse", "Curling", "Canadian Football"],
    "q_entity": ["Canada"],
    "a_entity": ["Ice Hockey", "Lacrosse", "Curling", "Canadian Football"],
    "graph": [
        ["Canada", "sports.sports_team.location", "Toronto Maple Leafs"],
        ["Toronto Maple Leafs", "sports.sports_team.sport", "Ice Hockey"],
        ["Canada", "sports.sports_team.location", "Toronto Rock"],
        ["Toronto Rock", "sports.sports_team.sport", "Lacrosse"],
        ["Canada", "sports.sports_team.location", "Team Canada"],
        ["Team Canada", "sports.sports_team.sport", "Curling"]
    ],
    "paths": [
        {
            "full_path": "(Canada) --[sports.sports_team.location]--> (Toronto Maple Leafs) --[sports.sports_team.sport]--> (Ice Hockey)",
            "relation_chain": "sports.sports_team.location -> sports.sports_team.sport",
            "entities": ["Canada", "Toronto Maple Leafs", "Ice Hockey"],
            "relations": ["sports.sports_team.location", "sports.sports_team.sport"]
        },
        {
            "full_path": "(Canada) --[sports.sports_team.location]--> (Toronto Rock) --[sports.sports_team.sport]--> (Lacrosse)",
            "relation_chain": "sports.sports_team.location -> sports.sports_team.sport",
            "entities": ["Canada", "Toronto Rock", "Lacrosse"],
            "relations": ["sports.sports_team.location", "sports.sports_team.sport"]
        },
        {
            "full_path": "(Canada) --[sports.sports_team.location]--> (Team Canada) --[sports.sports_team.sport]--> (Curling)",
            "relation_chain": "sports.sports_team.location -> sports.sports_team.sport",
            "entities": ["Canada", "Team Canada", "Curling"],
            "relations": ["sports.sports_team.location", "sports.sports_team.sport"]
        }
    ]
}
```

See the config files for exact data paths.

### Notes

- Core code lives in `Core/` (models, modules, training, utils).
- Outputs (checkpoints, vocab, logs) are written to `outputs_*` directories configured in the YAML files.
