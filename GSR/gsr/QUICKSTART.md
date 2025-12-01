# GSR Quick Start Guide

Get started with GSR subgraph retrieval in 5 minutes!

## Prerequisites

```bash
pip install transformers torch pandas pyarrow
```

## Step-by-Step Guide

### Step 1: Build Subgraph Index (2 minutes)

```bash
python Core/gsr/subgraph_index.py \
    --data_path ../Data/webqsp_final/train.parquet \
    --output_path ../Data/webqsp_final/subgraph_index.json \
    --min_frequency 1
```

**What this does**: Extracts relation patterns from your training data and creates an index.

**Output**: `subgraph_index.json` with all relation patterns

### Step 2: Prepare Training Data (1 minute)

```bash
python Core/gsr/prepare_gsr_data.py \
    --train_data ../Data/webqsp_final/train.parquet \
    --output_dir ../Data/webqsp_final/gsr_data
```

**What this does**: Creates (question, subgraph_id) pairs for training.

**Output**: 
- `gsr_data/gsr_training_data.jsonl` - Training data
- `gsr_data/subgraph_index.json` - Subgraph index

### Step 3: Train GSR Model (10-30 minutes depending on data size)

```bash
python Core/gsr/train_gsr.py \
    --train_data ../Data/webqsp_final/gsr_data/gsr_training_data.jsonl \
    --val_data ../Data/webqsp_final/gsr_data/gsr_val_data.jsonl \
    --model_name t5-small \
    --output_dir outputs_gsr \
    --batch_size 16 \
    --num_epochs 5 \
    --learning_rate 1e-4
```

**What this does**: Trains T5 to generate subgraph IDs from questions.

**Output**: Trained model in `outputs_gsr/`

### Step 4: Generate Subgraph IDs (1 minute)

```bash
python Core/gsr/inference_gsr.py \
    --model_path outputs_gsr \
    --subgraph_index_path ../Data/webqsp_final/gsr_data/subgraph_index.json \
    --input_path ../Data/webqsp_final/test.jsonl \
    --output_path outputs_gsr/predictions.jsonl \
    --num_beams 5 \
    --top_k 3
```

**What this does**: Generates subgraph IDs for test questions and retrieves subgraphs.

**Output**: `predictions.jsonl` with predicted subgraph IDs and retrieved patterns

## Example Output

### Subgraph Index Entry
```json
{
  "path_people_person_sibling_s": {
    "relations": ["people.person.sibling_s"],
    "example_count": 150,
    "example_triples": [
      ["m.0justin_bieber", "people.person.sibling_s", "m.0jaxon_bieber"]
    ]
  }
}
```

### GSR Prediction
```json
{
  "question": "What is the name of justin bieber brother?",
  "predictions": [
    {
      "subgraph_id": "path_people_person_sibling_s|people.person.name",
      "relations": ["people.person.sibling_s", "people.person.name"],
      "example_count": 50
    }
  ]
}
```

## All-in-One Script

For convenience, use the complete pipeline:

```bash
python Core/gsr/prepare_gsr_data.py \
    --train_data ../Data/webqsp_final/train.parquet \
    --output_dir ../Data/webqsp_final/gsr_data
```

This runs steps 1-2 automatically!

## Troubleshooting

### Issue: "No module named 'gsr'"
**Solution**: Make sure you're running from the `Core/` directory or add it to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/Core"
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size:
```bash
python Core/gsr/train_gsr.py ... --batch_size 8
```

### Issue: "Pattern not found in index"
**Solution**: This is normal for new patterns. The model can generate patterns not in the training index.

## Next Steps

1. **Evaluate**: Add evaluation metrics to `inference_gsr.py`
2. **Reader Model**: Train a reader model to generate answers from retrieved subgraphs
3. **Hybrid Approach**: Combine GSR with direct path generation for best results

## See Also

- Full documentation: `Core/gsr/README.md`
- Comparison with direct path: `GSR_Comparison_Summary.md`
- Detailed explanation: `GSR_Subgraph_ID_Details.md`

