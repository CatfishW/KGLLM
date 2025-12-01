# GSR (Generative Subgraph Retrieval) Integration

This module integrates the GSR approach from the EMNLP 2024 paper "Less is More: Making Smaller Language Models Competent Subgraph Retrievers for Multi-hop KGQA" into our codebase.

## Overview

GSR uses a two-stage approach:
1. **Retriever (GSR)**: Generates subgraph IDs (relation patterns) from questions
2. **Reader**: Generates answers from question + retrieved subgraph

## Components

### 1. Subgraph Index (`subgraph_index.py`)
- Builds index of relation patterns from training data
- Maps subgraph IDs to relation sequences and example triples
- Enables fast retrieval of subgraphs by pattern

### 2. Subgraph ID Generator (`subgraph_id_generator.py`)
- T5-based model for generating subgraph IDs
- Converts questions to relation pattern strings
- Supports training and inference

### 3. Reader Model (`reader_model.py`)
- Generates answers from retrieved subgraphs
- Placeholder for fine-tuned LLM (e.g., using unsloth/QLoRA)

## Usage

### Step 1: Build Subgraph Index

Extract relation patterns from your training data:

```bash
python Core/gsr/subgraph_index.py \
    --data_path ../Data/webqsp_final/train.parquet \
    --output_path ../Data/webqsp_final/subgraph_index.json \
    --min_frequency 1
```

This creates an index mapping relation patterns to subgraph structures.

### Step 2: Prepare GSR Training Data

Create training data in (question, subgraph_id) format:

```bash
python Core/gsr/subgraph_id_generator.py \
    --data_path ../Data/webqsp_final/train.parquet \
    --subgraph_index_path ../Data/webqsp_final/subgraph_index.json \
    --output_path ../Data/webqsp_final/gsr_training_data.jsonl
```

### Step 3: Train GSR Model

Train the T5 model to generate subgraph IDs:

```bash
python Core/gsr/train_gsr.py \
    --train_data ../Data/webqsp_final/gsr_training_data.jsonl \
    --val_data ../Data/webqsp_final/gsr_val_data.jsonl \
    --model_name t5-small \
    --output_dir outputs_gsr \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 1e-4
```

### Step 4: Inference

Generate subgraph IDs for questions:

```bash
python Core/gsr/inference_gsr.py \
    --model_path outputs_gsr \
    --subgraph_index_path ../Data/webqsp_final/subgraph_index.json \
    --input_path ../Data/webqsp_final/test.jsonl \
    --output_path outputs_gsr/predictions.jsonl \
    --num_beams 5 \
    --top_k 3
```

### Step 5: Prepare Reader Data (Optional)

If you want to train a reader model:

```bash
python Core/gsr/reader_model.py \
    --gsr_predictions_path outputs_gsr/predictions.jsonl \
    --subgraph_index_path ../Data/webqsp_final/subgraph_index.json \
    --original_data_path ../Data/webqsp_final/test.jsonl \
    --output_path ../Data/webqsp_final/reader_training_data.jsonl
```

## Data Formats

### Subgraph Index Format
```json
{
  "patterns": {
    "path_people_person_sibling_s": {
      "subgraph_id": "path_people_person_sibling_s",
      "relations": ["people.person.sibling_s"],
      "relation_pattern": "people.person.sibling_s",
      "example_count": 150,
      "example_triples": [
        ["m.0justin_bieber", "people.person.sibling_s", "m.0jaxon_bieber"]
      ]
    }
  }
}
```

### GSR Training Data Format
```json
{
  "question": "What is the name of justin bieber brother?",
  "subgraph_id": "path_people_person_sibling_s|people.person.name",
  "relations": ["people.person.sibling_s", "people.person.name"]
}
```

### GSR Prediction Format
```json
{
  "id": "WebQTest-0",
  "question": "What does jamaican people speak?",
  "predictions": [
    {
      "subgraph_id": "path_language_spoken",
      "relations": ["people.person.language"],
      "relation_pattern": "people.person.language",
      "example_count": 50,
      "example_triples": [...]
    }
  ]
}
```

## Comparison with Direct Path Generation

| Aspect | GSR (Subgraph ID) | Direct Path Generation |
|--------|-------------------|------------------------|
| **Output** | Relation pattern string | Full entity-relation sequence |
| **Model** | T5-small/base | Custom diffusion/flow matching |
| **Speed** | Fast (short generation) | Slower (longer sequence) |
| **Flexibility** | Limited to pre-indexed patterns | Any valid path |
| **Index Required** | Yes | No |

## Integration with Existing Codebase

You can use GSR alongside the existing path diffusion model:

1. **For fast retrieval**: Use GSR to quickly identify relevant subgraph patterns
2. **For detailed paths**: Use path diffusion to generate full reasoning paths
3. **Hybrid approach**: Use GSR to narrow down candidates, then use path diffusion for detailed paths

## Future Improvements

- [ ] Fine-tune reader model with unsloth/QLoRA
- [ ] Add evaluation metrics (hit rate, recall)
- [ ] Support for pseudo-question generation
- [ ] Integration with existing training pipeline
- [ ] Support for multiple subgraph retrieval strategies

## References

- GSR Paper: "Less is More: Making Smaller Language Models Competent Subgraph Retrievers for Multi-hop KGQA" (EMNLP 2024)
- Repository: https://github.com/hwy9855/GSR

