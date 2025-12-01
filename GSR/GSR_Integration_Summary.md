# GSR Integration Summary

## What Was Added

The GSR (Generative Subgraph Retrieval) approach has been fully integrated into the repository. This adds a two-stage retrieval approach alongside the existing direct path generation method.

## New Files Created

### Core Components

1. **`Core/gsr/subgraph_index.py`**
   - Subgraph index builder and lookup system
   - Extracts relation patterns from training data
   - Maps subgraph IDs to relation sequences and example triples
   - Supports pattern search and retrieval

2. **`Core/gsr/subgraph_id_generator.py`**
   - T5-based subgraph ID generator
   - Converts questions to relation pattern strings
   - Supports training and inference
   - Includes data preparation utilities

3. **`Core/gsr/reader_model.py`**
   - Reader model component (placeholder for fine-tuned LLM)
   - Generates answers from retrieved subgraphs
   - Includes reader data preparation utilities

4. **`Core/gsr/__init__.py`**
   - Module initialization

### Scripts

5. **`Core/gsr/train_gsr.py`**
   - Training script for GSR model
   - Uses HuggingFace Transformers Trainer
   - Supports validation and checkpointing

6. **`Core/gsr/inference_gsr.py`**
   - Inference script for subgraph ID generation
   - Retrieves subgraphs from index
   - Supports evaluation metrics

7. **`Core/gsr/prepare_gsr_data.py`**
   - Complete data preparation pipeline
   - Automates all GSR data preparation steps

### Documentation

8. **`Core/gsr/README.md`**
   - Comprehensive usage guide
   - Examples and data format specifications
   - Comparison with direct path generation

## How It Works

### Architecture

```
Question → [GSR Model (T5)] → Subgraph ID → [Index Lookup] → Subgraph → [Reader Model] → Answer
```

### Key Differences from Direct Path Generation

| Feature | GSR | Direct Path |
|---------|-----|-------------|
| Output | Relation pattern string | Full entity-relation sequence |
| Model | T5-small/base | Custom diffusion/flow matching |
| Index | Required (pre-computed) | Not required |
| Speed | Fast (short generation) | Slower (longer sequence) |
| Flexibility | Pre-indexed patterns only | Any valid path |

## Usage Workflow

### 1. Build Subgraph Index
```bash
python Core/gsr/subgraph_index.py \
    --data_path ../Data/webqsp_final/train.parquet \
    --output_path ../Data/webqsp_final/subgraph_index.json
```

### 2. Prepare Training Data
```bash
python Core/gsr/prepare_gsr_data.py \
    --train_data ../Data/webqsp_final/train.parquet \
    --output_dir ../Data/webqsp_final/gsr_data
```

### 3. Train GSR Model
```bash
python Core/gsr/train_gsr.py \
    --train_data ../Data/webqsp_final/gsr_data/gsr_training_data.jsonl \
    --val_data ../Data/webqsp_final/gsr_data/gsr_val_data.jsonl \
    --model_name t5-small \
    --output_dir outputs_gsr
```

### 4. Inference
```bash
python Core/gsr/inference_gsr.py \
    --model_path outputs_gsr \
    --subgraph_index_path ../Data/webqsp_final/gsr_data/subgraph_index.json \
    --input_path ../Data/webqsp_final/test.jsonl \
    --output_path outputs_gsr/predictions.jsonl
```

## Integration Points

### With Existing Codebase

1. **Shared Data Format**: Uses same dataset format (parquet/jsonl)
2. **Compatible Vocab**: Can use existing entity/relation vocab
3. **Parallel Approach**: Can run alongside path diffusion model
4. **Hybrid Strategy**: Can combine both approaches

### Potential Hybrid Usage

```python
# Use GSR for fast candidate retrieval
subgraph_ids = gsr_model.generate(question)

# Use path diffusion for detailed paths
detailed_paths = path_diffusion_model.generate(question, graph)

# Combine results
final_answer = combine_results(subgraph_ids, detailed_paths)
```

## Data Formats

### Subgraph Index
```json
{
  "patterns": {
    "path_people_person_sibling_s": {
      "subgraph_id": "path_people_person_sibling_s",
      "relations": ["people.person.sibling_s"],
      "example_count": 150,
      "example_triples": [...]
    }
  }
}
```

### GSR Training Data
```json
{
  "question": "What is the name of justin bieber brother?",
  "subgraph_id": "path_people_person_sibling_s|people.person.name",
  "relations": ["people.person.sibling_s", "people.person.name"]
}
```

## Benefits

1. **Efficiency**: Faster inference with smaller models
2. **Scalability**: Pre-computed index enables fast retrieval
3. **Modularity**: Separates retrieval from answer generation
4. **Flexibility**: Can be used standalone or combined with other approaches

## Future Enhancements

- [ ] Fine-tune reader model with unsloth/QLoRA
- [ ] Add comprehensive evaluation metrics
- [ ] Support pseudo-question generation
- [ ] Integration with existing training pipeline
- [ ] Hybrid retrieval strategies

## References

- GSR Paper: "Less is More: Making Smaller Language Models Competent Subgraph Retrievers for Multi-hop KGQA" (EMNLP 2024)
- Original Repository: https://github.com/hwy9855/GSR
- Comparison Document: `GSR_Comparison_Summary.md`
- Detailed Explanation: `GSR_Subgraph_ID_Details.md`

