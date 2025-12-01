# GSR Model Evaluation Summary

## Quick Results

### Test Set (1,557 questions)
- **Top-1 Recall**: **52.67%** (820/1557)
- **Top-5 Recall**: **68.27%** (1063/1557)
- **Top-10 Recall**: **73.28%** (1141/1557)

### Validation Set (235 questions)
- **Top-1 Recall**: **49.36%** (116/235)
- **Top-5 Recall**: **67.23%** (158/235)
- **Top-10 Recall**: **74.89%** (176/235)

## Key Findings

✅ **Model is working!** The GSR integration successfully:
- Trained T5-small to generate subgraph IDs
- Achieved ~51% top-1 recall on both test and validation
- Shows consistent performance across splits

📊 **Performance Characteristics:**
- Top-10 recall of ~74% means most questions have correct pattern in top predictions
- Model generates 10 diverse predictions per question
- Performance is consistent between test and validation sets

## Files Generated

- **Test Predictions**: `outputs_gsr/test_predictions.jsonl`
- **Validation Predictions**: `outputs_gsr/val_predictions.jsonl`
- **Trained Model**: `outputs_gsr/` (checkpoints and final model)

## Next Steps

1. **Improve Accuracy**: Try T5-base or longer training
2. **Analyze Failures**: Examine questions where top-10 predictions don't contain correct pattern
3. **Reader Model**: Train reader to generate answers from retrieved subgraphs
4. **Hybrid Approach**: Combine GSR with direct path generation

## Usage

To run inference on new questions:
```bash
python Core/gsr/inference_gsr.py \
    --model_path outputs_gsr \
    --subgraph_index_path ../Data/webqsp_final/gsr_data/subgraph_index.json \
    --input_path <your_questions.jsonl> \
    --output_path <predictions.jsonl> \
    --num_beams 10 \
    --top_k 10
```

