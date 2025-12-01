# GSR Training and Evaluation - Complete Summary

## ✅ Training Complete!

The GSR model has been successfully trained and evaluated on WebQSP dataset.

---

## Training Summary

### Model Configuration
- **Base Model**: T5-small (~60M parameters)
- **Training Samples**: 36,470
- **Validation Samples**: 2,419
- **Epochs**: 10
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Training Time**: ~3 hours 9 minutes

### Training Statistics
- **Total Steps**: 22,800
- **Final Training Loss**: 0.1567
- **Final Validation Loss**: 0.0583
- **Model Saved**: `Core/outputs_gsr/`

---

## Evaluation Results

### Test Set Performance (1,557 questions with ground truth)

| Metric | Recall | Hits/Total |
|--------|--------|------------|
| **Top-1 (Hit Rate)** | **52.67%** | 820/1557 |
| **Top-3** | **64.42%** | 1003/1557 |
| **Top-5** | **68.27%** | 1063/1557 |
| **Top-10** | **73.28%** | 1141/1557 |

**Pattern Coverage:**
- Unique patterns in ground truth: 3,475
- Patterns correctly predicted: 224 (6.45%)

### Validation Set Performance (235 questions with ground truth)

| Metric | Recall | Hits/Total |
|--------|--------|------------|
| **Top-1 (Hit Rate)** | **49.36%** | 116/235 |
| **Top-3** | **63.40%** | 149/235 |
| **Top-5** | **67.23%** | 158/235 |
| **Top-10** | **74.89%** | 176/235 |

**Pattern Coverage:**
- Unique patterns in ground truth: 834
- Patterns correctly predicted: 94 (11.27%)

---

## Key Achievements

✅ **Model Training**: Successfully trained T5-small to generate subgraph IDs  
✅ **Evaluation Metrics**: Implemented comprehensive recall metrics  
✅ **Consistent Performance**: ~51% top-1 recall on both test and validation  
✅ **High Coverage**: 73-75% top-10 recall means most questions have correct pattern in top predictions  

---

## Generated Files

### Model Files
- `Core/outputs_gsr/model.safetensors` - Trained model weights
- `Core/outputs_gsr/checkpoint-*/` - Training checkpoints
- `Core/outputs_gsr/tokenizer_config.json` - Tokenizer configuration

### Predictions
- `Core/outputs_gsr/test_predictions.jsonl` - Test set predictions
- `Core/outputs_gsr/val_predictions.jsonl` - Validation set predictions

### Data Files
- `Data/webqsp_final/gsr_data/subgraph_index.json` - Subgraph index (5,059 patterns)
- `Data/webqsp_final/gsr_data/gsr_training_data.jsonl` - Training data
- `Data/webqsp_final/gsr_data/gsr_val_data.jsonl` - Validation data

---

## Performance Analysis

### Strengths
1. **Consistent Results**: Similar performance on test (52.67%) and validation (49.36%) shows good generalization
2. **High Top-10 Recall**: ~74% means model finds correct pattern in top predictions for most questions
3. **Efficient**: T5-small is fast and memory-efficient compared to larger models

### Areas for Improvement
1. **Top-1 Accuracy**: Could improve from ~51% with:
   - Larger model (T5-base: 220M params)
   - Longer training
   - Better data augmentation
2. **Pattern Diversity**: Only 6-11% of unique patterns are being hit
3. **Pattern Recall**: Low average pattern recall suggests many patterns are rarely predicted

---

## Comparison with Direct Path Generation

| Approach | Top-1 Accuracy | Top-5 Accuracy | Model Size | Speed |
|----------|----------------|----------------|------------|-------|
| **GSR (T5-small)** | **51.02%** | **67.75%** | ~60M | Fast |
| **Direct Path (Diffusion)** | TBD | TBD | Larger | Slower |

**GSR Advantages:**
- Faster inference (text-to-text generation)
- Smaller model size
- Pre-computed index for fast retrieval
- Modular design (retriever + reader)

---

## Next Steps

### Immediate
1. ✅ Training complete
2. ✅ Evaluation complete
3. ✅ Metrics calculated

### Future Improvements
1. **Model Scaling**: Try T5-base for better accuracy
2. **Reader Model**: Train reader to generate answers from retrieved subgraphs
3. **Hybrid Approach**: Combine GSR with direct path generation
4. **Failure Analysis**: Analyze questions where top-10 predictions don't contain correct pattern
5. **Pattern Balancing**: Address class imbalance in training data

---

## Usage

### Run Inference
```bash
cd Core
conda activate Wu
python gsr/inference_gsr.py \
    --model_path outputs_gsr \
    --subgraph_index_path ../Data/webqsp_final/gsr_data/subgraph_index.json \
    --input_path ../Data/webqsp_final/test.jsonl \
    --output_path outputs_gsr/predictions.jsonl \
    --num_beams 10 \
    --top_k 10 \
    --ground_truth_path ../Data/webqsp_final/test.parquet
```

### View Results
- Test predictions: `Core/outputs_gsr/test_predictions.jsonl`
- Validation predictions: `Core/outputs_gsr/val_predictions.jsonl`
- Evaluation report: `Core/gsr/evaluation_report.md`

---

## Conclusion

The GSR integration is **fully functional and evaluated**! 

- ✅ Model trained successfully
- ✅ Evaluation metrics implemented
- ✅ Test and validation sets evaluated
- ✅ Recall metrics calculated: **~51% top-1, ~68% top-5, ~74% top-10**

The model is ready for use and can be further improved with the suggestions above.

