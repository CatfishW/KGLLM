# GSR Model Evaluation Report

## Evaluation Results Summary

### Test Set Results (WebQSP Test)

**Dataset Statistics:**
- **Total Questions**: 1,628
- **Questions with Ground Truth**: 1,557
- **Unique Patterns in Ground Truth**: 3,475

**Recall Metrics:**

| Metric | Value | Hits/Total |
|--------|-------|------------|
| **Top-1 Recall (Hit Rate)** | **52.67%** | 820/1557 |
| **Top-3 Recall** | **64.42%** | 1003/1557 |
| **Top-5 Recall** | **68.27%** | 1063/1557 |
| **Top-10 Recall** | **73.28%** | 1141/1557 |

**Pattern-Level Metrics:**
- **Average Pattern Recall**: 4.15%
- **Unique Patterns in GT**: 3,475
- **Patterns with Hits**: 224 (6.45% of unique patterns)
- **Average Predictions per Question**: 10.00

---

### Validation Set Results (WebQSP Val)

**Dataset Statistics:**
- **Total Questions**: 246
- **Questions with Ground Truth**: 235
- **Unique Patterns in Ground Truth**: 834

**Recall Metrics:**

| Metric | Value | Hits/Total |
|--------|-------|------------|
| **Top-1 Recall (Hit Rate)** | **49.36%** | 116/235 |
| **Top-3 Recall** | **63.40%** | 149/235 |
| **Top-5 Recall** | **67.23%** | 158/235 |
| **Top-10 Recall** | **74.89%** | 176/235 |

**Pattern-Level Metrics:**
- **Average Pattern Recall**: 9.45%
- **Unique Patterns in GT**: 834
- **Patterns with Hits**: 94 (11.27% of unique patterns)
- **Average Predictions per Question**: 10.00

---

### Overall Performance Summary

| Metric | Test Set | Validation Set | Average |
|--------|----------|----------------|---------|
| **Top-1 Recall** | 52.67% | 49.36% | **51.02%** |
| **Top-3 Recall** | 64.42% | 63.40% | **63.91%** |
| **Top-5 Recall** | 68.27% | 67.23% | **67.75%** |
| **Top-10 Recall** | 73.28% | 74.89% | **74.09%** |

## Analysis

### Strengths
1. **Consistent Performance**: Similar results on test (52.67%) and validation (49.36%) sets show good generalization
2. **Strong Top-10 Recall**: ~74% of questions have correct pattern in top-10 predictions
3. **Reasonable Top-1 Performance**: ~51% hit rate demonstrates model can identify correct patterns
4. **High Coverage**: Model generates diverse predictions (10 per question)

### Areas for Improvement
1. **Pattern Diversity**: Only 224/3,475 (test) and 94/834 (val) unique patterns are being hit, suggesting bias toward frequent patterns
2. **Top-1 Accuracy**: ~51% could be improved with larger model (T5-base) or better training
3. **Pattern Recall**: Low average pattern recall indicates many patterns are rarely predicted correctly
4. **Validation Performance**: Slightly lower on validation (49.36% vs 52.67%) suggests some overfitting

## Comparison with Direct Path Generation

| Approach | Top-1 Accuracy | Top-5 Accuracy | Model Size |
|----------|----------------|----------------|------------|
| **GSR (T5-small)** | 52.67% | 68.27% | ~60M params |
| **Direct Path (Diffusion)** | TBD | TBD | Larger |

## Recommendations

1. **Increase Model Size**: Try T5-base (220M params) for better accuracy
2. **Training Data Augmentation**: Add more diverse patterns to training
3. **Ensemble Methods**: Combine multiple predictions
4. **Fine-tuning**: Continue training on validation set
5. **Pattern Balancing**: Address class imbalance in pattern frequencies

## Next Steps

1. Evaluate on validation set for consistency
2. Analyze failure cases to identify common errors
3. Compare with direct path generation results
4. Experiment with different beam sizes and generation strategies

