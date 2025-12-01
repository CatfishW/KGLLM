# Training Data Ground Truth Paths Check Report

## Summary

Found **11 validation samples** with empty `ground_truth_paths` in `Data/webqsp_final/val.parquet`.

## Affected Samples

1. **WebQTrn-9**: "how old is sacha baron cohen"
2. **WebQTrn-255**: "what year was the new york blackout"
3. **WebQTrn-258**: "what are major exports of the usa"
4. **WebQTrn-339**: "when was the musical annie written"
5. **WebQTrn-701**: "what year did the new york mets start"
6. **WebQTrn-907**: "what did nick carter sister died of"
7. **WebQTrn-1466**: "what does the letters eu stand for"
8. **WebQTrn-1992**: "who shot j lennon"
9. **WebQTrn-2037**: "what was abe lincoln shot with"
10. **WebQTrn-2059**: "what year magic johnson retired"
11. (1 more - see check_rog_data.py output)

## Data Sources Checked

- ✅ **Labeled data** (`Data/webqsp_labeled/train_paths.jsonl`): 2,743 samples, **0 with empty paths**
- ✅ **Combined data** (`Data/webqsp_combined/train_combined.parquet`): 2,743 samples, **0 with empty paths**
- ✅ **Combined validation** (`Data/webqsp_combined/val.jsonl`): 274 samples, **0 with empty paths**
- ❌ **Final validation** (`Data/webqsp_final/val.parquet`): 246 samples, **11 with empty paths**

## Root Cause

The `webqsp_final` dataset appears to be a different version/processing of the data. Some samples in the validation set don't have ground truth paths, which causes issues during training evaluation.

## Recommendations

1. **Check if paths can be derived from the graph**: For each sample with empty paths, check if there's a valid path from `q_entity` to `a_entity` in the `graph` field.

2. **Filter out samples without paths**: Remove these 11 samples from the validation set if paths cannot be generated.

3. **Handle empty paths gracefully**: Modify the training/evaluation code to skip samples with empty paths during validation.

4. **Regenerate validation data**: If the original source data has paths for these samples, regenerate the `webqsp_final/val.parquet` file.

## Next Steps

1. Check if these samples exist in the original `webqsp_rog` data with paths
2. Attempt to generate paths from the graph structure
3. Update the validation dataset to exclude or fix these samples

