# Model Evaluation Summary

## Evaluation Results

### Metrics (on 100 validation samples)

- **Hits@1**: 0.43 (43%)
- **Hits@5**: 0.46 (46%)
- **Exact Recall**: 0.0 (0%)
- **Partial Recall**: 0.6 (60%)
- **Average Relation Overlap**: 0.5367 (53.67%)

### Observations

1. **Model Performance**: The model achieves 43% Hits@1 and 46% Hits@5, indicating that it can generate relevant relation chains for about half of the questions.

2. **Exact Match Issue**: 0% exact recall suggests the model generates longer chains than ground truth, or the chains don't match exactly.

3. **Partial Recall**: 60% partial recall shows the model often includes correct relations, even if not in the exact sequence.

4. **Relation Overlap**: 53.67% average overlap indicates the model captures about half of the correct relations on average.

### Example Outputs

#### Example 1: Time Zone Question
- **Question**: "what time zone am i in cleveland ohio"
- **Answer**: ['Eastern Time Zone']
- **Predicted**: `location.location.time_zones -> location.location.time_zones -> ...` (long chain with repetitions)
- **Ground Truth**: 
  - `location.location.time_zones`
  - `location.place_with_neighborhoods.neighborhoods -> location.location.time_zones`
  - `location.location.events -> time.event.locations -> location.location.time_zones`
- **Status**: Contains correct relation `location.location.time_zones` but with many repetitions

#### Example 2: Nationality Question
- **Question**: "what is nina dobrev nationality"
- **Answer**: ['Bulgaria', 'Canada']
- **Predicted**: `people.person.nationality -> ... -> people.person.nationality` (contains correct relation)
- **Ground Truth**: Multiple paths including `people.person.nationality`
- **Status**: Contains correct relation but in a longer chain

### Issues Identified

1. **Chain Length**: Generated chains are very long (15-20 relations) compared to ground truth (often 1-3 relations)
2. **Repetitions**: Same relations appear multiple times in generated chains
3. **Noise**: Many irrelevant relations are included

### Recommendations

1. **Early Stopping**: Implement better stopping criteria to prevent overly long chains
2. **Deduplication**: Remove repeated relations in the same chain
3. **Length Penalty**: Add length penalty during generation
4. **Training**: Continue training to improve relation selection accuracy

## Running Evaluation

To run evaluation with metrics:

```bash
# Using batch script (Windows)
run_evaluate.bat [conda_env] [checkpoint] [vocab] [test_data]

# Direct Python
python evaluate_with_metrics.py \
    --checkpoint outputs_multipath_diffusion_relation_only_2/checkpoints/last.ckpt \
    --vocab outputs_multipath_diffusion_relation_only_2/vocab.json \
    --test_data ../Data/webqsp_final/val.parquet \
    --max_examples 100 \
    --num_samples 5 \
    --batch_size 8
```

## Training Status

- **Checkpoint Used**: `outputs_multipath_diffusion_relation_only_2/checkpoints/last.ckpt`
- **Last Modified**: 2025-11-28 21:31:28
- **Training**: Currently running or completed

