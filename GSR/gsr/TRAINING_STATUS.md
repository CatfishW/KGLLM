# GSR Training Status

## Data Preparation: ✅ COMPLETE

### Subgraph Index
- **File**: `Data/webqsp_final/gsr_data/subgraph_index.json`
- **Patterns**: 5,059 unique relation patterns
- **Examples**: 36,470 total examples

### Training Data
- **File**: `Data/webqsp_final/gsr_data/gsr_training_data.jsonl`
- **Samples**: 36,470 training samples
- **Format**: (question, subgraph_id, relations)

### Validation Data
- **File**: `Data/webqsp_final/gsr_data/gsr_val_data.jsonl`
- **Samples**: 2,419 validation samples

## Top Patterns Found

1. `travel.travel_destination.tourist_attractions` - 1,393 examples
2. `film.actor.film|film.performance.film` - 902 examples
3. `people.person.nationality|film.film.country` - 683 examples
4. `language.human_language.countries_spoken_in` - 662 examples
5. `location.country.languages_spoken` - 573 examples

## Next Step: Training

To train the GSR model, run:

```bash
cd Core
python gsr/train_gsr.py \
    --train_data ../Data/webqsp_final/gsr_data/gsr_training_data.jsonl \
    --val_data ../Data/webqsp_final/gsr_data/gsr_val_data.jsonl \
    --model_name t5-small \
    --output_dir outputs_gsr \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 1e-4
```

Or use the batch script:
```bash
cd Core
gsr\run_training.bat [conda_env_name]
```

## Training Configuration

- **Model**: T5-small (60M parameters)
- **Batch Size**: 16
- **Epochs**: 10
- **Learning Rate**: 1e-4
- **Warmup Steps**: 1000
- **Max Input Length**: 512 tokens
- **Max Target Length**: 128 tokens

## Expected Training Time

- **T5-small**: ~30-60 minutes on GPU, ~2-4 hours on CPU
- **T5-base**: ~1-2 hours on GPU, ~6-8 hours on CPU

## Notes

- If you encounter torch DLL errors, this is a Windows environment issue
- The model will download T5-small from HuggingFace on first run (~240MB)
- Training checkpoints will be saved in `outputs_gsr/checkpoints/`
- Best model will be saved automatically based on validation loss

