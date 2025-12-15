"""Compare checkpoint evaluation results."""
import json

checkpoints = [
    ('Checkpoint 5 (54 epochs)', 'entity_retrieval_checkpoint_5.json'),
    ('Checkpoint 6 (62 epochs)', 'entity_retrieval_checkpoint_6.json'),
    ('Checkpoint 7 (2 epochs)', 'entity_retrieval_checkpoint_7.json'),
]

print("=" * 100)
print("CHECKPOINT COMPARISON - Entity Retrieval with Generated Relations")
print("=" * 100)
print()
print(f"{'Checkpoint':<30} {'Rel H@1':>10} {'Rel H@5':>10} {'Ent H@1':>10} {'Ent H@5':>10} {'Partial':>10}")
print("-" * 100)

for name, path in checkpoints:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    m = data['metrics']
    rel_h1 = m['relation_hits_at_1'] * 100
    rel_h5 = m['relation_hits_at_5'] * 100
    ent_h1 = m['entity_hits_at_1'] * 100
    ent_h5 = m['entity_hits_at_5'] * 100
    partial = m['entity_partial_match'] * 100
    print(f"{name:<30} {rel_h1:>9.1f}% {rel_h5:>9.1f}% {ent_h1:>9.1f}% {ent_h5:>9.1f}% {partial:>9.1f}%")

print("=" * 100)
print()
print("Key Observations:")
print("- Checkpoint 6 (62 epochs) performs best with 57.0% Relation Hits@1 and 13.0% Entity Hits@1")
print("- Checkpoint 7 (2 epochs) is underfitted with only 24.0% Relation Hits@1")
print("- Checkpoint 5 (54 epochs) has moderate performance at 37.5% Relation Hits@1")
print()
