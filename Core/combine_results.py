"""Combine all checkpoint results into a single JSON file."""
import json

checkpoints = [
    ('checkpoint_5', 'entity_retrieval_checkpoint_5.json', 'eval_results_checkpoint_5.json'),
    ('checkpoint_6', 'entity_retrieval_checkpoint_6.json', 'eval_results_checkpoint_6.json'),
    ('checkpoint_7', 'entity_retrieval_checkpoint_7.json', 'eval_results_checkpoint_7.json'),
]

combined_results = {
    'summary': {
        'description': 'Evaluation of 3 checkpoints with entity retrieval using generated relation paths',
        'test_samples': 200,
        'evaluation_date': '2024-12-02'
    },
    'checkpoints': {}
}

for name, entity_path, eval_path in checkpoints:
    with open(entity_path, 'r', encoding='utf-8') as f:
        entity_data = json.load(f)
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    combined_results['checkpoints'][name] = {
        'metrics': entity_data['metrics'],
        'examples': []
    }
    
    # Combine examples from both files
    entity_examples = entity_data.get('examples', [])
    eval_examples = eval_data.get('examples', [])
    
    for i, (ent_ex, eval_ex) in enumerate(zip(entity_examples, eval_examples)):
        combined_example = {
            'question': ent_ex.get('question', ''),
            'topic_entities': ent_ex.get('topic_entities', []),
            'predicted_relations': ent_ex.get('predicted_relations', []),
            'ground_truth_relations': eval_ex.get('ground_truth', []),
            'ground_truth_full_paths': eval_ex.get('ground_truth_full', []),
            'retrieved_entities': ent_ex.get('retrieved_entities', []),
            'ground_truth_entities': ent_ex.get('ground_truth_entities', []),
            'answer': eval_ex.get('answer', []),
            'entity_match': ent_ex.get('match', None)
        }
        combined_results['checkpoints'][name]['examples'].append(combined_example)

# Save combined results
with open('checkpoint_comparison_results.json', 'w', encoding='utf-8') as f:
    json.dump(combined_results, f, indent=2, ensure_ascii=False)

print("Combined results saved to: checkpoint_comparison_results.json")

# Print summary
print("\n" + "=" * 80)
print("CHECKPOINT EVALUATION SUMMARY")
print("=" * 80)

for name in ['checkpoint_5', 'checkpoint_6', 'checkpoint_7']:
    m = combined_results['checkpoints'][name]['metrics']
    print(f"\n{name.upper()}:")
    print(f"  Relation Hits@1: {m['relation_hits_at_1']*100:.1f}%")
    print(f"  Relation Hits@5: {m['relation_hits_at_5']*100:.1f}%")
    print(f"  Entity Hits@1:   {m['entity_hits_at_1']*100:.1f}%")
    print(f"  Entity Hits@5:   {m['entity_hits_at_5']*100:.1f}%")
    print(f"  Partial Match:   {m['entity_partial_match']*100:.1f}%")
