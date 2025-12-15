"""
Check shortest paths data format and save 25 random examples to JSON.
Excludes graph data to keep file size small.
Also investigates why some questions have no paths.
"""
import pandas as pd
import json
import random

# Load the test data
df = pd.read_parquet('Data/webqsp_final/shortest_paths/test.parquet')

# Parse JSON columns
def parse_json(val):
    if isinstance(val, str):
        return json.loads(val)
    return val

# Select 25 random examples
print("=" * 60)
print("SAVING 25 RANDOM EXAMPLES (without graph data)")
print("=" * 60)

random.seed(42)
indices = random.sample(range(len(df)), 25)

examples = []
for idx in indices:
    row = df.iloc[idx]
    example = {
        'id': row['id'],
        'question': row['question'],
        'answer': parse_json(row['answer']),
        'q_entity': parse_json(row['q_entity']),
        'a_entity': parse_json(row['a_entity']),
        # 'graph': parse_json(row['graph']),  # Excluded - too large
        'graph_size': len(parse_json(row['graph'])),  # Just the count
        'paths': parse_json(row['paths']),
        'shortest_gt_paths': parse_json(row['shortest_gt_paths'])
    }
    examples.append(example)

# Save to JSON
output_path = 'Data/webqsp_final/shortest_paths/examples_25.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)

print(f"Saved {len(examples)} examples to {output_path}")

# Investigate questions without paths
print("\n" + "=" * 60)
print("INVESTIGATING QUESTIONS WITHOUT PATHS")
print("=" * 60)

no_path_samples = []
for idx, row in df.iterrows():
    shortest = parse_json(row['shortest_gt_paths'])
    if not shortest or len(shortest) == 0:
        no_path_samples.append({
            'id': row['id'],
            'question': row['question'],
            'answer': parse_json(row['answer']),
            'q_entity': parse_json(row['q_entity']),
            'a_entity': parse_json(row['a_entity']),
            'graph_size': len(parse_json(row['graph'])),
            'all_paths_count': len(parse_json(row['paths']))
        })

print(f"\nTotal samples without shortest_gt_paths: {len(no_path_samples)} / {len(df)}")

print("\nAnalyzing reasons:")

# Categories
no_answer_entity = 0
no_q_entity = 0
no_graph = 0
no_paths_at_all = 0
has_paths_but_no_match = 0

for s in no_path_samples:
    if not s['a_entity']:
        no_answer_entity += 1
    elif not s['q_entity']:
        no_q_entity += 1
    elif s['graph_size'] == 0:
        no_graph += 1
    elif s['all_paths_count'] == 0:
        no_paths_at_all += 1
    else:
        has_paths_but_no_match += 1

print(f"  - No answer entities (a_entity empty): {no_answer_entity}")
print(f"  - No question entities (q_entity empty): {no_q_entity}")
print(f"  - No graph data: {no_graph}")
print(f"  - No paths found at all: {no_paths_at_all}")
print(f"  - Has paths but no match to answers: {has_paths_but_no_match}")

# Show examples
print("\n" + "=" * 60)
print("EXAMPLES OF QUESTIONS WITHOUT SHORTEST PATHS")
print("=" * 60)

for i, s in enumerate(no_path_samples[:5]):
    print(f"\n{i+1}. {s['question']}")
    print(f"   ID: {s['id']}")
    print(f"   Answer: {s['answer']}")
    print(f"   Q_entity: {s['q_entity']}")
    print(f"   A_entity: {s['a_entity']}")
    print(f"   Graph size: {s['graph_size']} triples")
    print(f"   All paths count: {s['all_paths_count']}")
