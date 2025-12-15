"""
Check shortest paths data format and save 25 random examples to JSON.
"""
import pandas as pd
import json
import random

# Load the test data
df = pd.read_parquet('Data/webqsp_final/shortest_paths/test.parquet')

print("=" * 60)
print("SHORTEST PATHS DATA FORMAT")
print("=" * 60)
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")

# Parse JSON columns
def parse_json(val):
    if isinstance(val, str):
        return json.loads(val)
    return val

# Check data types
print("\nData types:")
for col in df.columns:
    sample = df[col].iloc[0]
    parsed = parse_json(sample) if col in ['answer', 'q_entity', 'a_entity', 'graph', 'paths', 'shortest_gt_paths'] else sample
    print(f"  {col}: {type(parsed).__name__}")

# Sample one row in detail
print("\n" + "=" * 60)
print("SAMPLE ROW STRUCTURE")
print("=" * 60)
row = df.iloc[0]
print(f"id: {row['id']}")
print(f"question: {row['question']}")
print(f"answer: {parse_json(row['answer'])}")
print(f"q_entity: {parse_json(row['q_entity'])}")
print(f"a_entity: {parse_json(row['a_entity'])}")

graph = parse_json(row['graph'])
print(f"graph: {len(graph)} triples")
if graph:
    print(f"  First triple: {graph[0]}")

paths = parse_json(row['paths'])
print(f"paths: {len(paths)} paths")

shortest = parse_json(row['shortest_gt_paths'])
print(f"shortest_gt_paths: {len(shortest)} paths")
if shortest:
    print("  Shortest paths:")
    for p in shortest:
        print(f"    - relation_chain: {p.get('relation_chain')}")
        print(f"      entities: {p.get('entities')}")

# Select 25 random examples
print("\n" + "=" * 60)
print("SAVING 25 RANDOM EXAMPLES")
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
        'graph': parse_json(row['graph']),
        'paths': parse_json(row['paths']),
        'shortest_gt_paths': parse_json(row['shortest_gt_paths'])
    }
    examples.append(example)

# Save to JSON
output_path = 'Data/webqsp_final/shortest_paths/examples_25.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)

print(f"Saved {len(examples)} examples to {output_path}")

# Print summary of examples
print("\nExamples summary:")
for i, ex in enumerate(examples[:5]):
    print(f"  {i+1}. {ex['question'][:50]}...")
    print(f"     Shortest paths: {len(ex['shortest_gt_paths'])}")
