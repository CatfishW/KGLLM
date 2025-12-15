"""Compare ground truth and index formats."""
import pandas as pd
import json
import pickle

# Load test data ground truth
df = pd.read_parquet('Data/webqsp_final/test.parquet')

print("=" * 60)
print("GROUND TRUTH FORMAT (test.parquet)")
print("=" * 60)

for idx in range(3):
    row = df.iloc[idx]
    paths_str = row['paths']
    paths = json.loads(paths_str) if isinstance(paths_str, str) else paths_str
    
    print(f"\nSample {idx}: {row['question'][:50]}...")
    print(f"  Number of paths: {len(paths)}")
    if paths:
        for p in paths[:3]:
            print(f"    relation_chain: '{p['relation_chain']}'")

# Load index
print("\n" + "=" * 60)
print("INDEX FORMAT (EXP/index/metadata.pkl)")
print("=" * 60)

data = pickle.load(open('EXP/index/metadata.pkl', 'rb'))
meta = data['metadata']

print(f"\nTotal indexed paths: {len(meta)}")
print("\nSample indexed relation_chains:")
for m in meta[:5]:
    print(f"  '{m['relation_chain']}'")

# Check if GT chains exist in index
print("\n" + "=" * 60)
print("MATCHING ANALYSIS")
print("=" * 60)

indexed_chains = set(m['relation_chain'] for m in meta)

row = df.iloc[0]
paths_str = row['paths']
paths = json.loads(paths_str) if isinstance(paths_str, str) else paths_str
gt_chains = [p['relation_chain'] for p in paths]

print(f"\nQuestion: {row['question']}")
print(f"GT chains ({len(gt_chains)}):")
for gc in gt_chains:
    in_index = gc in indexed_chains
    print(f"  '{gc}' -> {'IN INDEX' if in_index else 'NOT IN INDEX'}")
