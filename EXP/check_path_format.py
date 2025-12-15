"""Check path formats in both ground truth and retriever output."""
import pandas as pd
import json
import sys
sys.path.insert(0, '.')

# Load test data
df = pd.read_parquet('Data/webqsp_final/test.parquet')
paths_str = df['paths'].iloc[0]
paths = json.loads(paths_str) if isinstance(paths_str, str) else paths_str

print("=" * 60)
print("GROUND TRUTH PATH FORMAT")
print("=" * 60)
print(f"Question: {df['question'].iloc[0]}")
print(f"Number of GT paths: {len(paths)}")
print("\nGT Path relation_chains:")
for p in paths[:5]:
    print(f"  '{p['relation_chain']}'")

# Check retriever output format
print("\n" + "=" * 60)
print("RETRIEVER OUTPUT FORMAT")
print("=" * 60)

from EXP.retriever import KGPathRetriever
from EXP.config import RAGConfig

config = RAGConfig(use_reranker=False, index_dir='EXP/index')
retriever = KGPathRetriever('EXP/index', config)

question = df['question'].iloc[0]
retrieved = retriever.retrieve(question, top_k=5)

print(f"\nRetrieved relation_chains for: '{question}'")
for p in retrieved:
    print(f"  '{p.relation_chain}' (score: {p.score:.4f})")

# Check if any match
print("\n" + "=" * 60)
print("MATCHING CHECK")
print("=" * 60)

gt_chains = set(p['relation_chain'] for p in paths)
retrieved_chains = [p.relation_chain for p in retrieved]

print(f"GT chains: {gt_chains}")
print(f"\nRetrieved chains: {retrieved_chains[:5]}")

# Check exact match
for i, rc in enumerate(retrieved_chains):
    if rc in gt_chains:
        print(f"\n✓ Match found at rank {i+1}: '{rc}'")
        break
else:
    print("\n✗ No exact match in top 5")
    
# Check partial match (single hop relations)
print("\nChecking single-hop relation matches:")
for gt in gt_chains:
    gt_rels = gt.split(' -> ')
    for i, rc in enumerate(retrieved_chains):
        rc_rels = rc.split(' -> ')
        if any(r in gt_rels for r in rc_rels):
            print(f"  Partial match: GT '{gt}' overlaps with retrieved '{rc}'")
