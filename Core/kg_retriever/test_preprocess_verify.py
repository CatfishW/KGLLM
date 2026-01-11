
import sys
import os
import json
# Add the directory to path so we can import the module
sys.path.append('/data/Yanlai/KGLLM/Core/kg_retriever')

from preprocess_candidate_paths import process_sample

# Mock data
# Note: we need to use json.dumps for the string fields because parse_json_field expects json strings
row_1hop = {
    'id': '1',
    'question': 'who is finding nemo?',
    'q_entity': '["Finding Nemo"]',
    'a_entity': None,
    'shortest_gt_paths': json.dumps([{"relations": ["relation1"]}]),  # 1-hop
    'graph': json.dumps([["Finding Nemo", "relation1", "Answer"], ["Finding Nemo", "relation2", "Node2"], ["Node2", "relation3", "Node3"]])
}

row_2hop = {
    'id': '2',
    'question': 'who directed the movie finding nemo?',
    'q_entity': '["Finding Nemo"]',
    'a_entity': None,
    'shortest_gt_paths': json.dumps([{"relations": ["relation2", "relation3"]}]),  # 2-hop
    'graph': json.dumps([["Finding Nemo", "relation1", "Node1"], ["Finding Nemo", "relation2", "Node2"], ["Node2", "relation3", "Answer"]])
}

print("Testing 1-hop sample...")
res1 = process_sample(row_1hop)
print(f"Is multihop: {res1['is_multihop']}")
print(f"Candidate paths lengths: {[len(p) for p in res1['candidate_paths']]}")
# Expect: only length 1 paths. 
# Graph has: r1 (len 1), r2 (len 1), r2->r3 (len 2).
# Should filter to just r1 and r2.

print("\nTesting 2-hop sample...")
res2 = process_sample(row_2hop)
print(f"Is multihop: {res2['is_multihop']}")
print(f"Candidate paths lengths: {[len(p) for p in res2['candidate_paths']]}")
# Expect: is_multihop True.
# Graph has: r1 (len 1), r2 (len 1), r2->r3 (len 2).
# Should filter to just r2->r3.
