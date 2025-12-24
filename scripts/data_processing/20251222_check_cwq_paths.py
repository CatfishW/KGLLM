"""
Check if Entity-Enhanced Paths provide better lexical overlap with questions.
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

import pandas as pd
import json

def parse_cwq_full_path(shortest_gt_paths_str):
    if not shortest_gt_paths_str or shortest_gt_paths_str == '[]':
        return []
    try:
        fixed = shortest_gt_paths_str.replace('"s ', "'s ").replace('"', '"').replace('"', '"')
        paths = json.loads(fixed)
        result = []
        for p in paths:
            if 'entities' in p and 'relations' in p:
                # Interleave entities and relations
                # e.g. E1 -> R1 -> E2 -> R2 -> E3
                path_str = ""
                entities = p['entities']
                relations = p['relations']
                
                safe_len = min(len(entities), len(relations) + 1)
                
                parts = []
                for i in range(len(relations)):
                    if i < len(entities):
                        parts.append(entities[i])
                    parts.append(relations[i])
                if len(entities) > len(relations):
                    parts.append(entities[-1])
                
                full_path = " -> ".join(parts)
                result.append(full_path)
        return result
    except:
        return []

def check_overlap():
    train = pd.read_parquet('Data/CWQ/shortest_paths/train.parquet')
    
    print("Comparing Relation Chains vs Entity-Enhanced Paths for first 5 samples:")
    
    for i in range(5):
        row = train.iloc[i]
        q = row['question']
        raw_json = row['shortest_gt_paths']
        
        # Current method
        try:
            fixed = raw_json.replace('"s ', "'s ").replace('"', '"').replace('"', '"')
            data = json.loads(fixed)
            rel_chains = [p['relation_chain'] for p in data if 'relation_chain' in p]
        except:
            rel_chains = []
            
        # Proposed method
        ent_paths = parse_cwq_full_path(raw_json)
        
        print(f"\nQ: {q}")
        print(f"  Relation Chain: {rel_chains[0] if rel_chains else 'None'}")
        print(f"  Entity Path:    {ent_paths[0] if ent_paths else 'None'}")

if __name__ == "__main__":
    check_overlap()
