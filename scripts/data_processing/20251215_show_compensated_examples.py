import pandas as pd
import json
import os

base_dir = "/data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/shortest_paths_missing_compensated"
files = ["train.parquet", "test.parquet", "val.parquet"]

def safe_json_loads(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return []
    return x if x is not None else []

def is_compensated(x):
    val = safe_json_loads(x)
    # Check if it matches the structure we created: list with 1 dict, having empty entities list
    if len(val) > 0 and isinstance(val[0], dict):
        return val[0].get('entities') == []
    return False

for f in files:
    path = os.path.join(base_dir, f)
    if not os.path.exists(path):
        continue
        
    print(f"--- {f} ---")
    df = pd.read_parquet(path)
    
    compensated_mask = df['shortest_gt_paths'].apply(is_compensated)
    compensated_df = df[compensated_mask]
    
    if compensated_df.empty:
        print("No compensated entries found (or logic mismatch).")
        continue
        
    for idx, row in compensated_df.iterrows():
        paths = safe_json_loads(row['shortest_gt_paths'])
        llm_path = paths[0].get('relation_chain', 'N/A')
        
        print(f"Question: {row['question']}")
        print(f"Answer: {row['answer']}")
        print(f"Compensated Path: {llm_path}")
        print("-" * 20)

