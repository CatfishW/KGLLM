
import pandas as pd
import json
import re
import sys
# Mock imports to avoid dependency issues if any
from typing import List, Optional

# Mock LABEL_MAP for 3 classes
LABEL_MAP_3 = {
    'one_hop': 0,
    'multi_hop': 1,
    'numeric': 2
}
LABEL_NAMES_3 = ['one_hop', 'multi_hop', 'numeric']

def is_numeric_answer(row) -> bool:
    """Check if the answer is numeric using multiple heuristics."""
    
    # 1. Check a_entity (usually contains entity names or values)
    if 'a_entity' in row:
        a_ent = row['a_entity']
        if isinstance(a_ent, (list, np.ndarray)) and len(a_ent) > 0:
            val = str(a_ent[0])
            # Check for pure numbers, dates, currency
            if re.match(r'^[\d\.,\$€£]+$', val.strip()):
                return True
            # Check for Year
            if re.match(r'^\d{4}$', val.strip()):
                return True
    
    # 2. Check answer (raw answer string/list)
    if 'answer' in row:
        ans = row['answer']
        # normalize
        if isinstance(ans, str):
            try:
                ans = json.loads(ans.replace("'", '"'))
            except:
                pass
        
        if isinstance(ans, list) and len(ans) > 0:
            val = str(ans[0])
            if re.match(r'^[\d\.,\$€£]+$', val.strip()):
                return True
    
    return False

def get_hops(row):
    """Get hop count from shortest_gt_paths"""
    if 'shortest_gt_paths' in row:
        try:
            paths = row['shortest_gt_paths']
            if isinstance(paths, str):
                 paths = json.loads(paths.replace("'", '"'))
            
            if paths and len(paths) > 0:
                path = paths[0]
                if isinstance(path, dict):
                    if 'full_path' in path: 
                        return path['full_path'].count('-->')
                    if 'relations' in path:
                        return len(path['relations'])
        except:
            pass
    
    # Fallback to gt_paths if available
    if 'gt_paths' in row:
         gt = row['gt_paths']
         if isinstance(gt, (list, np.ndarray)) and len(gt) > 0:
             return len(gt[0])
    return None

import numpy as np

def audit_labels():
    print("Loading WebQSP shortest_paths data...")
    df = pd.read_parquet('/data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/train.parquet')
    
    print(f"Total samples: {len(df)}")
    
    numeric_count = 0
    one_hop_count = 0
    multi_hop_count = 0
    
    print("\n--- Sampling Labels ---")
    
    for idx, row in df.iterrows():
        is_num = is_numeric_answer(row)
        hops = get_hops(row)
        
        label = "?"
        if is_num:
            label = "numeric"
            numeric_count += 1
        elif hops == 1:
            label = "one_hop"
            one_hop_count += 1
        elif hops and hops > 1:
            label = "multi_hop"
            multi_hop_count += 1
        
        # Print a few of each to verify
        if idx < 20: 
            print(f"Q: {row['question']}")
            print(f"   Answer: {row.get('answer', 'N/A')}")
            print(f"   Hops: {hops}, IsNumeric: {is_num} => Label: {label}")
            print("-" * 30)

    print("\n--- Summary ---")
    print(f"Numeric: {numeric_count} ({numeric_count/len(df)*100:.2f}%)")
    print(f"One Hop: {one_hop_count} ({one_hop_count/len(df)*100:.2f}%)")
    print(f"Multi Hop: {multi_hop_count} ({multi_hop_count/len(df)*100:.2f}%)")

if __name__ == "__main__":
    audit_labels()
