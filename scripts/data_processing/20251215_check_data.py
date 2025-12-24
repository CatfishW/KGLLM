import json
import pandas as pd
import os

# Check labeled data
print("Checking labeled data...")
count = 0
empty_paths = 0
webqtrn9_data = None

labeled_file = 'Data/webqsp_labeled/train_paths.jsonl'
if not os.path.exists(labeled_file):
    labeled_file = 'Data/webqsp_labeled/train_with_paths.json'
    
with open(labeled_file, 'r', encoding='utf-8') as f:
    if labeled_file.endswith('.jsonl'):
        for line in f:
            sample = json.loads(line)
            count += 1
            paths = sample.get('paths', [])
            if len(paths) == 0:
                empty_paths += 1
            if sample.get('id') == 'WebQTrn-9':
                webqtrn9_data = sample
    else:
        data = json.load(f)
        if isinstance(data, list):
            for sample in data:
                count += 1
                paths = sample.get('paths', [])
                if len(paths) == 0:
                    empty_paths += 1
                if sample.get('id') == 'WebQTrn-9':
                    webqtrn9_data = sample
        elif isinstance(data, dict):
            for sample_id, sample in data.items():
                count += 1
                paths = sample.get('paths', [])
                if len(paths) == 0:
                    empty_paths += 1
                if sample.get('id') == 'WebQTrn-9' or sample_id == 'WebQTrn-9':
                    webqtrn9_data = sample

print(f'Total samples in labeled data: {count}')
print(f'Samples with empty paths: {empty_paths}')
print(f'\nWebQTrn-9 data:')
if webqtrn9_data:
    print(f"  ID: {webqtrn9_data.get('id')}")
    print(f"  Question: {webqtrn9_data.get('question')}")
    print(f"  Paths: {len(webqtrn9_data.get('paths', []))}")
    if webqtrn9_data.get('paths'):
        print(f"  First path: {webqtrn9_data['paths'][0]}")
    else:
        print("  No paths found!")
else:
    print("  WebQTrn-9 not found in labeled data!")

# Check combined data
print("\nChecking combined data...")
df = pd.read_parquet('Data/webqsp_combined/train_combined.parquet')
print(f'Total samples in combined data: {len(df)}')

# Check for WebQTrn-9
sample = df[df['id'] == 'WebQTrn-9']
if len(sample) > 0:
    s = sample.iloc[0]
    paths_str = s.get('paths', '[]')
    if isinstance(paths_str, str):
        paths = json.loads(paths_str)
    else:
        paths = paths_str
    print(f'\nWebQTrn-9 in combined data:')
    print(f"  Question: {s.get('question')}")
    print(f"  Paths count: {len(paths) if isinstance(paths, list) else 'N/A'}")
    if isinstance(paths, list) and len(paths) > 0:
        print(f"  First path: {paths[0]}")
    else:
        print("  No paths found!")
else:
    print("  WebQTrn-9 not found in combined data!")

# Count empty paths in combined data
empty_count = 0
for idx, row in df.iterrows():
    paths_str = row.get('paths', '[]')
    if isinstance(paths_str, str):
        paths = json.loads(paths_str)
    else:
        paths = paths_str
    if not isinstance(paths, list) or len(paths) == 0:
        empty_count += 1

print(f'\nSamples with empty paths in combined data: {empty_count}')

# Check validation data (webqsp_combined)
print("\nChecking validation data (webqsp_combined)...")
val_file = 'Data/webqsp_combined/val.jsonl'
if os.path.exists(val_file):
    val_count = 0
    val_empty = 0
    val_webqtrn9 = None
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            val_count += 1
            paths = sample.get('paths', [])
            if len(paths) == 0:
                val_empty += 1
            if sample.get('id') == 'WebQTrn-9':
                val_webqtrn9 = sample
    
    print(f'Total samples in validation data: {val_count}')
    print(f'Samples with empty paths: {val_empty}')
    if val_webqtrn9:
        print(f'\nWebQTrn-9 in validation data:')
        print(f"  Question: {val_webqtrn9.get('question')}")
        print(f"  Paths count: {len(val_webqtrn9.get('paths', []))}")
        if val_webqtrn9.get('paths'):
            print(f"  First path: {val_webqtrn9['paths'][0]}")
        else:
            print("  No paths found!")
    else:
        print("  WebQTrn-9 not found in validation data!")
else:
    print(f"  Validation file not found: {val_file}")

# Check validation data (webqsp_final) - the one actually used in training
print("\nChecking validation data (webqsp_final - used in training)...")
val_file_final = 'Data/webqsp_final/val.parquet'
if os.path.exists(val_file_final):
    df_val = pd.read_parquet(val_file_final)
    print(f'Total samples in validation data: {len(df_val)}')
    
    # Check for WebQTrn-9
    sample = df_val[df_val['id'] == 'WebQTrn-9']
    if len(sample) > 0:
        s = sample.iloc[0]
        paths_str = s.get('paths', '[]')
        if isinstance(paths_str, str):
            paths = json.loads(paths_str)
        else:
            paths = paths_str
        print(f'\nWebQTrn-9 in validation data:')
        print(f"  Question: {s.get('question')}")
        print(f"  Paths count: {len(paths) if isinstance(paths, list) else 'N/A'}")
        if isinstance(paths, list) and len(paths) > 0:
            print(f"  First path: {paths[0]}")
        else:
            print("  No paths found!")
    else:
        print("  WebQTrn-9 not found in validation data!")
    
    # Count empty paths
    empty_count = 0
    for idx, row in df_val.iterrows():
        paths_str = row.get('paths', '[]')
        if isinstance(paths_str, str):
            paths = json.loads(paths_str)
        else:
            paths = paths_str
        if not isinstance(paths, list) or len(paths) == 0:
            empty_count += 1
    
    print(f'\nSamples with empty paths: {empty_count}')
else:
    print(f"  Validation file not found: {val_file_final}")

