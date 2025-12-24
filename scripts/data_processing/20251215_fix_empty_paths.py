"""
Script to identify and fix samples with empty ground truth paths in validation data.
"""
import pandas as pd
import json
from pathlib import Path

def check_and_fix_validation_data(val_file='Data/webqsp_final/val.parquet'):
    """Check validation data for empty paths and provide options to fix."""
    
    print(f"Loading validation data from {val_file}...")
    df = pd.read_parquet(val_file)
    print(f"Total samples: {len(df)}")
    
    # Find samples with empty paths
    empty_samples = []
    for idx, row in df.iterrows():
        paths_str = row.get('paths', '[]')
        if isinstance(paths_str, str):
            paths = json.loads(paths_str)
        else:
            paths = paths_str
        
        if not isinstance(paths, list) or len(paths) == 0:
            empty_samples.append({
                'index': idx,
                'id': row.get('id'),
                'question': row.get('question'),
                'answer': row.get('answer'),
                'q_entity': row.get('q_entity'),
                'a_entity': row.get('a_entity'),
                'graph_size': len(json.loads(row.get('graph', '[]')) if isinstance(row.get('graph', '[]'), str) else row.get('graph', []))
            })
    
    print(f"\nFound {len(empty_samples)} samples with empty paths:")
    for sample in empty_samples:
        print(f"  {sample['id']}: {sample['question']}")
        print(f"    Graph size: {sample['graph_size']} triples")
        print(f"    Q entity: {sample['q_entity']}")
        print(f"    A entity: {sample['a_entity']}")
    
    # Option 1: Create filtered validation set (without empty paths)
    if empty_samples:
        print(f"\n{'='*60}")
        print("Option 1: Create filtered validation set")
        print(f"{'='*60}")
        filtered_df = df.drop([s['index'] for s in empty_samples])
        output_file = val_file.replace('.parquet', '_filtered.parquet')
        filtered_df.to_parquet(output_file, index=False)
        print(f"Created filtered validation set: {output_file}")
        print(f"  Original: {len(df)} samples")
        print(f"  Filtered: {len(filtered_df)} samples (removed {len(empty_samples)})")
    
    # Option 2: Check if we can find paths in other data sources
    print(f"\n{'='*60}")
    print("Option 2: Check other data sources for paths")
    print(f"{'='*60}")
    
    # Check labeled data
    labeled_file = 'Data/webqsp_labeled/train_paths.jsonl'
    if Path(labeled_file).exists():
        print(f"\nChecking {labeled_file}...")
        found_in_labeled = []
        with open(labeled_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                if sample.get('id') in [s['id'] for s in empty_samples]:
                    paths = sample.get('paths', [])
                    if paths:
                        found_in_labeled.append(sample.get('id'))
        
        if found_in_labeled:
            print(f"  Found paths for {len(found_in_labeled)} samples in labeled data:")
            for sid in found_in_labeled:
                print(f"    {sid}")
        else:
            print("  No paths found for these samples in labeled data")
    
    # Check combined data
    combined_file = 'Data/webqsp_combined/train_combined.parquet'
    if Path(combined_file).exists():
        print(f"\nChecking {combined_file}...")
        df_combined = pd.read_parquet(combined_file)
        found_in_combined = []
        for empty_id in [s['id'] for s in empty_samples]:
            sample = df_combined[df_combined['id'] == empty_id]
            if len(sample) > 0:
                s = sample.iloc[0]
                paths_str = s.get('paths', '[]')
                paths = json.loads(paths_str) if isinstance(paths_str, str) else paths_str
                if isinstance(paths, list) and len(paths) > 0:
                    found_in_combined.append(empty_id)
        
        if found_in_combined:
            print(f"  Found paths for {len(found_in_combined)} samples in combined data:")
            for sid in found_in_combined:
                print(f"    {sid}")
        else:
            print("  No paths found for these samples in combined data")
    
    return empty_samples

if __name__ == '__main__':
    empty_samples = check_and_fix_validation_data()
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total samples with empty paths: {len(empty_samples)}")
    print("\nRecommendation:")
    print("1. Use the filtered validation set for training (val_filtered.parquet)")
    print("2. Or modify the training code to skip samples with empty paths during validation")
    print("3. Or regenerate the validation data from a source that has paths for all samples")

