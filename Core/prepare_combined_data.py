"""
Combine webqsp_rog (graph data) with labeled paths for training.
Creates a single dataset with both graph structure and target paths.
"""

import pandas as pd
import json
import os
from typing import Dict, List
from tqdm import tqdm


def load_rog_data(rog_dir: str) -> Dict[str, Dict]:
    """Load all ROG parquet files into a dictionary keyed by ID."""
    rog_data = {}
    
    parquet_files = [f for f in os.listdir(rog_dir) if f.endswith('.parquet')]
    
    for pf in parquet_files:
        print(f"Loading {pf}...")
        df = pd.read_parquet(os.path.join(rog_dir, pf))
        
        for _, row in df.iterrows():
            sample_id = row['id']
            rog_data[sample_id] = {
                'id': sample_id,
                'question': row['question'],
                'answer': list(row['answer']) if hasattr(row['answer'], '__iter__') and not isinstance(row['answer'], str) else [row['answer']],
                'q_entity': list(row['q_entity']) if hasattr(row['q_entity'], '__iter__') and not isinstance(row['q_entity'], str) else [row['q_entity']],
                'a_entity': list(row['a_entity']) if hasattr(row['a_entity'], '__iter__') and not isinstance(row['a_entity'], str) else [row['a_entity']],
                'graph': [list(triple) for triple in row['graph']] if row['graph'] is not None else []
            }
    
    print(f"Loaded {len(rog_data)} samples from ROG data")
    return rog_data


def load_labeled_data(labeled_path: str) -> Dict[str, Dict]:
    """Load labeled paths data."""
    labeled_data = {}
    
    print(f"Loading labeled data from {labeled_path}...")
    with open(labeled_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            labeled_data[sample['id']] = sample
    
    print(f"Loaded {len(labeled_data)} samples from labeled data")
    return labeled_data


def combine_data(rog_data: Dict[str, Dict], labeled_data: Dict[str, Dict]) -> List[Dict]:
    """Combine ROG graph data with labeled paths."""
    combined = []
    matched = 0
    
    for sample_id, labeled in tqdm(labeled_data.items(), desc="Combining data"):
        if sample_id in rog_data:
            rog = rog_data[sample_id]
            
            # Combine: use graph from ROG, paths from labeled
            combined_sample = {
                'id': sample_id,
                'question': labeled['question'],
                'answer': labeled['answer'] if isinstance(labeled['answer'], list) else [labeled['answer']],
                'q_entity': labeled['q_entity'],
                'a_entity': labeled['a_entity'],
                'graph': rog['graph'],  # Graph from ROG
                'paths': labeled['paths']  # Paths from labeled data
            }
            combined.append(combined_sample)
            matched += 1
    
    print(f"Combined {matched} samples (matched ROG with labeled)")
    return combined


def save_combined_data(data: List[Dict], output_dir: str):
    """Save combined data in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSONL
    jsonl_path = os.path.join(output_dir, 'train_combined.jsonl')
    print(f"Saving to {jsonl_path}...")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Save as Parquet
    parquet_path = os.path.join(output_dir, 'train_combined.parquet')
    print(f"Saving to {parquet_path}...")
    
    # Convert to DataFrame-friendly format
    df_data = []
    for sample in data:
        df_data.append({
            'id': sample['id'],
            'question': sample['question'],
            'answer': json.dumps(sample['answer']),
            'q_entity': json.dumps(sample['q_entity']),
            'a_entity': json.dumps(sample['a_entity']),
            'graph': json.dumps(sample['graph']),
            'paths': json.dumps(sample['paths'])
        })
    
    df = pd.DataFrame(df_data)
    df.to_parquet(parquet_path, index=False)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("COMBINED DATA STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(data)}")
    print(f"JSONL file: {jsonl_path} ({os.path.getsize(jsonl_path) / 1024 / 1024:.2f} MB)")
    print(f"Parquet file: {parquet_path} ({os.path.getsize(parquet_path) / 1024 / 1024:.2f} MB)")
    
    # Sample stats
    avg_graph_size = sum(len(s['graph']) for s in data) / len(data)
    avg_paths = sum(len(s['paths']) for s in data) / len(data)
    print(f"Avg graph triples per sample: {avg_graph_size:.1f}")
    print(f"Avg paths per sample: {avg_paths:.1f}")
    
    return jsonl_path, parquet_path


def create_train_val_split(data: List[Dict], val_ratio: float = 0.1, seed: int = 42):
    """Split data into train and validation sets."""
    import random
    random.seed(seed)
    
    data_shuffled = data.copy()
    random.shuffle(data_shuffled)
    
    val_size = int(len(data_shuffled) * val_ratio)
    val_data = data_shuffled[:val_size]
    train_data = data_shuffled[val_size:]
    
    return train_data, val_data


def main():
    # Paths
    rog_dir = "../Data/webqsp_rog"
    labeled_path = "../Data/webqsp_labeled/train_paths.jsonl"
    output_dir = "../Data/webqsp_combined"
    
    # Load data
    rog_data = load_rog_data(rog_dir)
    labeled_data = load_labeled_data(labeled_path)
    
    # Combine
    combined_data = combine_data(rog_data, labeled_data)
    
    # Split into train/val
    train_data, val_data = create_train_val_split(combined_data, val_ratio=0.1)
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Save all data
    save_combined_data(combined_data, output_dir)
    
    # Save train split
    train_path = os.path.join(output_dir, 'train.jsonl')
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved train split to {train_path}")
    
    # Save val split
    val_path = os.path.join(output_dir, 'val.jsonl')
    with open(val_path, 'w', encoding='utf-8') as f:
        for sample in val_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved val split to {val_path}")
    
    # Show sample
    print(f"\n{'='*60}")
    print("SAMPLE OUTPUT")
    print(f"{'='*60}")
    sample = combined_data[0]
    print(f"ID: {sample['id']}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Graph triples: {len(sample['graph'])}")
    print(f"Target paths: {len(sample['paths'])}")
    if sample['paths']:
        print(f"First path: {sample['paths'][0]['full_path']}")


if __name__ == '__main__':
    main()

