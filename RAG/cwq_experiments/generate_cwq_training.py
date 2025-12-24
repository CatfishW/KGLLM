"""
Generate CWQ training data and fine-tune retriever.
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

import pandas as pd
import json
import random
from pathlib import Path
from tqdm import tqdm


def parse_cwq_paths(shortest_gt_paths_str):
    """Parse CWQ shortest_gt_paths JSON format into Entity-Enhanced Paths."""
    if not shortest_gt_paths_str or shortest_gt_paths_str == '[]':
        return []
    try:
        # Fix malformed JSON
        fixed = shortest_gt_paths_str.replace('"s ', "'s ").replace('"', '"').replace('"', '"')
        paths = json.loads(fixed)
        result = []
        for p in paths:
            if 'entities' in p and 'relations' in p:
                entities = p['entities']
                relations = p['relations']
                
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


def generate_cwq_training_data():
    print("=" * 60)
    print("Generating CWQ Training Data")
    print("=" * 60)
    
    train = pd.read_parquet('Data/CWQ/shortest_paths/train.parquet')
    val = pd.read_parquet('Data/CWQ/shortest_paths/val.parquet')
    
    print(f"Train samples: {len(train)}")
    print(f"Val samples: {len(val)}")
    
    # Build path corpus from training data
    all_paths = set()
    question_path_pairs = []
    
    for _, row in tqdm(train.iterrows(), total=len(train), desc="Processing train"):
        question = row['question']
        gt_paths = parse_cwq_paths(row['shortest_gt_paths'])
        
        for path in gt_paths:
            if path:
                all_paths.add(path)
                question_path_pairs.append((question, path))
    
    corpus = list(all_paths)
    print(f"Unique paths in corpus: {len(corpus)}")
    print(f"Question-path pairs: {len(question_path_pairs)}")
    
    # Create training triplets (anchor, positive, negative)
    output_dir = Path('./RAG/training_data_cwq')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'train_st.jsonl', 'w') as f:
        for question, pos_path in tqdm(question_path_pairs, desc="Creating triplets"):
            # Sample hard negative (random path that's not the positive)
            neg_candidates = [p for p in corpus if p != pos_path]
            if neg_candidates:
                neg_path = random.choice(neg_candidates)
                triplet = {
                    "anchor": question,
                    "positive": pos_path,
                    "negative": neg_path
                }
                f.write(json.dumps(triplet) + '\n')
    
    # Create validation data
    val_pairs = []
    for _, row in tqdm(val.iterrows(), total=len(val), desc="Processing val"):
        question = row['question']
        gt_paths = parse_cwq_paths(row['shortest_gt_paths'])
        for path in gt_paths:
            if path:
                val_pairs.append((question, path))
    
    with open(output_dir / 'val_st.jsonl', 'w') as f:
        for question, pos_path in val_pairs[:5000]:  # Limit val size
            neg_candidates = [p for p in corpus if p != pos_path]
            if neg_candidates:
                neg_path = random.choice(neg_candidates)
                triplet = {
                    "anchor": question,
                    "positive": pos_path,
                    "negative": neg_path
                }
                f.write(json.dumps(triplet) + '\n')
    
    print(f"\nSaved training data to {output_dir}")
    print(f"Train triplets: {len(question_path_pairs)}")
    print(f"Val triplets: {min(len(val_pairs), 5000)}")


if __name__ == "__main__":
    generate_cwq_training_data()
