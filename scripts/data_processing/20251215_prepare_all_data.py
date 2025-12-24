"""
Prepare complete training data from webqsp_rog with proper vocabulary coverage.

This script:
1. Loads all train/test/validation data from webqsp_rog parquet files
2. Merges with labeled paths from webqsp_labeled
3. Builds a UNIFIED vocabulary from ALL splits (train + test + val)
4. Saves as parquet and jsonl files ready for training

Key fix: Vocabulary is built from ALL data to prevent UNK tokens during inference.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
from tqdm import tqdm


def load_rog_parquet_files(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all webqsp_rog parquet files and combine them."""
    rog_dir = Path(data_dir) / "webqsp_rog"
    
    # Load train files
    train_files = sorted(rog_dir.glob("train-*.parquet"))
    train_dfs = [pd.read_parquet(f) for f in train_files]
    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"Loaded {len(train_df)} training samples from {len(train_files)} files")
    
    # Load test files
    test_files = sorted(rog_dir.glob("test-*.parquet"))
    test_dfs = [pd.read_parquet(f) for f in test_files]
    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"Loaded {len(test_df)} test samples from {len(test_files)} files")
    
    # Load validation files
    val_files = sorted(rog_dir.glob("validation-*.parquet"))
    val_dfs = [pd.read_parquet(f) for f in val_files]
    val_df = pd.concat(val_dfs, ignore_index=True)
    print(f"Loaded {len(val_df)} validation samples from {len(val_files)} files")
    
    return train_df, test_df, val_df


def load_labeled_paths(data_dir: str) -> Dict[str, List[Dict]]:
    """Load labeled paths from webqsp_labeled."""
    labeled_path = Path(data_dir) / "webqsp_labeled" / "train_with_paths.json"
    
    if not labeled_path.exists():
        print(f"Warning: Labeled paths not found at {labeled_path}")
        return {}
    
    with open(labeled_path, 'r', encoding='utf-8') as f:
        labeled_data = json.load(f)
    
    # Create lookup by ID
    paths_by_id = {}
    for sample in labeled_data:
        sample_id = sample.get('id', '')
        reasoning_paths = sample.get('reasoning_paths', [])
        
        # Convert to simplified format
        paths = []
        seen = set()
        for p in reasoning_paths:
            path_key = p.get('path_string', '')
            if path_key and path_key not in seen:
                seen.add(path_key)
                paths.append({
                    'full_path': p.get('path_string', ''),
                    'relation_chain': p.get('relation_chain', ''),
                    'entities': p.get('entities', []),
                    'relations': p.get('relations', [])
                })
        
        # Sort by path length
        paths.sort(key=lambda x: len(x.get('relations', [])))
        paths_by_id[sample_id] = paths
    
    print(f"Loaded labeled paths for {len(paths_by_id)} samples")
    return paths_by_id


def extract_entities_and_relations(df: pd.DataFrame, paths_by_id: Dict) -> Tuple[Set[str], Set[str]]:
    """Extract all unique entities and relations from a dataframe."""
    entities = set()
    relations = set()
    
    for idx, row in df.iterrows():
        sample_id = row['id']
        
        # From graph
        graph = row.get('graph', [])
        if graph is not None:
            for triple in graph:
                if len(triple) >= 3:
                    entities.add(str(triple[0]))
                    relations.add(str(triple[1]))
                    entities.add(str(triple[2]))
        
        # From q_entity and a_entity
        q_entities = row.get('q_entity', [])
        a_entities = row.get('a_entity', [])
        
        if q_entities is not None:
            for e in q_entities:
                entities.add(str(e))
        if a_entities is not None:
            for e in a_entities:
                entities.add(str(e))
        
        # From paths
        if sample_id in paths_by_id:
            for path in paths_by_id[sample_id]:
                for e in path.get('entities', []):
                    entities.add(str(e))
                for r in path.get('relations', []):
                    relations.add(str(r))
    
    return entities, relations


def build_unified_vocabulary(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    val_df: pd.DataFrame,
    paths_by_id: Dict
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build vocabulary from ALL data splits.
    
    This ensures no UNK tokens during inference.
    """
    print("\nBuilding unified vocabulary from all splits...")
    
    all_entities = set()
    all_relations = set()
    
    # Extract from all splits
    for name, df in [("train", train_df), ("test", test_df), ("val", val_df)]:
        entities, relations = extract_entities_and_relations(df, paths_by_id)
        print(f"  {name}: {len(entities)} entities, {len(relations)} relations")
        all_entities.update(entities)
        all_relations.update(relations)
    
    print(f"\nTotal unique entities: {len(all_entities)}")
    print(f"Total unique relations: {len(all_relations)}")
    
    # Build entity vocabulary with special tokens
    entity2idx = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "<MASK>": 4
    }
    
    # Add all entities (sorted for reproducibility)
    for entity in sorted(all_entities):
        if entity not in entity2idx:
            entity2idx[entity] = len(entity2idx)
    
    # Build relation vocabulary with special tokens
    relation2idx = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<MASK>": 2
    }
    
    # Add all relations (sorted for reproducibility)
    for relation in sorted(all_relations):
        if relation not in relation2idx:
            relation2idx[relation] = len(relation2idx)
    
    print(f"\nFinal vocabulary sizes:")
    print(f"  Entities: {len(entity2idx)} (including 5 special tokens)")
    print(f"  Relations: {len(relation2idx)} (including 3 special tokens)")
    
    return entity2idx, relation2idx


def convert_df_to_samples(df: pd.DataFrame, paths_by_id: Dict) -> List[Dict]:
    """Convert dataframe to list of sample dicts with paths."""
    samples = []
    
    for idx, row in df.iterrows():
        sample_id = row['id']
        
        # Convert numpy arrays to lists
        def to_list(x):
            if x is None:
                return []
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, list):
                return x
            return [x]
        
        # Convert graph triples
        graph = row.get('graph', [])
        if graph is not None:
            graph_list = []
            for triple in graph:
                if len(triple) >= 3:
                    graph_list.append([str(triple[0]), str(triple[1]), str(triple[2])])
            graph = graph_list
        else:
            graph = []
        
        sample = {
            'id': sample_id,
            'question': row.get('question', ''),
            'answer': to_list(row.get('answer', [])),
            'q_entity': to_list(row.get('q_entity', [])),
            'a_entity': to_list(row.get('a_entity', [])),
            'graph': graph,
            'paths': paths_by_id.get(sample_id, [])
        }
        samples.append(sample)
    
    return samples


def save_jsonl(samples: List[Dict], filepath: str):
    """Save samples as JSONL."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved {len(samples)} samples to {filepath}")


def save_parquet(samples: List[Dict], filepath: str):
    """Save samples as Parquet."""
    df = pd.DataFrame(samples)
    
    # Convert list columns to JSON strings for parquet compatibility
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    
    df.to_parquet(filepath, compression='snappy', index=False)
    print(f"Saved {len(samples)} samples to {filepath}")


def save_vocabulary(entity2idx: Dict, relation2idx: Dict, filepath: str):
    """Save vocabulary to JSON."""
    vocab = {
        'entity2idx': entity2idx,
        'relation2idx': relation2idx
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to {filepath}")


def generate_paths_for_samples(samples: List[Dict], max_hops: int = 4) -> List[Dict]:
    """
    Generate paths for samples that don't have labeled paths.
    Uses improved graph traversal to find paths from q_entity to a_entity.
    
    Improvements:
    - Uses both forward and backward edges (original relations, not invented inverses)
    - Increased max hops (default 4)
    - Prioritizes finding paths to ALL answer entities
    - Better path formatting
    """
    samples_with_paths = []
    
    for sample in tqdm(samples, desc="Generating paths"):
        # Always regenerate paths to ensure complete coverage
        # The labeled paths may not cover all answer entities
        existing_paths = []
        existing_targets = set()
        
        # Try to generate paths from graph
        graph = sample.get('graph', [])
        q_entities = set(sample.get('q_entity', []))
        a_entities = set(sample.get('a_entity', []))
        
        if not graph or not q_entities or not a_entities:
            sample['paths'] = existing_paths
            samples_with_paths.append(sample)
            continue
        
        # Build adjacency lists (both directions with original relations)
        forward_adj = defaultdict(list)   # subj -> [(rel, obj)]
        backward_adj = defaultdict(list)  # obj -> [(rel, subj)]
        
        for triple in graph:
            if len(triple) >= 3:
                subj, rel, obj = str(triple[0]), str(triple[1]), str(triple[2])
                forward_adj[subj].append((rel, obj))
                backward_adj[obj].append((rel, subj))
        
        # Find missing answer entities (not covered by existing paths)
        missing_a_entities = a_entities - existing_targets
        
        # BFS to find paths from q_entity to ALL a_entities (not just missing ones)
        # This ensures complete coverage of answer entities
        new_paths = []
        found_targets = set()  # Track which answer entities we've found paths to
        
        for q_entity in q_entities:
            # BFS state: (current_entity, path_entities, path_relations, direction_flags)
            # direction_flags tracks if each edge was forward (True) or backward (False)
            queue = [(q_entity, [q_entity], [], [])]
            visited = {(q_entity, 0)}  # (entity, depth) to allow revisiting at different depths
            
            # Continue until we've found paths to all answer entities or exhausted search
            while queue and (len(found_targets) < len(a_entities) or len(new_paths) < 200):
                curr, path_entities, path_relations, directions = queue.pop(0)
                depth = len(path_relations)
                
                # Check if we reached an answer entity
                if curr in a_entities and depth > 0:
                    found_targets.add(curr)
                    
                    # Build path string
                    path_parts = [f"({path_entities[0]})"]
                    for i, (rel, is_forward) in enumerate(zip(path_relations, directions)):
                        if is_forward:
                            path_parts.append(f" --[{rel}]--> ({path_entities[i+1]})")
                        else:
                            # Backward edge: show as inverse relationship  
                            path_parts.append(f" <--[{rel}]-- ({path_entities[i+1]})")
                    
                    full_path = "".join(path_parts)
                    
                    new_paths.append({
                        'full_path': full_path,
                        'relation_chain': " -> ".join(path_relations),
                        'entities': path_entities.copy(),
                        'relations': path_relations.copy()
                    })
                    
                    # Don't stop here - continue to find alternative paths
                    if depth >= max_hops:
                        continue
                
                # Expand if within depth limit
                if depth < max_hops:
                    # Forward edges
                    for rel, neighbor in forward_adj[curr]:
                        state = (neighbor, depth + 1)
                        if state not in visited:
                            visited.add(state)
                            queue.append((
                                neighbor,
                                path_entities + [neighbor],
                                path_relations + [rel],
                                directions + [True]
                            ))
                    
                    # Backward edges (traverse in reverse direction)
                    for rel, neighbor in backward_adj[curr]:
                        state = (neighbor, depth + 1)
                        if state not in visited:
                            visited.add(state)
                            queue.append((
                                neighbor,
                                path_entities + [neighbor],
                                path_relations + [rel],
                                directions + [False]
                            ))
        
        # Combine existing and new paths, deduplicate
        all_paths = existing_paths + new_paths
        seen = set()
        unique_paths = []
        
        # Prioritize paths to different answer entities
        paths_by_target = defaultdict(list)
        for p in all_paths:
            entities = p.get('entities', [])
            if entities:
                target = entities[-1]
                paths_by_target[target].append(p)
        
        # Select best path for each target (shortest first)
        for target, target_paths in paths_by_target.items():
            target_paths.sort(key=lambda x: len(x.get('relations', [])))
            for p in target_paths[:3]:  # Keep up to 3 paths per target
                key = p.get('full_path', '')
                if key and key not in seen:
                    seen.add(key)
                    unique_paths.append(p)
        
        # Sort final paths by length
        unique_paths.sort(key=lambda x: len(x.get('relations', [])))
        sample['paths'] = unique_paths[:100]  # Keep top 100 paths (increased from 50)
        samples_with_paths.append(sample)
    
    return samples_with_paths


def main():
    data_dir = "Data"
    output_dir = "Data/webqsp_final"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PREPARING COMPLETE WEBQSP DATASET")
    print("=" * 70)
    
    # Step 1: Load all parquet files
    print("\n[1/6] Loading webqsp_rog parquet files...")
    train_df, test_df, val_df = load_rog_parquet_files(data_dir)
    
    # Step 2: Load labeled paths
    print("\n[2/6] Loading labeled paths...")
    paths_by_id = load_labeled_paths(data_dir)
    
    # Step 3: Build unified vocabulary from ALL data
    print("\n[3/6] Building unified vocabulary...")
    entity2idx, relation2idx = build_unified_vocabulary(
        train_df, test_df, val_df, paths_by_id
    )
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, "vocab.json")
    save_vocabulary(entity2idx, relation2idx, vocab_path)
    
    # Step 4: Convert to sample format
    print("\n[4/6] Converting to sample format...")
    train_samples = convert_df_to_samples(train_df, paths_by_id)
    test_samples = convert_df_to_samples(test_df, paths_by_id)
    val_samples = convert_df_to_samples(val_df, paths_by_id)
    
    # Step 5: Generate paths for samples without labeled paths
    print("\n[5/6] Generating paths for unlabeled samples...")
    train_samples = generate_paths_for_samples(train_samples)
    test_samples = generate_paths_for_samples(test_samples)
    val_samples = generate_paths_for_samples(val_samples)
    
    # Count samples with paths
    train_with_paths = sum(1 for s in train_samples if s.get('paths'))
    test_with_paths = sum(1 for s in test_samples if s.get('paths'))
    val_with_paths = sum(1 for s in val_samples if s.get('paths'))
    
    print(f"\nSamples with paths:")
    print(f"  Train: {train_with_paths}/{len(train_samples)}")
    print(f"  Test: {test_with_paths}/{len(test_samples)}")
    print(f"  Val: {val_with_paths}/{len(val_samples)}")
    
    # Step 6: Save all files
    print("\n[6/6] Saving output files...")
    
    # Save JSONL
    save_jsonl(train_samples, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(test_samples, os.path.join(output_dir, "test.jsonl"))
    save_jsonl(val_samples, os.path.join(output_dir, "val.jsonl"))
    
    # Save Parquet
    save_parquet(train_samples, os.path.join(output_dir, "train.parquet"))
    save_parquet(test_samples, os.path.join(output_dir, "test.parquet"))
    save_parquet(val_samples, os.path.join(output_dir, "val.parquet"))
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"\nVocabulary sizes:")
    print(f"  Entities: {len(entity2idx)}")
    print(f"  Relations: {len(relation2idx)}")
    print(f"\nOutput files:")
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {f}: {size_mb:.2f} MB")
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print("\nTo train with this data, use:")
    print(f"  --train_path {output_dir}/train.jsonl")
    print(f"  --val_path {output_dir}/val.jsonl")
    print(f"  --vocab_path {output_dir}/vocab.json")


if __name__ == "__main__":
    main()

