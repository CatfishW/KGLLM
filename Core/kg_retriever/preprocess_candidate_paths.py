"""
Preprocess KG data to extract candidate relation paths.

For each sample:
1. Build graph adjacency from triples
2. Extract all unique relation paths (1-4 hops) starting from topic entities
3. Save as preprocessed dataset with question + candidate paths

This enables faster training with better relation path embeddings.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def parse_json_field(value):
    """Parse JSON or ast.literal_eval string fields."""
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            import ast
            try:
                return ast.literal_eval(value)
            except:
                return None
    return None


def build_graph_adjacency(triples: List[List[str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build adjacency list from triples.
    
    Returns:
        adj: Dict mapping entity -> list of (relation, neighbor) tuples
    """
    adj = defaultdict(list)
    for triple in triples:
        if len(triple) >= 3:
            head, rel, tail = triple[0], triple[1], triple[2]
            adj[head].append((rel, tail))
            # Add reverse edges for bidirectional traversal
            adj[tail].append((rel, head))
    return adj


def extract_relation_paths(
    adj: Dict[str, List[Tuple[str, str]]],
    start_entities: List[str],
    max_hops: int = 4,
    max_paths: int = 1000,
) -> List[List[str]]:
    """
    Extract all unique relation paths from start entities up to max_hops.
    
    Uses BFS to find all paths, returns unique relation sequences.
    """
    all_paths = set()
    
    # BFS queue: (current_entity, path_so_far)
    queue = [(e, []) for e in start_entities if e in adj]
    
    visited_states = set()  # (entity, tuple(path)) to avoid cycles
    
    while queue and len(all_paths) < max_paths:
        entity, path = queue.pop(0)
        
        # Add current path if non-empty
        if path:
            path_tuple = tuple(path)
            all_paths.add(path_tuple)
        
        # Stop if max hops reached
        if len(path) >= max_hops:
            continue
        
        # Explore neighbors
        for rel, neighbor in adj.get(entity, []):
            new_path = path + [rel]
            state = (neighbor, tuple(new_path))
            
            # Avoid revisiting same state
            if state in visited_states:
                continue
            visited_states.add(state)
            
            queue.append((neighbor, new_path))
    
    # Convert back to list of lists
    return [list(p) for p in all_paths]


def get_unique_relations(triples: List[List[str]]) -> List[str]:
    """Get all unique relations from graph triples."""
    relations = set()
    for triple in triples:
        if len(triple) >= 3:
            relations.add(triple[1])
    return sorted(list(relations))


def process_sample(row: Dict, max_hops: int = 4, max_paths: int = 500) -> Dict:
    """Process a single sample to extract candidate paths."""
    try:
        # Parse graph
        graph = parse_json_field(row.get('graph'))
        if not graph:
            return None
        
        # Parse topic entities
        q_entities = parse_json_field(row.get('q_entity', []))
        if isinstance(q_entities, str):
            q_entities = [q_entities]
        if not q_entities:
            q_entities = []
        
        # Parse ground truth paths
        gt_paths_data = parse_json_field(row.get('shortest_gt_paths') or row.get('paths'))
        gt_paths = []
        if gt_paths_data:
            for p in gt_paths_data:
                if isinstance(p, dict) and 'relations' in p:
                    gt_paths.append(p['relations'])
                elif isinstance(p, list):
                    gt_paths.append(p)
        
        # Build adjacency
        adj = build_graph_adjacency(graph)
        
        # Extract candidate paths
        if q_entities:
            candidate_paths = extract_relation_paths(adj, q_entities, max_hops, max_paths)
        else:
            # If no topic entities, use all unique relations as 1-hop candidates
            candidate_paths = [[r] for r in get_unique_relations(graph)]
        
        # Also add ground truth paths to candidates (ensure they're included)
        for gt_path in gt_paths:
            if gt_path and gt_path not in candidate_paths:
                candidate_paths.append(gt_path)
        
        # Get unique relations for this sample
        unique_relations = get_unique_relations(graph)
        
        return {
            'id': row['id'],
            'question': row['question'],
            'q_entity': q_entities,
            'a_entity': parse_json_field(row.get('a_entity', [])),
            'gt_paths': gt_paths,
            'candidate_paths': candidate_paths,
            'unique_relations': unique_relations,
            'num_triples': len(graph),
            'num_candidates': len(candidate_paths),
        }
    except Exception as e:
        print(f"Error processing {row.get('id', 'unknown')}: {e}")
        return None


def process_dataset(
    input_path: str,
    output_path: str,
    max_hops: int = 4,
    max_paths: int = 500,
    num_workers: int = 8,
):
    """Process entire dataset to extract candidate paths."""
    print(f"Loading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples")
    
    # Convert to list of dicts
    rows = df.to_dict('records')
    
    # Process samples
    results = []
    process_fn = partial(process_sample, max_hops=max_hops, max_paths=max_paths)
    
    # Use multiprocessing for speed
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for result in tqdm(pool.imap(process_fn, rows), total=len(rows), desc="Processing"):
                if result:
                    results.append(result)
    else:
        for row in tqdm(rows, desc="Processing"):
            result = process_fn(row)
            if result:
                results.append(result)
    
    print(f"Processed {len(results)} samples")
    
    # Statistics
    avg_candidates = np.mean([r['num_candidates'] for r in results])
    avg_triples = np.mean([r['num_triples'] for r in results])
    print(f"Avg candidates per sample: {avg_candidates:.1f}")
    print(f"Avg triples per sample: {avg_triples:.1f}")
    
    # Save as parquet
    output_df = pd.DataFrame(results)
    output_df.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Preprocess KG data to extract candidate paths")
    parser.add_argument('--input', type=str, required=True, help="Input parquet file")
    parser.add_argument('--output', type=str, required=True, help="Output parquet file")
    parser.add_argument('--max_hops', type=int, default=4, help="Maximum path length")
    parser.add_argument('--max_paths', type=int, default=500, help="Maximum paths per sample")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()
    
    process_dataset(
        args.input,
        args.output,
        max_hops=args.max_hops,
        max_paths=args.max_paths,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
