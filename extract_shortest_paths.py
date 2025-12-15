"""
Extract shortest ground truth paths that cover all answers.

For each sample, finds the minimum set of shortest paths that together
cover all answer entities.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Any
from tqdm import tqdm
from collections import defaultdict


def parse_json_column(value):
    """Parse JSON string to Python object if needed."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def get_path_target(path: Dict) -> str:
    """Get the target entity (last entity) of a path."""
    entities = path.get('entities', [])
    if entities:
        return entities[-1]
    # Try to parse from full_path
    full_path = path.get('full_path', '')
    if full_path:
        # Format: "(entity1) --[rel]-- (entity2)"
        # Get the last entity in parentheses
        import re
        matches = re.findall(r'\(([^)]+)\)', full_path)
        if matches:
            return matches[-1]
    return ''


def get_path_length(path: Dict) -> int:
    """Get the number of hops in a path."""
    relations = path.get('relations', [])
    return len(relations)


def find_shortest_covering_paths(paths: List[Dict], answer_entities: Set[str]) -> List[Dict]:
    """
    Find the minimum set of shortest paths that cover all answer entities.
    
    Strategy:
    1. Group paths by their target (answer) entity
    2. For each answer entity, select the shortest path
    3. Return the combined set
    """
    if not paths or not answer_entities:
        return []
    
    # Normalize answer entities for matching
    answer_norm = {a.lower().strip() for a in answer_entities}
    
    # Group paths by target entity
    paths_by_target = defaultdict(list)
    for p in paths:
        target = get_path_target(p)
        if target:
            paths_by_target[target.lower().strip()].append(p)
    
    # For each answer entity, find the shortest path
    shortest_paths = []
    covered_answers = set()
    
    for answer in answer_entities:
        answer_norm_key = answer.lower().strip()
        
        if answer_norm_key in paths_by_target:
            # Get all paths to this answer
            target_paths = paths_by_target[answer_norm_key]
            
            # Sort by path length
            target_paths.sort(key=lambda p: get_path_length(p))
            
            # Take the shortest path
            if target_paths:
                shortest_paths.append(target_paths[0])
                covered_answers.add(answer_norm_key)
    
    # If some answers weren't covered directly, try partial matches
    uncovered = answer_norm - covered_answers
    if uncovered:
        for answer in uncovered:
            # Check if any path target contains this answer or vice versa
            for target, target_paths in paths_by_target.items():
                if answer in target or target in answer:
                    target_paths.sort(key=lambda p: get_path_length(p))
                    if target_paths:
                        shortest_paths.append(target_paths[0])
                        break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for p in shortest_paths:
        path_key = p.get('full_path', '') or p.get('relation_chain', '')
        if path_key not in seen:
            seen.add(path_key)
            unique_paths.append(p)
    
    return unique_paths


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process a dataframe to add shortest_gt_paths field."""
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Parse existing fields
        paths = parse_json_column(row.get('paths', '[]'))
        answer = parse_json_column(row.get('answer', '[]'))
        a_entity = parse_json_column(row.get('a_entity', '[]'))
        
        # Use a_entity as answer entities (these are the actual entity IDs)
        # Also include answer text for matching
        answer_entities = set()
        if isinstance(a_entity, list):
            answer_entities.update(a_entity)
        if isinstance(answer, list):
            answer_entities.update(answer)
        
        # Find shortest covering paths
        shortest_paths = find_shortest_covering_paths(paths, answer_entities)
        
        # Create new row with all original fields plus shortest_gt_paths
        new_row = row.to_dict()
        new_row['shortest_gt_paths'] = json.dumps(shortest_paths, ensure_ascii=False)
        results.append(new_row)
    
    return pd.DataFrame(results)


def main():
    input_dir = Path("Data/webqsp_final")
    output_dir = Path("Data/webqsp_final/shortest_paths")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EXTRACTING SHORTEST GROUND TRUTH PATHS")
    print("=" * 60)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        input_path = input_dir / f"{split}.parquet"
        output_path = output_dir / f"{split}.parquet"
        
        if not input_path.exists():
            print(f"\nSkipping {split}: file not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split}...")
        print(f"{'='*60}")
        
        # Load data
        df = pd.read_parquet(input_path)
        print(f"Loaded {len(df)} samples")
        
        # Process
        df_processed = process_dataframe(df)
        
        # Save
        df_processed.to_parquet(output_path, compression='snappy', index=False)
        print(f"Saved to {output_path}")
        
        # Print statistics
        total_paths = 0
        total_shortest = 0
        samples_with_paths = 0
        
        for idx, row in df_processed.iterrows():
            paths = parse_json_column(row.get('paths', '[]'))
            shortest = parse_json_column(row.get('shortest_gt_paths', '[]'))
            
            total_paths += len(paths) if paths else 0
            total_shortest += len(shortest) if shortest else 0
            if shortest:
                samples_with_paths += 1
        
        print(f"\nStatistics for {split}:")
        print(f"  Samples with shortest paths: {samples_with_paths}/{len(df_processed)}")
        print(f"  Avg paths per sample: {total_paths/len(df_processed):.2f}")
        print(f"  Avg shortest paths per sample: {total_shortest/len(df_processed):.2f}")
        
        # Show sample
        sample_row = df_processed.iloc[0]
        print(f"\nSample output:")
        print(f"  Question: {sample_row['question'][:60]}...")
        shortest = parse_json_column(sample_row.get('shortest_gt_paths', '[]'))
        print(f"  Shortest paths ({len(shortest)}):")
        for p in shortest[:3]:
            print(f"    - {p.get('relation_chain', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
