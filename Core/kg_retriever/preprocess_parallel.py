
import os
import sys
import json
import ast
import pandas as pd
import numpy as np
import hashlib
import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import yaml

# ============================================================
# duplicate logic from dataset.py to ensure compatibility
# ============================================================

def parse_row(row):
    """Parse a single data row (static function for multiprocessing)."""
    try:
        # Parse graph
        graph_data = row.get('graph', None)
        if isinstance(graph_data, str):
            try:
                graph = ast.literal_eval(graph_data)
            except:
                return None
        else:
            graph = graph_data
        
        if not graph or len(graph) == 0:
            return None
        
        # Parse ground truth paths
        gt_paths_data = None
        for col in ['shortest_gt_paths', 'ground_truth_paths', 'paths']:
             if col in row and row[col] is not None:
                gt_paths_data = row[col]
                break
        
        if gt_paths_data is None:
            return None

        if isinstance(gt_paths_data, str):
            try:
                gt_paths = json.loads(gt_paths_data)
            except json.JSONDecodeError:
                try:
                    gt_paths = ast.literal_eval(gt_paths_data)
                except:
                    # Try heuristic fix for unescaped quotes in JSON-like string
                    # e.g. ["Men"s Basketball"] -> ["Men's Basketball"]
                    try:
                        # Very naive attempt: replace "s with 's
                        fixed = gt_paths_data.replace('\"s', "'s")
                        gt_paths = ast.literal_eval(fixed)
                    except:
                        return None
        else:
            gt_paths = gt_paths_data
        
        if not gt_paths:
            return None
        
        # Extract q_entity (optional)
        q_entity = row.get('q_entity', None)
        
        # Extract question
        question = row.get('question', None)
        
        return {
            'id': row.get('id', 'unknown'),
            'question': question,
            'q_entity': q_entity,
            'graph': graph,
            'gt_paths': gt_paths
        }
    except Exception:
        return None

def process_file(file_path):
    print(f"Processing {file_path}...")
    
    # Calculate cache path
    # dataset.py logic: cache_key = hashlib.md5("_".join(sorted(data_path)).encode()).hexdigest()[:12]
    # Here we process one file at a time. 
    # WAIT: dataset.py hashes the LIST of paths.
    # If dataset.py loads [path1, path2], cache is hash(path1_path2).
    # THIS IS A PROBLEM. My preprocessor processes files individually.
    # dataset.py must be modified to accept PRE-COMPUTED individual caches?
    # OR dataset.py hashes individual path if passed as list of 1?
    
    # In dataset.py:
    # def _load_data(self, data_path: Union[str, List[str]]):
    #     if isinstance(data_path, str): data_path = [data_path]
    #     cache_key = hashlib.md5("_".join(sorted(data_path)).encode()).hexdigest()[:12]
    
    # So if I run this script, I can't generate the cache file for a "list of paths".
    # BUT, I can generate cache content and save it to a file that *I can force dataset.py to use*?
    # No, better to generate the cache file that matches what dataset.py expects.
    
    # In default.yaml, train_data is a list of 2 files.
    # So dataset.py computes hash of the 2 files combined.
    # And saves ONE pickle file containing data from BOTH.
    
    # So this script must process the LIST of files together.
    return

def main():
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <config_path>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process train, val, test lists
    for key in ['train_data', 'val_data', 'test_data']:
        paths = config.get(key)
        if not paths:
            continue
            
        if isinstance(paths, str):
            paths = [paths]
            
        print(f"\nProcessing {key}: {paths}")
        
        # 1. Determine cache file path
        cache_key = hashlib.md5("_".join(sorted(paths)).encode()).hexdigest()[:12]
        cache_dir = Path(paths[0]).parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"parsed_data_{cache_key}.pkl"
        print(f"Target cache file: {cache_file}")
        
        if cache_file.exists():
            print(f"Cache file already exists. Skipping.")
            continue
        
        # 2. Load and merge all dataframes
        all_rows = []
        for path in paths:
            print(f"  Reading {path}...")
            if path.endswith('.parquet'):
                df = pd.read_parquet(path)
            else:
                df = pd.read_json(path, lines=True)
            print(f"  Loaded {len(df)} rows")
            all_rows.extend(df.to_dict('records'))
            
        print(f"Total rows to parse: {len(all_rows)}")
        
        # 3. Parallel parsing
        num_cores = cpu_count()
        print(f"Parsing with {num_cores} cores...")
        
        with Pool(num_cores) as pool:
            # Use chunks for progress bar
            results = list(tqdm(pool.imap(parse_row, all_rows, chunksize=100), total=len(all_rows)))
        
        # Filter None
        valid_samples = [r for r in results if r is not None]
        print(f"Valid samples: {len(valid_samples)} (Dropped {len(all_rows) - len(valid_samples)})")
        
        # 4. Save pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(valid_samples, f)
        print(f"Saved to {cache_file}")

if __name__ == '__main__':
    main()
