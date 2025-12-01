"""
Data preparation utilities for GSR (no torch dependency).

These functions only process data and don't require torch/transformers.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsr.subgraph_index import SubgraphIndex


def prepare_gsr_training_data(
    data_path: str,
    subgraph_index_path: str,
    output_path: str
):
    """
    Prepare training data for GSR model.
    
    Format: (question, subgraph_id) pairs
    
    Args:
        data_path: Path to dataset with questions and paths
        subgraph_index_path: Path to subgraph index
        output_path: Path to save training data
    """
    # Load subgraph index
    index = SubgraphIndex.load(subgraph_index_path)
    
    # Load dataset
    path = Path(data_path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
        samples = df.to_dict('records')
        for sample in samples:
            for key in ['graph', 'paths', 'answer', 'q_entity', 'a_entity']:
                if key in sample and isinstance(sample[key], str):
                    try:
                        sample[key] = json.loads(sample[key])
                    except:
                        pass
    elif path.suffix in ['.jsonl', '.json']:
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            if path.suffix == '.jsonl':
                for line in f:
                    samples.append(json.loads(line))
            else:
                samples = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Create training samples
    training_samples = []
    
    for sample in samples:
        question = sample.get('question', '')
        paths = sample.get('paths', [])
        
        if not question or not paths:
            continue
        
        # Get subgraph ID for each path
        for path in paths:
            if isinstance(path, dict):
                relations = path.get('relations', [])
                if relations:
                    # Find matching pattern in index
                    subgraph_id = index._create_subgraph_id([str(r) for r in relations])
                    
                    # Verify pattern exists in index
                    if subgraph_id in index.patterns:
                        training_samples.append({
                            'question': question,
                            'subgraph_id': subgraph_id,
                            'relations': [str(r) for r in relations]
                        })
    
    # Save training data
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path_obj.suffix == '.jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    elif output_path_obj.suffix == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_samples, f, ensure_ascii=False, indent=2)
    else:
        # Default to jsonl
        with open(output_path + '.jsonl', 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Prepared {len(training_samples)} training samples")
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare GSR training data')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--subgraph_index_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    args = parser.parse_args()
    
    prepare_gsr_training_data(
        data_path=args.data_path,
        subgraph_index_path=args.subgraph_index_path,
        output_path=args.output_path
    )

