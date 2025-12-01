"""
Complete pipeline to prepare all GSR data from scratch.

This script automates the entire GSR data preparation process:
1. Build subgraph index
2. Prepare GSR training data
3. Prepare reader training data (optional)
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsr.subgraph_index import build_subgraph_index_from_dataset
from gsr.prepare_data import prepare_gsr_training_data


def main():
    parser = argparse.ArgumentParser(
        description='Complete GSR data preparation pipeline'
    )
    
    # Input data
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training dataset (parquet or jsonl)')
    parser.add_argument('--val_data', type=str, default=None,
                       help='Path to validation dataset (optional)')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test dataset (optional)')
    
    # Output paths
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for all GSR data')
    parser.add_argument('--subgraph_index_name', type=str, default='subgraph_index.json',
                       help='Name for subgraph index file')
    parser.add_argument('--gsr_training_name', type=str, default='gsr_training_data.jsonl',
                       help='Name for GSR training data file')
    parser.add_argument('--reader_training_name', type=str, default='reader_training_data.jsonl',
                       help='Name for reader training data file')
    
    # Options
    parser.add_argument('--min_pattern_frequency', type=int, default=1,
                       help='Minimum pattern frequency to include in index')
    parser.add_argument('--prepare_reader_data', action='store_true',
                       help='Also prepare reader training data (requires GSR predictions)')
    parser.add_argument('--gsr_predictions_path', type=str, default=None,
                       help='Path to GSR predictions (required if --prepare_reader_data)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GSR DATA PREPARATION PIPELINE")
    print("="*60)
    
    # Step 1: Build subgraph index
    print("\n[Step 1/3] Building subgraph index...")
    subgraph_index_path = output_dir / args.subgraph_index_name
    
    index = build_subgraph_index_from_dataset(
        data_path=args.train_data,
        output_path=str(subgraph_index_path),
        min_pattern_frequency=args.min_pattern_frequency
    )
    
    print(f"[OK] Subgraph index saved to {subgraph_index_path}")
    
    # Step 2: Prepare GSR training data
    print("\n[Step 2/3] Preparing GSR training data...")
    gsr_training_path = output_dir / args.gsr_training_name
    
    prepare_gsr_training_data(
        data_path=args.train_data,
        subgraph_index_path=str(subgraph_index_path),
        output_path=str(gsr_training_path)
    )
    
    print(f"[OK] GSR training data saved to {gsr_training_path}")
    
    # Step 3: Prepare reader data (optional)
    if args.prepare_reader_data:
        if not args.gsr_predictions_path:
            print("\n⚠ Warning: --gsr_predictions_path required for reader data preparation")
            print("  Skipping reader data preparation")
        else:
            try:
                from gsr.reader_model import prepare_reader_data
                print("\n[Step 3/3] Preparing reader training data...")
                reader_training_path = output_dir / args.reader_training_name
                
                prepare_reader_data(
                    gsr_predictions_path=args.gsr_predictions_path,
                    subgraph_index_path=str(subgraph_index_path),
                    original_data_path=args.train_data,
                    output_path=str(reader_training_path)
                )
                
                print(f"[OK] Reader training data saved to {reader_training_path}")
            except (ImportError, OSError) as e:
                print(f"\n[WARNING] Could not prepare reader data: {e}")
                print("  (This requires torch/transformers to be working)")
    else:
        print("\n[Step 3/3] Skipping reader data preparation (use --prepare_reader_data to enable)")
    
    # Summary
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Subgraph index: {subgraph_index_path}")
    print(f"GSR training data: {gsr_training_path}")
    if args.prepare_reader_data and args.gsr_predictions_path:
        print(f"Reader training data: {output_dir / args.reader_training_name}")
    print("\nNext steps:")
    print("1. Train GSR model: python Core/gsr/train_gsr.py --train_data <gsr_training_path>")
    print("2. Run inference: python Core/gsr/inference_gsr.py --model_path <model_path> --subgraph_index_path <index_path>")


if __name__ == '__main__':
    main()

