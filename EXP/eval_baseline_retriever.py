"""
Baseline Path Retriever Evaluation Script.

This script evaluates the baseline KGPathRetriever on the WebQSP test set,
computing Hits@K metrics by comparing retrieved relation chains against 
ground truth paths.
"""

import os
import json
import logging
import argparse
import pandas as pd
from typing import List, Dict, Any, Set
from tqdm import tqdm
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from EXP.retriever import KGPathRetriever
from EXP.config import RAGConfig


def load_test_data(parquet_path: str) -> pd.DataFrame:
    """Load and preprocess test data."""
    logger.info(f"Loading data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Parse JSON columns if they are strings
    json_cols = ['q_entity', 'a_entity', 'graph', 'answer', 'paths']
    for col in json_cols:
        if col in df.columns:
            # Check if first element is string
            if len(df) > 0 and isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(lambda x: json.loads(x) if x else [])
    
    return df


def extract_relation_chains(paths: List[Dict]) -> Set[str]:
    """Extract unique relation chains from ground truth paths."""
    chains = set()
    for p in paths:
        if isinstance(p, dict):
            # relation_chain is a string like "location.country.official_language"
            rc = p.get('relation_chain', '')
            if rc:
                chains.add(rc)
    return chains


def compute_hits(retrieved_chains: List[str], ground_truth_chains: Set[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, int]:
    """
    Compute Hits@K for a single sample.
    
    A hit occurs if any of the top-K retrieved chains matches a ground truth chain.
    """
    hits = {}
    for k in k_values:
        top_k_chains = set(retrieved_chains[:k])
        # Check if any retrieved chain matches ground truth
        hit = 1 if top_k_chains & ground_truth_chains else 0
        hits[f"hits@{k}"] = hit
    return hits


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline path retriever")
    parser.add_argument("--test_file", type=str, default="Data/webqsp_final/test.parquet")
    parser.add_argument("--index_path", type=str, default="EXP/index")
    parser.add_argument("--output_file", type=str, default="EXP/baseline_retriever_results.json")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--top_k", type=int, default=10, help="Number of paths to retrieve")
    args = parser.parse_args()
    
    # Load Data
    df = load_test_data(args.test_file)
    if args.num_samples:
        df = df.head(args.num_samples)
    
    logger.info(f"Loaded {len(df)} samples")
    
    # Initialize Retriever
    config = RAGConfig(
        use_reranker=False,  # Baseline without reranker
        index_dir=args.index_path,
    )
    retriever = KGPathRetriever(
        index_path=args.index_path,
        config=config
    )
    
    # Get retriever stats
    stats = retriever.get_stats()
    logger.info(f"Retriever stats: {stats}")
    
    # Evaluation
    results = []
    hits_accumulator = {f"hits@{k}": 0 for k in [1, 3, 5, 10]}
    
    logger.info("Starting evaluation...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            question = row['question']
            gt_paths = row['paths']  # Ground truth paths
            
            # Extract ground truth relation chains
            gt_chains = extract_relation_chains(gt_paths)
            
            # Retrieve paths using baseline retriever
            retrieved = retriever.retrieve(question, top_k=args.top_k)
            retrieved_chains = [p.relation_chain for p in retrieved]
            
            # Compute hits
            hits = compute_hits(retrieved_chains, gt_chains)
            
            # Accumulate
            for key, val in hits.items():
                hits_accumulator[key] += val
            
            # Store result
            result = {
                "id": row.get('id', idx),
                "question": question,
                "ground_truth_chains": list(gt_chains),
                "retrieved_chains": retrieved_chains,
                "retrieved_scores": [p.score for p in retrieved],
                "hits": hits
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute final metrics
    n = len(results)
    if n > 0:
        metrics = {key: val / n for key, val in hits_accumulator.items()}
    else:
        metrics = hits_accumulator
    
    # Print results
    print("\n" + "=" * 50)
    print(f"Baseline Retriever Evaluation Results (N={n})")
    print("=" * 50)
    for key, val in sorted(metrics.items()):
        print(f"{key}: {val:.4f}")
    print("=" * 50)
    
    # Save results
    output = {
        "metrics": metrics,
        "config": {
            "embedding_model": config.embedding_model,
            "top_k": args.top_k,
            "num_samples": n
        },
        "results": results
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
