"""
Evaluate multiple checkpoints with entity retrieval.

This script evaluates different model checkpoints and compares their performance.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_with_entity_retrieval import evaluate_with_entity_retrieval


def evaluate_checkpoint(
    checkpoint_name: str,
    checkpoint_path: str,
    vocab_path: str,
    test_data_path: str,
    eval_results_path: str = None,
    max_examples: int = None
) -> Dict[str, Any]:
    """
    Evaluate a single checkpoint.
    
    If eval_results_path is provided, use pre-computed predictions.
    Otherwise, run inference first (requires GPU).
    """
    print("\n" + "=" * 80)
    print(f"EVALUATING: {checkpoint_name}")
    print("=" * 80)
    
    if eval_results_path and os.path.exists(eval_results_path):
        # Use pre-computed evaluation results
        results = evaluate_with_entity_retrieval(
            eval_results_path=eval_results_path,
            test_data_path=test_data_path,
            vocab_path=vocab_path,
            max_examples=max_examples
        )
    else:
        print(f"Warning: No pre-computed results found at {eval_results_path}")
        print("Please run evaluate_with_metrics.py first to generate predictions.")
        return None
    
    return results


def compare_checkpoints(results: Dict[str, Dict]) -> None:
    """Print comparison table of checkpoint results."""
    print("\n" + "=" * 100)
    print("CHECKPOINT COMPARISON")
    print("=" * 100)
    
    # Header
    print(f"\n{'Checkpoint':<50} {'Rel H@1':>10} {'Rel H@5':>10} {'Ent H@1':>10} {'Ent H@5':>10} {'Partial':>10}")
    print("-" * 100)
    
    for name, result in results.items():
        if result is None:
            print(f"{name:<50} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue
            
        metrics = result.get('metrics', {})
        rel_h1 = metrics.get('relation_hits_at_1', 0) * 100
        rel_h5 = metrics.get('relation_hits_at_5', 0) * 100
        ent_h1 = metrics.get('entity_hits_at_1', 0) * 100
        ent_h5 = metrics.get('entity_hits_at_5', 0) * 100
        partial = metrics.get('entity_partial_match', 0) * 100
        
        print(f"{name:<50} {rel_h1:>9.2f}% {rel_h5:>9.2f}% {ent_h1:>9.2f}% {ent_h5:>9.2f}% {partial:>9.2f}%")
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Evaluate multiple checkpoints')
    parser.add_argument('--test_data', type=str,
                       default='../Data/webqsp_final/test.parquet',
                       help='Path to test data parquet')
    parser.add_argument('--max_examples', type=int, default=200,
                       help='Maximum examples to evaluate per checkpoint')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Define checkpoints to evaluate
    checkpoints = [
        {
            'name': 'outputs_multipath_relation_only_5',
            'checkpoint': 'outputs_multipath_relation_only_5/checkpoints/last.ckpt',
            'vocab': 'outputs_multipath_relation_only_5/vocab.json',
            'eval_results': None  # Will need to generate
        },
        {
            'name': 'outputs_multipath_relation_only_6',
            'checkpoint': 'outputs_multipath_relation_only_6/checkpoints/last.ckpt',
            'vocab': 'outputs_multipath_relation_only_6/vocab.json',
            'eval_results': None
        },
        {
            'name': 'outputs_multipath_relation_only_7',
            'checkpoint': 'outputs_multipath_relation_only_7/checkpoints/last.ckpt',
            'vocab': 'outputs_multipath_relation_only_7/vocab.json',
            'eval_results': None
        },
    ]
    
    # For now, we'll use the existing evaluation results from checkpoint 8
    # and run the evaluation with entity retrieval
    print("Note: This script requires pre-computed evaluation results.")
    print("Use evaluate_with_metrics.py to generate predictions first.")
    print("\nWe will evaluate using the existing webqsp_final_evaluation.json")
    
    # Evaluate using existing results
    results = {}
    
    # Use the existing evaluation results
    if os.path.exists('webqsp_final_evaluation.json'):
        print("\n\nEvaluating with existing predictions from webqsp_final_evaluation.json...")
        result = evaluate_with_entity_retrieval(
            eval_results_path='webqsp_final_evaluation.json',
            test_data_path=args.test_data,
            vocab_path='outputs_multipath_relation_only_8_freeze_question_encoder/vocab.json',
            max_examples=args.max_examples
        )
        results['outputs_multipath_relation_only_8_freeze'] = result
        
        # Save results
        output_path = os.path.join(args.output_dir, 'checkpoint_8_freeze_eval.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
    
    # Compare results
    if results:
        compare_checkpoints(results)


if __name__ == '__main__':
    main()
