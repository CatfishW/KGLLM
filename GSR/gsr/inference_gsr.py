"""
Inference script for GSR Subgraph ID Generator.

Generates subgraph IDs for questions and retrieves corresponding subgraphs.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsr.subgraph_id_generator import SubgraphIDGenerator
from gsr.subgraph_index import SubgraphIndex


def generate_subgraph_ids(
    model_path: str,
    questions: List[str],
    subgraph_index_path: str,
    num_beams: int = 5,
    top_k: int = 3,
    output_path: str = None
) -> List[Dict]:
    """
    Generate subgraph IDs for questions and retrieve subgraphs.
    
    Args:
        model_path: Path to trained GSR model
        questions: List of question strings
        subgraph_index_path: Path to subgraph index
        num_beams: Number of beams for generation
        top_k: Number of top predictions per question
        output_path: Optional path to save results
    
    Returns:
        List of results with predicted subgraph IDs and retrieved subgraphs
    """
    # Load model
    print(f"Loading model from {model_path}...")
    generator = SubgraphIDGenerator.from_pretrained(model_path)
    generator.model.eval()
    
    if torch.cuda.is_available():
        generator.model = generator.model.cuda()
    
    # Load subgraph index
    print(f"Loading subgraph index from {subgraph_index_path}...")
    index = SubgraphIndex.load(subgraph_index_path)
    
    # Generate subgraph IDs
    print(f"Generating subgraph IDs for {len(questions)} questions...")
    all_predictions = generator.generate(
        questions=questions,
        num_beams=num_beams,
        top_k=top_k
    )
    
    # Retrieve subgraphs and format results
    results = []
    for i, (question, predictions) in enumerate(zip(questions, all_predictions)):
        result = {
            'question': question,
            'predictions': []
        }
        
        for subgraph_id in predictions:
            pattern = index.get_pattern(subgraph_id)
            
            if pattern:
                result['predictions'].append({
                    'subgraph_id': subgraph_id,
                    'relations': pattern.relations,
                    'relation_pattern': pattern.relation_pattern,
                    'example_count': pattern.example_count,
                    'example_triples': pattern.example_triples[:3]  # Top 3 examples
                })
            else:
                # Pattern not found in index (might be a new pattern)
                result['predictions'].append({
                    'subgraph_id': subgraph_id,
                    'relations': [],
                    'relation_pattern': subgraph_id,
                    'example_count': 0,
                    'example_triples': [],
                    'note': 'Pattern not found in index'
                })
        
        results.append(result)
    
    # Save results if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path_obj.suffix == '.jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Saved results to {output_path}")
    
    return results


def evaluate_subgraph_ids(
    results: List[Dict],
    ground_truth_path: str = None,
    subgraph_index_path: str = None
) -> Dict:
    """
    Evaluate subgraph ID predictions against ground truth.
    
    Args:
        results: List of prediction results with 'id' and 'predictions'
        ground_truth_path: Path to ground truth data with paths
        subgraph_index_path: Path to subgraph index (to create IDs from ground truth)
    
    Returns:
        Dictionary with evaluation metrics
    """
    from pathlib import Path
    import pandas as pd
    from gsr.subgraph_index import SubgraphIndex
    
    metrics = {
        'total_questions': len(results),
        'avg_predictions_per_question': sum(len(r['predictions']) for r in results) / len(results) if results else 0
    }
    
    if not ground_truth_path or not subgraph_index_path:
        return metrics
    
    # Load subgraph index
    index = SubgraphIndex.load(subgraph_index_path)
    
    # Load ground truth
    gt_path = Path(ground_truth_path)
    ground_truth = {}
    
    if gt_path.suffix == '.parquet':
        df = pd.read_parquet(ground_truth_path)
        for _, row in df.iterrows():
            sample_id = row.get('id', '')
            paths = row.get('paths', [])
            if isinstance(paths, str):
                try:
                    paths = json.loads(paths)
                except:
                    paths = []
            ground_truth[sample_id] = paths
    elif gt_path.suffix in ['.jsonl', '.json']:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            if gt_path.suffix == '.jsonl':
                for line in f:
                    data = json.loads(line)
                    sample_id = data.get('id', '')
                    ground_truth[sample_id] = data.get('paths', [])
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        sample_id = item.get('id', '')
                        ground_truth[sample_id] = item.get('paths', [])
                else:
                    ground_truth[data.get('id', '')] = data.get('paths', [])
    
    # Extract ground truth subgraph IDs
    gt_subgraph_ids = {}
    for sample_id, paths in ground_truth.items():
        gt_ids = set()
        for path in paths:
            if isinstance(path, dict):
                relations = path.get('relations', [])
                if relations:
                    subgraph_id = index._create_subgraph_id([str(r) for r in relations])
                    gt_ids.add(subgraph_id)
        gt_subgraph_ids[sample_id] = gt_ids
    
    # Calculate metrics
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    top10_hits = 0
    total_with_gt = 0
    pattern_matches = {}
    
    for result in results:
        sample_id = result.get('id', '')
        if sample_id not in gt_subgraph_ids:
            continue
        
        gt_ids = gt_subgraph_ids[sample_id]
        if not gt_ids:
            continue
        
        total_with_gt += 1
        predictions = result.get('predictions', [])
        pred_ids = [p.get('subgraph_id', '') for p in predictions]
        
        # Top-1 recall
        if pred_ids and pred_ids[0] in gt_ids:
            top1_hits += 1
        
        # Top-k recall
        if any(pid in gt_ids for pid in pred_ids[:3]):
            top3_hits += 1
        if any(pid in gt_ids for pid in pred_ids[:5]):
            top5_hits += 1
        if any(pid in gt_ids for pid in pred_ids[:10]):
            top10_hits += 1
        
        # Pattern-level matching
        for gt_id in gt_ids:
            if gt_id not in pattern_matches:
                pattern_matches[gt_id] = {'total': 0, 'hits': 0}
            pattern_matches[gt_id]['total'] += 1
            if any(pid == gt_id for pid in pred_ids[:5]):
                pattern_matches[gt_id]['hits'] += 1
    
    # Calculate recall metrics
    if total_with_gt > 0:
        metrics.update({
            'total_with_ground_truth': total_with_gt,
            'top1_recall': top1_hits / total_with_gt,
            'top3_recall': top3_hits / total_with_gt,
            'top5_recall': top5_hits / total_with_gt,
            'top10_recall': top10_hits / total_with_gt,
            'top1_hits': top1_hits,
            'top3_hits': top3_hits,
            'top5_hits': top5_hits,
            'top10_hits': top10_hits
        })
        
        # Pattern-level metrics
        pattern_recalls = [pm['hits'] / pm['total'] for pm in pattern_matches.values() if pm['total'] > 0]
        if pattern_recalls:
            metrics.update({
                'avg_pattern_recall': sum(pattern_recalls) / len(pattern_recalls),
                'unique_patterns_in_gt': len(pattern_matches),
                'patterns_with_hits': sum(1 for pm in pattern_matches.values() if pm['hits'] > 0)
            })
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='GSR Inference: Generate Subgraph IDs')
    
    # Input/Output
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained GSR model')
    parser.add_argument('--subgraph_index_path', type=str, required=True,
                       help='Path to subgraph index')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input questions (jsonl or json)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save predictions')
    
    # Generation parameters
    parser.add_argument('--num_beams', type=int, default=5,
                       help='Number of beams for beam search')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions per question')
    
    # Evaluation
    parser.add_argument('--ground_truth_path', type=str, default=None,
                       help='Path to ground truth for evaluation')
    
    args = parser.parse_args()
    
    # Load questions
    input_path = Path(args.input_path)
    questions = []
    question_ids = []
    
    if input_path.suffix == '.jsonl':
        with open(args.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                questions.append(data.get('question', ''))
                question_ids.append(data.get('id', ''))
    elif input_path.suffix == '.json':
        with open(args.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    questions.append(item.get('question', ''))
                    question_ids.append(item.get('id', ''))
            else:
                questions.append(data.get('question', ''))
                question_ids.append(data.get('id', ''))
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    print(f"Loaded {len(questions)} questions")
    
    # Generate subgraph IDs
    results = generate_subgraph_ids(
        model_path=args.model_path,
        questions=questions,
        subgraph_index_path=args.subgraph_index_path,
        num_beams=args.num_beams,
        top_k=args.top_k,
        output_path=args.output_path
    )
    
    # Add question IDs to results
    for i, result in enumerate(results):
        if i < len(question_ids) and question_ids[i]:
            result['id'] = question_ids[i]
    
    # Evaluate if ground truth provided
    if args.ground_truth_path:
        metrics = evaluate_subgraph_ids(
            results, 
            args.ground_truth_path,
            args.subgraph_index_path
        )
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Total questions: {metrics.get('total_questions', 0)}")
        print(f"Questions with ground truth: {metrics.get('total_with_ground_truth', 0)}")
        print(f"\nRecall Metrics:")
        if 'top1_recall' in metrics:
            print(f"  Top-1 Recall (Hit Rate): {metrics['top1_recall']:.4f} ({metrics.get('top1_hits', 0)}/{metrics.get('total_with_ground_truth', 0)})")
            print(f"  Top-3 Recall: {metrics['top3_recall']:.4f} ({metrics.get('top3_hits', 0)}/{metrics.get('total_with_ground_truth', 0)})")
            print(f"  Top-5 Recall: {metrics['top5_recall']:.4f} ({metrics.get('top5_hits', 0)}/{metrics.get('total_with_ground_truth', 0)})")
            print(f"  Top-10 Recall: {metrics['top10_recall']:.4f} ({metrics.get('top10_hits', 0)}/{metrics.get('total_with_ground_truth', 0)})")
        if 'avg_pattern_recall' in metrics:
            print(f"\nPattern-Level Metrics:")
            print(f"  Average Pattern Recall: {metrics['avg_pattern_recall']:.4f}")
            print(f"  Unique Patterns in GT: {metrics.get('unique_patterns_in_gt', 0)}")
            print(f"  Patterns with Hits: {metrics.get('patterns_with_hits', 0)}")
        print(f"\nAverage predictions per question: {metrics.get('avg_predictions_per_question', 0):.2f}")
    
    # Print sample results
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    for i, result in enumerate(results[:3]):
        print(f"\nQuestion {i+1}: {result['question']}")
        for j, pred in enumerate(result['predictions'][:2]):
            print(f"  Prediction {j+1}: {pred['subgraph_id']}")
            print(f"    Relations: {pred['relation_pattern']}")
            print(f"    Examples: {pred['example_count']}")


if __name__ == '__main__':
    main()

