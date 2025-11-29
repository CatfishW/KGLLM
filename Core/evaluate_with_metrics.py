"""
Evaluation script with metrics: Hits@K, Recall, and example outputs.
"""

import os
import sys
import argparse
import torch
import json
from tqdm import tqdm
from typing import List, Dict, Any
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import KGPathDataset, EntityRelationVocab, collate_kg_batch
from kg_path_diffusion import KGPathDiffusionLightning
from torch.utils.data import DataLoader


def calculate_hits_at_k(predicted_relations_list: List[List[str]], ground_truth_relations: List[List[str]], k: int = 1) -> float:
    """
    Calculate Hits@K metric.
    
    Args:
        predicted_relations_list: List of lists of predicted relation chains (multiple predictions per sample)
        ground_truth_relations: List of lists of ground truth relation chains
        k: Number of top predictions to consider
    
    Returns:
        Hits@K score (0.0 to 1.0)
    """
    hits = 0
    total = 0
    
    for pred_list, gt_list in zip(predicted_relations_list, ground_truth_relations):
        total += 1
        if not gt_list:
            continue
        
        # Check top k predictions
        found_match = False
        for pred in pred_list[:k]:
            if not pred:
                continue
            pred_relations = set(pred.split(" -> ")) if isinstance(pred, str) else set(pred)
            
            for gt in gt_list:
                gt_relations = set(gt.split(" -> ")) if isinstance(gt, str) else set(gt)
                # Exact match
                if pred_relations == gt_relations:
                    hits += 1
                    found_match = True
                    break
                # Partial match (subset or significant overlap)
                elif pred_relations and gt_relations:
                    overlap = len(pred_relations & gt_relations) / len(gt_relations) if gt_relations else 0
                    if overlap >= 0.8:  # 80% overlap threshold
                        hits += 1
                        found_match = True
                        break
            
            if found_match:
                break
    
    return hits / total if total > 0 else 0.0


def calculate_recall(predicted_relations: List[str], ground_truth_relations: List[List[str]]) -> Dict[str, float]:
    """
    Calculate recall metrics.
    
    Returns:
        Dictionary with 'exact', 'partial', 'relation_overlap' scores
    """
    exact_matches = 0
    partial_matches = 0
    relation_overlaps = 0
    total = 0
    
    for pred, gt_list in zip(predicted_relations, ground_truth_relations):
        total += 1
        pred_relations = set(pred.split(" -> ")) if pred else set()
        
        best_match = False
        best_overlap = 0.0
        
        for gt in gt_list:
            gt_relations = set(gt.split(" -> ")) if isinstance(gt, str) else set(gt)
            
            # Exact match
            if pred_relations == gt_relations:
                exact_matches += 1
                best_match = True
                best_overlap = 1.0
                break
            
            # Calculate overlap
            if pred_relations and gt_relations:
                overlap = len(pred_relations & gt_relations) / len(gt_relations)
                best_overlap = max(best_overlap, overlap)
                
                # Partial match (at least 50% overlap)
                if overlap >= 0.5 and not best_match:
                    partial_matches += 1
                    best_match = True
        
        if best_overlap > 0:
            relation_overlaps += best_overlap
    
    return {
        'exact_recall': exact_matches / total if total > 0 else 0.0,
        'partial_recall': partial_matches / total if total > 0 else 0.0,
        'avg_relation_overlap': relation_overlaps / total if total > 0 else 0.0
    }


def evaluate_model(
    checkpoint_path: str,
    vocab_path: str,
    test_data_path: str,
    device: str = 'cuda',
    batch_size: int = 8,
    num_samples: int = 5,
    max_examples: int = 100
) -> Dict[str, Any]:
    """Evaluate model and compute metrics."""
    
    print("=" * 70)
    print("Loading model and vocabulary...")
    print("=" * 70)
    
    # Load vocabulary
    vocab = EntityRelationVocab.load(vocab_path)
    print(f"Vocabulary: {vocab.num_entities} entities, {vocab.num_relations} relations")
    
    # Load model with strict=False to handle architecture changes
    try:
        model = KGPathDiffusionLightning.load_from_checkpoint(
            checkpoint_path,
            num_entities=vocab.num_entities,
            num_relations=vocab.num_relations,
            map_location=device,
            strict=False  # Allow loading checkpoints from old architecture
        )
    except Exception as e:
        print(f"Warning: Error loading checkpoint with strict=False: {e}")
        print("Attempting to load with hparams from checkpoint...")
        # Try loading without specifying num_entities/num_relations
        model = KGPathDiffusionLightning.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            strict=False
        )
    model.eval()
    model.to(device)
    print(f"Model loaded from {checkpoint_path}")
    
    # Create dataset
    print(f"\nLoading test data from {test_data_path}...")
    tokenizer_name = model.hparams.get('question_encoder', 'sentence-transformers/all-MiniLM-L6-v2')
    
    dataset = KGPathDataset(
        test_data_path,
        vocab=vocab,
        build_vocab=False,
        max_path_length=100,
        training=False,
        tokenizer_name=tokenizer_name
    )
    
    # Limit to max_examples for faster evaluation
    if max_examples and len(dataset) > max_examples:
        dataset.samples = dataset.samples[:max_examples]
        print(f"Limited to {max_examples} samples for evaluation")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_kg_batch
    )
    
    print(f"\nEvaluating on {len(dataset)} samples...")
    print("=" * 70)
    
    # Collect predictions and ground truth
    all_predictions = []  # Primary prediction (first)
    all_predictions_list = []  # All predictions for Hits@K
    all_ground_truth = []
    all_questions = []
    all_answers = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch['question_input_ids'] = batch['question_input_ids'].to(device)
            batch['question_attention_mask'] = batch['question_attention_mask'].to(device)
            
            # Generate multiple paths per question
            # Use split_long_paths to get multiple shorter paths instead of one long path
            all_entities, all_relations = model.model.generate_multiple(
                batch['question_input_ids'],
                batch['question_attention_mask'],
                num_paths=num_samples,
                path_length=20,
                temperature=1.0,
                split_long_paths=True,
                max_path_length=5  # Split into paths of max 5 relations
            )
            
            # Decode predictions
            for i in range(batch['question_input_ids'].shape[0]):
                sample_id = batch['ids'][i]
                
                # Get all generated relation chains for this sample
                pred_relations_list = []
                for path_idx in range(num_samples):
                    relations = all_relations[i, path_idx]
                    
                    # Split long path into multiple shorter paths
                    # Find PAD/MASK boundaries to split
                    rel_indices = relations.tolist()
                    current_path = []
                    
                    for rel_idx in rel_indices:
                        rel_name = vocab.idx2relation.get(rel_idx, "<UNK>")
                        if rel_name in ["<PAD>", "<MASK>", "<UNK>"]:
                            # End of current path, start new one
                            if current_path:
                                pred_relations_list.append(" -> ".join(current_path))
                                current_path = []
                        else:
                            current_path.append(rel_name)
                    
                    # Add final path if any
                    if current_path:
                        pred_relations_list.append(" -> ".join(current_path))
                
                # Remove duplicates while preserving order
                seen = set()
                unique_pred_relations_list = []
                for path in pred_relations_list:
                    if path and path not in seen:
                        seen.add(path)
                        unique_pred_relations_list.append(path)
                pred_relations_list = unique_pred_relations_list
                
                # Use all predictions as a list (matching ground truth format)
                # Primary is the first path, but we keep all paths
                primary_pred = pred_relations_list[0] if pred_relations_list else ""
                
                # Get ground truth from dataset
                try:
                    sample = next(s for s in dataset.samples if s.get('id') == sample_id)
                except StopIteration:
                    # Fallback: use index
                    sample_idx = batch_idx * batch_size + i
                    if sample_idx < len(dataset.samples):
                        sample = dataset.samples[sample_idx]
                    else:
                        sample = {}
                
                gt_paths = sample.get('paths', [])
                gt_relations_list = []
                for path in gt_paths:
                    if isinstance(path, dict):
                        rel_chain = path.get('relation_chain', '')
                        if rel_chain:
                            gt_relations_list.append(rel_chain)
                    elif isinstance(path, str):
                        # Handle string format
                        gt_relations_list.append(path)
                
                # Store primary prediction (first path) and all paths as list
                all_predictions.append(primary_pred)
                all_predictions_list.append(pred_relations_list)  # This is already a list of paths
                all_ground_truth.append(gt_relations_list)
                all_questions.append(sample.get('question', ''))
                all_answers.append(sample.get('answer', []))
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("CALCULATING METRICS")
    print("=" * 70)
    
    hits_at_1 = calculate_hits_at_k(all_predictions_list, all_ground_truth, k=1)
    hits_at_5 = calculate_hits_at_k(all_predictions_list, all_ground_truth, k=5)
    recall_metrics = calculate_recall(all_predictions, all_ground_truth)
    
    metrics = {
        'hits_at_1': hits_at_1,
        'hits_at_5': hits_at_5,
        **recall_metrics,
        'total_samples': len(all_predictions),
        'samples_with_gt': sum(1 for gt in all_ground_truth if gt)
    }
    
    # Print metrics
    print(f"\nHits@1: {hits_at_1:.4f}")
    print(f"Hits@5: {hits_at_5:.4f}")
    print(f"Exact Recall: {recall_metrics['exact_recall']:.4f}")
    print(f"Partial Recall: {recall_metrics['partial_recall']:.4f}")
    print(f"Avg Relation Overlap: {recall_metrics['avg_relation_overlap']:.4f}")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Samples with Ground Truth: {metrics['samples_with_gt']}")
    
    # Show example outputs
    print("\n" + "=" * 70)
    print("EXAMPLE OUTPUTS")
    print("=" * 70)
    
    num_examples = min(10, len(all_predictions))
    for i in range(num_examples):
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {all_questions[i]}")
        print(f"Answer: {all_answers[i]}")
        
        # Show all predicted paths (list format, matching ground truth)
        if all_predictions_list[i]:
            print(f"Predicted ({len(all_predictions_list[i])} paths):")
            for j, pred_path in enumerate(all_predictions_list[i][:5]):  # Show first 5 predicted paths
                print(f"  [{j+1}] {pred_path}")
        else:
            print("Predicted: (empty)")
        
        if all_ground_truth[i]:
            print(f"Ground Truth ({len(all_ground_truth[i])} paths):")
            for j, gt in enumerate(all_ground_truth[i][:5]):  # Show first 5 GT paths
                print(f"  [{j+1}] {gt}")
        else:
            print("Ground Truth: (no paths available)")
        
        # Check if any predicted path matches any ground truth path
        if all_ground_truth[i] and all_predictions_list[i]:
            matched = False
            for pred_path in all_predictions_list[i]:
                pred_set = set(pred_path.split(" -> ")) if pred_path else set()
                for gt in all_ground_truth[i]:
                    gt_set = set(gt.split(" -> ")) if isinstance(gt, str) else set(gt)
                    if pred_set == gt_set:
                        print("  [MATCH] EXACT MATCH")
                        matched = True
                        break
                    elif pred_set and gt_set and pred_set.issubset(gt_set):
                        print("  [MATCH] PARTIAL MATCH")
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                print("  [NO MATCH]")
    
    return {
        'metrics': metrics,
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'questions': all_questions,
        'answers': all_answers
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with metrics')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of paths to generate per question')
    parser.add_argument('--max_examples', type=int, default=100,
                        help='Maximum number of examples to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        test_data_path=args.test_data,
        device=args.device,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        max_examples=args.max_examples
    )
    
    # Save results
    output_data = {
        'metrics': results['metrics'],
        'examples': [
            {
                'question': q,
                'answer': a,
                'predicted': p,
                'ground_truth': gt
            }
            for q, a, p, gt in zip(
                results['questions'],
                results['answers'],
                results['predictions'],
                results['ground_truth']
            )
        ]
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

