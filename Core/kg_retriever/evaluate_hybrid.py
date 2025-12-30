"""
Evaluation script for Hybrid Path Ranking (Bi-Encoder + Diffusion).
"""

import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

from models.path_ranker import PathRankerModel
from models.diffusion_reranker import DiffusionPathScorer, HybridPathRanker
from data.hop_aware_dataset import HopAwarePathRankerDataModule, hop_aware_collate_fn as collate_fn

def evaluate_hybrid(
    bi_encoder_ckpt: str,
    diffusion_ckpt: str,
    data_path: str,
    output_dir: str = 'outputs_hybrid',
    batch_size: int = 16,
    gpu: int = 0,
    diffusion_weight: float = 0.3,
    limit: int = 0,
):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Hybrid Model
    print("Loading models...")
    model = HybridPathRanker(
        bi_encoder_checkpoint=bi_encoder_ckpt,
        diffusion_scorer_checkpoint=diffusion_ckpt,
        fusion_type='weighted',
        diffusion_weight=diffusion_weight,
    )
    model.to(device)
    model.eval()
    
    # Load data
    print(f"Loading data from {data_path}...")
    # Note: We use the base collate_fn or the hop_aware one. 
    # Since we are evaluating, we just need the standard inputs.
    from data.path_ranker_dataset import PathRankerDataset
    
    # We use the dataset class directly to get the collate_fn and settings
    dataset = PathRankerDataset(
        data_path=data_path,
        tokenizer_name=model.bi_encoder.hparams.encoder_name,
        max_question_length=128,
        max_path_length=64,
        max_candidates=100,
        training=False,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    total = 0
    hits_1 = 0
    hits_5 = 0
    mrr = 0.0
    
    # Hop-wise metrics
    correct_by_hop = {}
    total_by_hop = {}
    
    results = []
    
    with torch.no_grad():
        for i_batch, batch in enumerate(tqdm(dataloader)):
            if limit > 0 and i_batch >= limit:
                break
                
            # Move to device
            batch_inputs = {
                'question_input_ids': batch['question_input_ids'].to(device),
                'question_attention_mask': batch['question_attention_mask'].to(device),
                'path_input_ids': batch['path_input_ids'].to(device),
                'path_attention_mask': batch['path_attention_mask'].to(device),
                'candidate_mask': batch['candidate_mask'].to(device),
            }
            
            # Forward pass
            outputs = model(**batch_inputs)
            logits = outputs['logits']  # [B, num_candidates]
            
            # Calculate metrics
            probs = F.softmax(logits, dim=-1)
            sorted_scores, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            
            labels = batch['labels'].to(device)
            
            for i in range(len(labels)):
                gt_idx = labels[i].item()
                rank = (sorted_indices[i] == gt_idx).nonzero(as_tuple=True)[0].item() + 1
                
                # Global metrics
                mrr += 1.0 / rank
                if rank <= 1: hits_1 += 1
                if rank <= 5: hits_5 += 1
                total += 1
                
                # Hop-wise metrics
                # Get hop count of ground truth
                gt_path_text = batch['candidate_text'][i][gt_idx]
                # gt_path_text is a list of relations
                hop_count = len(gt_path_text)
                
                if hop_count not in total_by_hop:
                    total_by_hop[hop_count] = 0
                    correct_by_hop[hop_count] = 0
                
                total_by_hop[hop_count] += 1
                if rank <= 1:
                    correct_by_hop[hop_count] += 1
                
                # Save result
                results.append({
                    'id': batch['id'][i],
                    'question': batch['question_text'][i],
                    'rank': rank,
                    'hop_count': hop_count,
                    'bi_score': outputs['bi_scores'][i][gt_idx].item(),
                    'diff_score': outputs['diff_scores'][i][gt_idx].item() if outputs['diff_scores'] is not None else 0.0,
                    'fused_score': logits[i][gt_idx].item(),
                })
    
    # Compute summary metrics
    metrics = {
        'hits@1': hits_1 / total,
        'hits@5': hits_5 / total,
        'mrr': mrr / total,
        'total': total,
    }
    
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\n=== Performance by Hop Count ===")
    hop_metrics = []
    for hop in sorted(total_by_hop.keys()):
        acc = correct_by_hop[hop] / total_by_hop[hop]
        print(f"{hop}-hop: {acc:.4f} ({correct_by_hop[hop]}/{total_by_hop[hop]})")
        hop_metrics.append({'hop': hop, 'acc': acc, 'count': total_by_hop[hop]})
    
    # Save results
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        pd.DataFrame([metrics]).to_csv(f"{output_dir}/metrics.csv", index=False)
        pd.DataFrame(hop_metrics).to_csv(f"{output_dir}/hop_metrics.csv", index=False)
        print(f"Saved results to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bi_encoder_ckpt', type=str, required=True)
    parser.add_argument('--diffusion_ckpt', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs_hybrid')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--diffusion_weight', type=float, default=0.3)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    
    evaluate_hybrid(
        bi_encoder_ckpt=args.bi_encoder_ckpt,
        diffusion_ckpt=args.diffusion_ckpt,
        data_path=args.data,
        output_dir=args.output_dir,
        gpu=args.gpu,
        diffusion_weight=args.diffusion_weight,
        limit=args.limit,
    )
