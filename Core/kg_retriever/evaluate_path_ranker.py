"""
Evaluation script for Path Ranker Model.
"""

import argparse
import json
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List, Dict

from models.path_ranker import PathRankerModel
from data.path_ranker_dataset import PathRankerDataset, collate_fn


def evaluate(
    checkpoint_path: str,
    data_path: str,
    batch_size: int = 16,
    gpu: int = 0,
    output_dir: str = 'outputs',
):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {checkpoint_path}...")
    
    # Load model
    model = PathRankerModel.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    
    # Load data
    print(f"Loading data from {data_path}...")
    dataset = PathRankerDataset(
        data_path=data_path,
        tokenizer_name=model.hparams.encoder_name,
        max_question_length=model.hparams.max_question_length if hasattr(model.hparams, 'max_question_length') else 128,
        max_path_length=model.hparams.max_path_length if hasattr(model.hparams, 'max_path_length') else 64,
        max_candidates=100,  # Eval with up to 100 candidates
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
    hits_3 = 0
    hits_5 = 0
    hits_10 = 0
    mrr = 0.0
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                question_input_ids=batch['question_input_ids'],
                question_attention_mask=batch['question_attention_mask'],
                path_input_ids=batch['path_input_ids'],
                path_attention_mask=batch['path_attention_mask'],
                candidate_mask=batch['candidate_mask'],
            )
            
            logits = outputs['logits']  # [B, num_candidates]
            labels = batch['labels']    # [B]
            
            # Calculate metrics
            probs = F.softmax(logits, dim=-1)
            
            # Get rankings
            # Sort descending
            sorted_scores, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            
            for i in range(len(labels)):
                gt_idx = labels[i].item()
                # Find rank of gt_idx in sorted_indices
                # Note: sorted_indices[i] contains candidate indices sorted by score
                # We need to find where gt_idx appears in sorted_indices[i]
                
                rank = (sorted_indices[i] == gt_idx).nonzero(as_tuple=True)[0].item() + 1
                
                mrr += 1.0 / rank
                if rank <= 1: hits_1 += 1
                if rank <= 3: hits_3 += 1
                if rank <= 5: hits_5 += 1
                if rank <= 10: hits_10 += 1
                total += 1
                
                # Save result
                result_entry = {
                    'id': batch['id'][i],
                    'question': batch['question_text'][i],
                    'rank': rank,
                    'gt_idx': gt_idx,
                    'gt_path': batch['candidate_text'][i][gt_idx],
                    'top_pred_idx': sorted_indices[i][0].item(),
                    'top_pred_path': batch['candidate_text'][i][sorted_indices[i][0].item()],
                    'top_pred_score': probs[i][sorted_indices[i][0]].item(),
                    'top_5_preds': [
                        {
                            'path': batch['candidate_text'][i][idx.item()],
                            'score': probs[i][idx].item()
                        }
                        for idx in sorted_indices[i][:5]
                    ]
                }
                
                results.append(result_entry)

    # Summary
    metrics = {
        'hits@1': hits_1 / total,
        'hits@3': hits_3 / total,
        'hits@5': hits_5 / total,
        'hits@10': hits_10 / total,
        'mrr': mrr / total,
        'total': total,
    }
    
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save results
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        # Save as pretty-printed JSON
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        pd.DataFrame([metrics]).to_csv(f"{output_dir}/metrics.csv", index=False)
        print(f"Saved results to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs_eval')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.data, output_dir=args.output_dir, gpu=args.gpu)
