"""
Evaluate trained KG Retriever models.
"""
import os
import sys
import json
import argparse
import torch
import pytorch_lightning as pl
from pathlib import Path

from models.diffusion_retriever import KGDiffusionRetriever
from models.gnn_retriever import GNNRetriever
from data.dataset import KGRetrieverDataset, KGRetrieverDataModule, collate_fn
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate KG Retriever')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='diffusion',
                        choices=['diffusion', 'gnn'],
                        help='Model type')
    parser.add_argument('--test_data', type=str, nargs='+', required=True,
                        help='Path(s) to test data')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Path to vocabulary JSON')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load vocabulary
    with open(args.vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    num_relations = vocab_data['num_relations']
    relation_to_idx = vocab_data['relation_to_idx']
    
    print(f"Loaded vocabulary with {num_relations} relations")
    
    # Create test dataset with training vocabulary
    test_dataset = KGRetrieverDataset(
        data_path=args.test_data,
        vocab_path=None,  # Don't build from test
        max_triples=0,
        max_path_length=8,
    )
    
    # Override with training vocabulary
    test_dataset.relation_to_idx = relation_to_idx
    
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    # Load model
    print(f"Loading {args.model_type} model from {args.checkpoint}")
    
    if args.model_type == 'diffusion':
        model = KGDiffusionRetriever.load_from_checkpoint(
            args.checkpoint,
            num_relations=num_relations,
            strict=False,
        )
    else:
        model = GNNRetriever.load_from_checkpoint(
            args.checkpoint,
            num_relations=num_relations,
            strict=False,
        )
    
    model.eval()
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[args.gpu],
        precision='16-mixed',
        logger=False,
    )
    
    # Run test
    print("\nRunning evaluation...")
    results = trainer.test(model, dataloaders=test_loader)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in results[0].items():
        print(f"  {key}: {value:.4f}")
    print("="*60)
    
    # Generate some example paths
    print("\n" + "="*60)
    print("EXAMPLE GENERATIONS")
    print("="*60)
    
    device = torch.device(f'cuda:{args.gpu}')
    model = model.to(device)
    
    # Get a batch
    batch = next(iter(test_loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Generate paths
    if args.model_type == 'diffusion':
        generated = model.generate(
            question_input_ids=batch['question_input_ids'][:2],
            question_attention_mask=batch['question_attention_mask'][:2],
            kg_relation_ids=batch['kg_relation_ids'][:2],
            kg_head_hash_ids=batch['kg_head_hash_ids'][:2],
            kg_tail_hash_ids=batch['kg_tail_hash_ids'][:2],
            kg_triple_mask=batch['kg_triple_mask'][:2],
            path_length=4,
            num_samples=3,
        )
    else:
        generated = model.generate(
            question_input_ids=batch['question_input_ids'][:2],
            question_attention_mask=batch['question_attention_mask'][:2],
            kg_relation_ids=batch['kg_relation_ids'][:2],
            kg_head_hash_ids=batch['kg_head_hash_ids'][:2],
            kg_tail_hash_ids=batch['kg_tail_hash_ids'][:2],
            kg_triple_mask=batch['kg_triple_mask'][:2],
            max_length=4,
        ).unsqueeze(1)  # Add sample dim
    
    # Decode relations
    idx_to_relation = {v: k for k, v in relation_to_idx.items()}
    
    for i in range(min(2, len(batch['id']))):
        print(f"\nSample {i}: ID={batch['id'][i]}")
        
        # Ground truth
        gt = batch['target_relations'][i].cpu().tolist()
        gt_rels = [idx_to_relation.get(r, f'<{r}>') for r in gt if r > 2]
        print(f"  GT Path: {' -> '.join(gt_rels[:4])}")
        
        # Generated
        if args.model_type == 'diffusion':
            for j in range(min(3, generated.size(1))):
                gen = generated[i, j].cpu().tolist()
                gen_rels = [idx_to_relation.get(r, f'<{r}>') for r in gen if r > 2]
                print(f"  Gen {j}: {' -> '.join(gen_rels[:4])}")
        else:
            gen = generated[i, 0].cpu().tolist()
            gen_rels = [idx_to_relation.get(r, f'<{r}>') for r in gen if r > 2]
            print(f"  Gen: {' -> '.join(gen_rels[:4])}")


if __name__ == '__main__':
    main()
