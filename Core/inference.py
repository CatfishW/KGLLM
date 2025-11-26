"""
Inference script for KG Path Diffusion Model.

Generate reasoning paths given a question and knowledge graph.
"""

import os
import sys
import argparse
import torch
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import KGPathDataset, EntityRelationVocab, collate_kg_batch
from kg_path_diffusion import KGPathDiffusionLightning
from torch.utils.data import DataLoader


def load_model(
    checkpoint_path: str,
    vocab_path: str,
    device: str = 'cuda'
) -> tuple:
    """Load trained model and vocabulary."""
    # Load vocabulary
    vocab = EntityRelationVocab.load(vocab_path)
    
    # Load model from checkpoint
    model = KGPathDiffusionLightning.load_from_checkpoint(
        checkpoint_path,
        num_entities=vocab.num_entities,
        num_relations=vocab.num_relations
    )
    model.eval()
    model.to(device)
    
    return model, vocab


def decode_path(
    entities: torch.Tensor,
    relations: torch.Tensor,
    vocab: EntityRelationVocab
) -> Dict[str, Any]:
    """Decode generated indices back to entity/relation names."""
    entity_names = []
    relation_names = []
    
    for idx in entities.tolist():
        name = vocab.idx2entity.get(idx, "<UNK>")
        if name not in ["<PAD>", "<BOS>", "<EOS>"]:
            entity_names.append(name)
    
    for idx in relations.tolist():
        name = vocab.idx2relation.get(idx, "<UNK>")
        if name != "<PAD>":
            relation_names.append(name)
    
    # Build path string
    path_parts = []
    for i, entity in enumerate(entity_names):
        path_parts.append(f"({entity})")
        if i < len(relation_names):
            path_parts.append(f" --[{relation_names[i]}]--> ")
    
    path_string = "".join(path_parts)
    
    return {
        "entities": entity_names,
        "relations": relation_names,
        "path_string": path_string,
        "relation_chain": " -> ".join(relation_names) if relation_names else ""
    }


@torch.no_grad()
def generate_paths(
    model: KGPathDiffusionLightning,
    dataloader: DataLoader,
    vocab: EntityRelationVocab,
    device: str = 'cuda',
    path_length: int = 10,
    temperature: float = 1.0,
    num_samples: int = 1
) -> List[Dict[str, Any]]:
    """Generate paths for all samples in dataloader."""
    results = []
    
    for batch in tqdm(dataloader, desc="Generating paths"):
        # Move batch to device
        batch['question_input_ids'] = batch['question_input_ids'].to(device)
        batch['question_attention_mask'] = batch['question_attention_mask'].to(device)
        batch['graph_batch'] = batch['graph_batch'].to(device)
        
        batch_size = batch['question_input_ids'].shape[0]
        
        # Generate multiple samples if requested
        for sample_idx in range(num_samples):
            entities, relations = model.model.generate(
                batch['question_input_ids'],
                batch['question_attention_mask'],
                batch['graph_batch'],
                path_length=path_length,
                temperature=temperature
            )
            
            # Decode each sample in batch
            for i in range(batch_size):
                decoded = decode_path(entities[i], relations[i], vocab)
                
                result = {
                    "id": batch['ids'][i],
                    "sample_idx": sample_idx,
                    **decoded
                }
                results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate paths with KG Path Diffusion Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data (parquet or jsonl)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output file (jsonl)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--path_length', type=int, default=10,
                        help='Generated path length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (0 for greedy)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of path samples per question')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    print("="*60)
    print("KG Path Diffusion - Inference")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, vocab = load_model(args.checkpoint, args.vocab, args.device)
    print(f"Vocabulary: {vocab.num_entities} entities, {vocab.num_relations} relations")
    
    # Create dataset
    print(f"\nLoading data from {args.data}...")
    dataset = KGPathDataset(
        args.data,
        vocab=vocab,
        build_vocab=False
    )
    print(f"Loaded {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_kg_batch,
        pin_memory=True
    )
    
    # Generate paths
    print(f"\nGenerating paths (length={args.path_length}, temp={args.temperature})...")
    results = generate_paths(
        model, dataloader, vocab,
        device=args.device,
        path_length=args.path_length,
        temperature=args.temperature,
        num_samples=args.num_samples
    )
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(results)} generated paths")
    
    # Print some examples
    print("\n" + "="*60)
    print("SAMPLE OUTPUTS")
    print("="*60)
    for result in results[:5]:
        print(f"\nID: {result['id']}")
        print(f"Path: {result['path_string']}")
        print(f"Relations: {result['relation_chain']}")


if __name__ == '__main__':
    main()

