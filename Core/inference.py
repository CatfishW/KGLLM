"""
Inference script for KG Path Diffusion Model.

Generate reasoning paths given questions and knowledge graphs from validation data.

Usage:
    python inference.py --checkpoint outputs_1/checkpoints/last.ckpt \
                        --vocab outputs_1/vocab.json \
                        --data ../Data/webqsp_combined/val.jsonl \
                        --output results.jsonl
"""

import os
import sys
import argparse
import torch
import json
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from pathlib import Path

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
    print(f"Loading vocabulary from {vocab_path}...")
    vocab = EntityRelationVocab.load(vocab_path)
    
    print(f"Loading model from {checkpoint_path}...")
    model = KGPathDiffusionLightning.load_from_checkpoint(
        checkpoint_path,
        num_entities=vocab.num_entities,
        num_relations=vocab.num_relations,
        map_location=device,
        strict=False  # Allow loading checkpoints with extra/missing keys
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
    
    # Decode entities (skip if all zeros, which means predict_entities=False)
    for idx in entities.tolist():
        if idx == 0:
            # Skip zero entities (used when predict_entities=False)
            continue
        name = vocab.idx2entity.get(idx, "<UNK>")
        if name not in ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]:
            entity_names.append(name)
    
    # Decode relations
    for idx in relations.tolist():
        name = vocab.idx2relation.get(idx, "<UNK>")
        if name not in ["<PAD>", "<MASK>", "<UNK>"]:
            relation_names.append(name)
    
    # Build path string
    # If no entities (relation-only mode), just show relations
    if not entity_names:
        # Relation-only path: just show the relation chain
        path_string = " -> ".join(relation_names) if relation_names else ""
    else:
        # Entity-relation path: (entity1) --[relation1]--> (entity2) --[relation2]--> ...
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


def load_raw_data(data_path: str) -> Dict[str, Dict]:
    """Load raw data to get questions and ground truth paths."""
    raw_data = {}
    path = Path(data_path)
    
    if path.suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                raw_data[sample.get('id', '')] = sample
    elif path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for sample in data:
                    raw_data[sample.get('id', '')] = sample
            else:
                raw_data = data
    elif path.suffix == '.parquet':
        # Load parquet file
        import pandas as pd
        df = pd.read_parquet(path)
        # Convert to dict format
        for _, row in df.iterrows():
            sample = row.to_dict()
            # Handle JSON-encoded fields
            for key in ['graph', 'paths', 'answer', 'q_entity', 'a_entity']:
                if key in sample and isinstance(sample[key], str):
                    try:
                        sample[key] = json.loads(sample[key])
                    except:
                        pass
            raw_data[sample.get('id', '')] = sample
    
    return raw_data


@torch.no_grad()
def generate_paths(
    model: KGPathDiffusionLightning,
    dataloader: DataLoader,
    vocab: EntityRelationVocab,
    raw_data: Dict[str, Dict],
    device: str = 'cuda',
    path_length: int = 10,
    temperature: float = 1.0,
    num_samples: int = 1,
    generate_multiple: bool = False,
    num_paths_per_sample: int = 5
) -> List[Dict[str, Any]]:
    """Generate paths for all samples in dataloader.
    
    Args:
        model: The trained model
        dataloader: DataLoader for input data
        vocab: Vocabulary for decoding
        raw_data: Raw data dict for getting original info
        device: Device to run on
        path_length: Length of paths to generate
        temperature: Sampling temperature
        num_samples: Number of independent generation runs (legacy)
        generate_multiple: If True, use generate_multiple to get diverse paths
        num_paths_per_sample: Number of diverse paths to generate per sample
    """
    results = []
    
    for batch in tqdm(dataloader, desc="Generating paths"):
        # Move batch to device
        batch['question_input_ids'] = batch['question_input_ids'].to(device)
        batch['question_attention_mask'] = batch['question_attention_mask'].to(device)
        
        batch_size = batch['question_input_ids'].shape[0]
        
        if generate_multiple:
            # Generate multiple diverse paths at once
            all_entities, all_relations = model.model.generate_multiple(
                batch['question_input_ids'],
                batch['question_attention_mask'],
                num_paths=num_paths_per_sample,
                path_length=path_length,
                temperature=temperature
            )
            
            # Decode each sample in batch
            for i in range(batch_size):
                sample_id = batch['ids'][i]
                original = raw_data.get(sample_id, {})
                
                # Decode all generated paths
                generated_paths = []
                for path_idx in range(num_paths_per_sample):
                    decoded = decode_path(
                        all_entities[i, path_idx], 
                        all_relations[i, path_idx], 
                        vocab
                    )
                    generated_paths.append(decoded)
                
                result = {
                    "id": sample_id,
                    "question": original.get('question', ''),
                    "answer": original.get('answer', ''),
                    "q_entity": original.get('q_entity', []),
                    "a_entity": original.get('a_entity', []),
                    "generated_paths": [
                        {
                            "path_string": p['path_string'],
                            "entities": p['entities'],
                            "relations": p['relations'],
                            "relation_chain": p['relation_chain']
                        }
                        for p in generated_paths
                    ],
                    # First path for backward compatibility
                    "generated_path": generated_paths[0]['path_string'],
                    "generated_entities": generated_paths[0]['entities'],
                    "generated_relations": generated_paths[0]['relations'],
                    "generated_relation_chain": generated_paths[0]['relation_chain'],
                }
                
                # Add ground truth paths if available
                gt_paths = original.get('paths', [])
                if gt_paths:
                    result['ground_truth_paths'] = [
                        {
                            'path_string': p.get('full_path', ''),
                            'relation_chain': p.get('relation_chain', ''),
                            'entities': p.get('entities', []),
                            'relations': p.get('relations', [])
                        }
                        for p in gt_paths[:10]  # Show more ground truth paths
                    ]
                
                results.append(result)
        else:
            # Legacy: generate single paths multiple times
            for sample_idx in range(num_samples):
                entities, relations = model.model.generate(
                    batch['question_input_ids'],
                    batch['question_attention_mask'],
                    path_length=path_length,
                    temperature=temperature
                )
                
                # Decode each sample in batch
                for i in range(batch_size):
                    sample_id = batch['ids'][i]
                    decoded = decode_path(entities[i], relations[i], vocab)
                    
                    # Get original data for this sample
                    original = raw_data.get(sample_id, {})
                    
                    result = {
                        "id": sample_id,
                        "sample_idx": sample_idx,
                        "question": original.get('question', ''),
                        "answer": original.get('answer', ''),
                        "q_entity": original.get('q_entity', []),
                        "a_entity": original.get('a_entity', []),
                        "generated_path": decoded['path_string'],
                        "generated_entities": decoded['entities'],
                        "generated_relations": decoded['relations'],
                        "generated_relation_chain": decoded['relation_chain'],
                    }
                    
                    # Add ground truth paths if available
                    gt_paths = original.get('paths', [])
                    if gt_paths:
                        result['ground_truth_paths'] = [
                            {
                                'path_string': p.get('full_path', ''),
                                'relation_chain': p.get('relation_chain', ''),
                                'entities': p.get('entities', []),
                                'relations': p.get('relations', [])
                            }
                            for p in gt_paths[:3]
                        ]
                    
                    results.append(result)
    
    return results


def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    total = len(results)
    if total == 0:
        return {}
    
    # Track various metrics
    relation_match = 0
    entity_match = 0
    exact_path_match = 0
    partial_relation_match = 0
    
    for result in results:
        gt_paths = result.get('ground_truth_paths', [])
        if not gt_paths:
            continue
        
        gen_relations = set(result.get('generated_relations', []))
        gen_entities = set(result.get('generated_entities', []))
        gen_chain = result.get('generated_relation_chain', '')
        
        # Check against all ground truth paths
        for gt in gt_paths:
            gt_relations = set(gt.get('relations', []))
            gt_entities = set(gt.get('entities', []))
            gt_chain = gt.get('relation_chain', '')
            
            # Exact relation chain match
            if gen_chain == gt_chain:
                relation_match += 1
                break
            
            # Check relation overlap
            if gen_relations and gt_relations:
                overlap = len(gen_relations & gt_relations) / len(gt_relations)
                if overlap > 0.5:
                    partial_relation_match += 1
                    break
    
    samples_with_gt = sum(1 for r in results if r.get('ground_truth_paths'))
    
    metrics = {
        'total_samples': total,
        'samples_with_ground_truth': samples_with_gt,
    }
    
    if samples_with_gt > 0:
        metrics['relation_chain_accuracy'] = relation_match / samples_with_gt
        metrics['partial_relation_match'] = partial_relation_match / samples_with_gt
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Generate reasoning paths with KG Path Diffusion Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic inference with default settings
    python inference.py --checkpoint outputs_1/checkpoints/last.ckpt \\
                        --vocab outputs_1/vocab.json \\
                        --data ../Data/webqsp_combined/val.jsonl

    # Generate longer paths with lower temperature (more deterministic)
    python inference.py --checkpoint outputs_1/checkpoints/last.ckpt \\
                        --vocab outputs_1/vocab.json \\
                        --data ../Data/webqsp_combined/val.jsonl \\
                        --path_length 15 --temperature 0.5

    # Generate multiple path samples per question
    python inference.py --checkpoint outputs_1/checkpoints/last.ckpt \\
                        --vocab outputs_1/vocab.json \\
                        --data ../Data/webqsp_combined/val.jsonl \\
                        --num_samples 3
        """
    )
    
    # Model and data paths
    parser.add_argument('--checkpoint', type=str, 
                        default='outputs_1/checkpoints/last.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, 
                        default='outputs_1/vocab.json',
                        help='Path to vocabulary file')
    parser.add_argument('--data', type=str, 
                        default='../Data/webqsp_combined/val.jsonl',
                        help='Path to input data (jsonl or parquet)')
    parser.add_argument('--output', type=str, 
                        default='inference_results.jsonl',
                        help='Path to output file (jsonl)')
    
    # Generation parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--path_length', type=int, default=10,
                        help='Maximum generated path length (num entities)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (0 for greedy, >1 for more random)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of path samples per question (legacy mode)')
    parser.add_argument('--multipath', action='store_true',
                        help='Enable multi-path generation (generates diverse paths)')
    parser.add_argument('--num_paths', type=int, default=5,
                        help='Number of diverse paths to generate per question (with --multipath)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # Output options
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--show_examples', type=int, default=10,
                        help='Number of example outputs to display')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("=" * 70)
    print("KG Path Diffusion Model - Inference")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Vocabulary: {args.vocab}")
    print(f"Data: {args.data}")
    print(f"Device: {args.device}")
    print(f"Path length: {args.path_length}")
    print(f"Temperature: {args.temperature}")
    print(f"Num samples: {args.num_samples}")
    print("=" * 70)
    
    # Check if files exist
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    if not os.path.exists(args.vocab):
        print(f"Error: Vocabulary not found: {args.vocab}")
        return
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        return
    
    # Load model
    model, vocab = load_model(args.checkpoint, args.vocab, args.device)
    print(f"Vocabulary: {vocab.num_entities} entities, {vocab.num_relations} relations")
    
    # Load raw data for questions and ground truth
    print(f"\nLoading raw data from {args.data}...")
    raw_data = load_raw_data(args.data)
    print(f"Loaded {len(raw_data)} samples")
    
    # Create dataset
    print(f"\nPreparing dataset...")
    tokenizer_name = model.hparams.get('question_encoder', 'sentence-transformers/all-MiniLM-L6-v2')
    
    dataset = KGPathDataset(
        args.data,
        vocab=vocab,
        build_vocab=False,
        max_path_length=args.path_length,
        training=False,  # Use first path for consistent evaluation
        tokenizer_name=tokenizer_name
    )
    
    # Limit samples if requested
    if args.max_samples is not None:
        dataset.samples = dataset.samples[:args.max_samples]
    
    print(f"Processing {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_kg_batch,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Generate paths
    if args.multipath:
        print(f"\nGenerating {args.num_paths} diverse paths per sample...")
    else:
        print(f"\nGenerating paths...")
    
    results = generate_paths(
        model, dataloader, vocab, raw_data,
        device=args.device,
        path_length=args.path_length,
        temperature=args.temperature,
        num_samples=args.num_samples,
        generate_multiple=args.multipath,
        num_paths_per_sample=args.num_paths
    )
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(results)} generated paths")
    
    # Compute and display metrics
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)
    metrics = compute_metrics(results)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Print sample outputs
    if args.show_examples > 0 and not args.quiet:
        print("\n" + "=" * 70)
        print(f"SAMPLE OUTPUTS (showing {min(args.show_examples, len(results))})")
        print("=" * 70)
        
        # Set UTF-8 encoding for Windows console if possible
        import sys
        if sys.platform == 'win32':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except:
                pass  # If reconfiguration fails, continue with default encoding
        
        for i, result in enumerate(results[:args.show_examples]):
            print(f"\n--- Sample {i+1} ---")
            print(f"ID: {result['id']}")
            
            # Safely print Unicode strings
            def safe_print(label, value):
                try:
                    if isinstance(value, (list, dict)):
                        print(f"{label}: {value}")
                    else:
                        print(f"{label}: {value}")
                except UnicodeEncodeError:
                    # Fallback: encode to ASCII with error handling
                    safe_value = str(value).encode('ascii', 'replace').decode('ascii')
                    print(f"{label}: {safe_value}")
            
            safe_print("Question", result.get('question', 'N/A'))
            safe_print("Answer", result.get('answer', 'N/A'))
            safe_print("Q Entity", result.get('q_entity', 'N/A'))
            safe_print("A Entity", result.get('a_entity', 'N/A'))
            
            # Show all generated paths if multipath mode
            if 'generated_paths' in result:
                print(f"\nGenerated Paths ({len(result['generated_paths'])}):")
                for j, gp in enumerate(result['generated_paths'][:5]):
                    safe_print(f"  [{j+1}]", gp.get('path_string', 'N/A'))
            else:
                safe_print("\nGenerated Path", result.get('generated_path', 'N/A'))
                safe_print("Generated Relations", result.get('generated_relation_chain', 'N/A'))
            
            gt_paths = result.get('ground_truth_paths', [])
            if gt_paths:
                print(f"\nGround Truth Paths ({len(gt_paths)}):")
                for j, gt in enumerate(gt_paths[:3]):
                    safe_print(f"  [{j+1}]", gt.get('path_string', 'N/A'))
            
            print("-" * 50)
    
    print("\nInference complete!")


if __name__ == '__main__':
    main()
