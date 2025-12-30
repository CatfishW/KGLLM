"""
Comprehensive Evaluation Script for KG Diffusion Retriever.

Evaluates on WebQSP and CWQ test sets with:
- Path Retrieval Metrics: Hits@1, Hits@5, Hits@10
- Question Answering Metrics: F1 Score, Exact Match

Usage:
    python evaluate_full_qa.py \
        --checkpoint outputs_fullkg/checkpoints/kg_retriever-epoch=64-val/loss=1.0174.ckpt \
        --test_data /data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/test.parquet \
        --vocab_path outputs_fullkg/vocab.json \
        --output_dir outputs_fullkg/eval_results/webqsp \
        --dataset_name webqsp
"""

import os
import sys
import json
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any
import ast

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.diffusion_retriever import KGDiffusionRetriever
from models.gnn_retriever import GNNRetriever
from data.dataset import KGRetrieverDataset, collate_fn
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive KG Retriever Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='diffusion',
                        choices=['diffusion', 'gnn'],
                        help='Model type')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data parquet')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Path to vocabulary JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--dataset_name', type=str, default='webqsp',
                        help='Dataset name for labeling')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of paths to generate per sample')
    parser.add_argument('--max_path_length', type=int, default=4,
                        help='Maximum path length to generate')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max samples to evaluate (0 = all)')
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    return text.lower().strip().replace("_", " ")


def compute_f1(prediction: Set[str], ground_truth: Set[str]) -> float:
    """Compute F1 score between predicted and ground truth answer sets."""
    if not prediction and not ground_truth:
        return 1.0
    if not prediction or not ground_truth:
        return 0.0
    
    # Normalize
    pred_norm = {normalize_text(p) for p in prediction}
    gt_norm = {normalize_text(g) for g in ground_truth}
    
    # Token-level F1 for each predicted answer
    total_f1 = 0.0
    matched = 0
    
    for pred in pred_norm:
        pred_tokens = set(pred.split())
        best_f1 = 0.0
        for gt in gt_norm:
            gt_tokens = set(gt.split())
            if not pred_tokens or not gt_tokens:
                continue
            intersection = pred_tokens & gt_tokens
            precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
            recall = len(intersection) / len(gt_tokens) if gt_tokens else 0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)
        total_f1 += best_f1
        if best_f1 > 0.5:
            matched += 1
    
    # Average F1 across predictions
    if pred_norm:
        return total_f1 / len(pred_norm)
    return 0.0


def compute_exact_match(prediction: Set[str], ground_truth: Set[str]) -> float:
    """Compute exact match score."""
    pred_norm = {normalize_text(p) for p in prediction}
    gt_norm = {normalize_text(g) for g in ground_truth}
    
    if not gt_norm:
        return 1.0 if not pred_norm else 0.0
    
    # Check if any prediction exactly matches any ground truth
    if pred_norm & gt_norm:
        return 1.0
    return 0.0


def compute_hits_at_k(generated_paths: List[List[str]], 
                      ground_truth_paths: List[List[str]], 
                      k: int) -> bool:
    """
    Check if any of top-k generated paths match ground truth.
    
    Matching criteria: GT path is a prefix of generated path, or exact match.
    This handles cases where generated paths are longer than GT paths.
    """
    if not generated_paths or not ground_truth_paths:
        return False
    
    # Normalize GT paths  
    gt_paths_clean = []
    for path in ground_truth_paths:
        if isinstance(path, list):
            gt_paths_clean.append(path)
        elif isinstance(path, dict) and 'relations' in path:
            gt_paths_clean.append(path['relations'])
    
    if not gt_paths_clean:
        return False
    
    # Check each generated path against GT
    for i, gen_path in enumerate(generated_paths[:k]):
        if not gen_path:
            continue
        for gt_path in gt_paths_clean:
            if not gt_path:
                continue
            gt_len = len(gt_path)
            # Check if GT is prefix of generated, or exact match
            if len(gen_path) >= gt_len:
                if gen_path[:gt_len] == gt_path:
                    return True
            # Also check if generated is prefix of GT (for longer GT paths)
            else:
                if gt_path[:len(gen_path)] == gen_path:
                    return True
    return False


def compute_relation_hits_at_1(generated_paths: List[List[str]], 
                                ground_truth_paths: List[List[str]]) -> bool:
    """
    Check if the first relation in any top-k path matches first GT relation.
    This is a more lenient metric for path retrieval.
    """
    if not generated_paths or not ground_truth_paths:
        return False
    
    # Get first relations from GT paths
    gt_first_rels = set()
    for path in ground_truth_paths:
        if isinstance(path, list) and path:
            gt_first_rels.add(path[0])
        elif isinstance(path, dict) and 'relations' in path and path['relations']:
            gt_first_rels.add(path['relations'][0])
    
    if not gt_first_rels:
        return False
    
    # Check first relation of top generated path
    for gen_path in generated_paths[:1]:
        if gen_path and gen_path[0] in gt_first_rels:
            return True
    
    return False


def parse_ground_truth_paths(paths_data: Any) -> List[Dict]:
    """Parse ground truth paths from various formats."""
    if paths_data is None:
        return []
    
    if isinstance(paths_data, str):
        try:
            paths_data = json.loads(paths_data)
        except json.JSONDecodeError:
            try:
                paths_data = ast.literal_eval(paths_data)
            except:
                return []
    
    if not isinstance(paths_data, list):
        return []
    
    return paths_data


def extract_answer_entities_from_path(
    path_relations: List[str],
    topic_entities: List[str],
    graph: List[List[str]]
) -> Set[str]:
    """
    Follow path in KG to extract answer entities.
    
    Args:
        path_relations: List of relations forming the path
        topic_entities: Starting entities (question entities)
        graph: KG as list of triples [head, relation, tail]
    
    Returns:
        Set of answer entity strings reached by following the path
    """
    if not path_relations or not topic_entities or not graph:
        return set()
    
    # Build efficient lookup: (head, relation) -> list of tails
    graph_dict = defaultdict(list)
    for triple in graph:
        if len(triple) >= 3:
            head, rel, tail = triple[0], triple[1], triple[2]
            graph_dict[(head, rel)].append(tail)
            # Also add reverse for some relations
            graph_dict[(tail, rel)].append(head)
    
    # Start from topic entities
    current_entities = set(topic_entities)
    
    # Follow each relation in the path
    for rel in path_relations:
        next_entities = set()
        for entity in current_entities:
            # Try forward direction
            next_entities.update(graph_dict.get((entity, rel), []))
        
        if not next_entities:
            break
        current_entities = next_entities
    
    return current_entities


def load_test_data(test_path: str) -> pd.DataFrame:
    """Load and preprocess test data."""
    df = pd.read_parquet(test_path)
    return df


class EvaluationRunner:
    """Main evaluation class."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary
        with open(args.vocab_path, 'r') as f:
            self.vocab_data = json.load(f)
        
        self.num_relations = self.vocab_data['num_relations']
        self.relation_to_idx = self.vocab_data['relation_to_idx']
        self.idx_to_relation = {v: k for k, v in self.relation_to_idx.items()}
        
        print(f"Loaded vocabulary with {self.num_relations} relations")
        
        # Load model
        self._load_model()
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_model(self):
        """Load the trained model."""
        print(f"Loading {self.args.model_type} model from {self.args.checkpoint}")
        
        if self.args.model_type == 'diffusion':
            self.model = KGDiffusionRetriever.load_from_checkpoint(
                self.args.checkpoint,
                num_relations=self.num_relations,
                strict=False,
                map_location='cpu',
            )
        else:
            self.model = GNNRetriever.load_from_checkpoint(
                self.args.checkpoint,
                num_relations=self.num_relations,
                strict=False,
                map_location='cpu',
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _create_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        test_dataset = KGRetrieverDataset(
            data_path=self.args.test_data,
            vocab_path=None,
            max_triples=0,
            max_path_length=8,
            training=False,
        )
        
        # Override with training vocabulary
        test_dataset.relation_to_idx = self.relation_to_idx
        
        print(f"Test dataset: {len(test_dataset)} samples")
        
        return DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
    
    def decode_path(self, relation_ids: List[int]) -> List[str]:
        """Decode relation IDs to relation names, stopping at EOS token."""
        relations = []
        EOS_IDX = 3  # End of sequence token
        for r in relation_ids:
            if r == EOS_IDX:  # Stop at EOS token
                break
            if r > 3:  # Skip PAD (0), UNK (1), MASK (2), EOS (3)
                relations.append(self.idx_to_relation.get(r, f'<UNK:{r}>'))
        return relations
    
    @torch.no_grad()
    def generate_paths(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate paths for a batch."""
        batch_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
        
        if self.args.model_type == 'diffusion':
            generated = self.model.generate(
                question_input_ids=batch_gpu['question_input_ids'],
                question_attention_mask=batch_gpu['question_attention_mask'],
                kg_relation_ids=batch_gpu['kg_relation_ids'],
                kg_head_hash_ids=batch_gpu['kg_head_hash_ids'],
                kg_tail_hash_ids=batch_gpu['kg_tail_hash_ids'],
                kg_triple_mask=batch_gpu['kg_triple_mask'],
                path_length=self.args.max_path_length,
                num_samples=self.args.num_samples,
            )
        else:
            # GNN generates single path
            generated = self.model.generate(
                question_input_ids=batch_gpu['question_input_ids'],
                question_attention_mask=batch_gpu['question_attention_mask'],
                kg_relation_ids=batch_gpu['kg_relation_ids'],
                kg_head_hash_ids=batch_gpu['kg_head_hash_ids'],
                kg_tail_hash_ids=batch_gpu['kg_tail_hash_ids'],
                kg_triple_mask=batch_gpu['kg_triple_mask'],
                max_length=self.args.max_path_length,
            ).unsqueeze(1)  # Add sample dim
        
        return generated.cpu()
    
    def run(self) -> Dict[str, Any]:
        """Run full evaluation."""
        # Load raw test data for answer extraction
        raw_df = load_test_data(self.args.test_data)
        raw_data = raw_df.to_dict('records')
        
        # Create id to row mapping
        id_to_row = {}
        for row in raw_data:
            id_to_row[row['id']] = row
        
        # Create dataloader
        dataloader = self._create_dataloader()
        
        # Metrics accumulators
        results = []
        hits = {1: 0, 5: 0, 10: 0}
        relation_hits_1 = 0
        f1_scores = []
        em_scores = []
        total_samples = 0
        
        print(f"\nRunning evaluation on {self.args.dataset_name}...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # Generate paths
            generated = self.generate_paths(batch)
            B = len(batch['id'])
            
            for i in range(B):
                sample_id = batch['id'][i]
                
                # Get raw data for this sample
                raw_row = id_to_row.get(sample_id, {})
                if not raw_row:
                    continue
                
                # Parse ground truth
                gt_paths_data = raw_row.get('shortest_gt_paths') or raw_row.get('paths') or []
                gt_paths = parse_ground_truth_paths(gt_paths_data)
                
                # Get ground truth answer entities
                gt_answers = raw_row.get('a_entity', [])
                if isinstance(gt_answers, str):
                    try:
                        gt_answers = json.loads(gt_answers)
                    except:
                        gt_answers = [gt_answers]
                gt_answers = set(gt_answers) if gt_answers else set()
                
                # Get topic entities
                topic_entities = raw_row.get('q_entity', [])
                if isinstance(topic_entities, str):
                    try:
                        topic_entities = json.loads(topic_entities)
                    except:
                        topic_entities = [topic_entities]
                
                # Parse graph
                graph = raw_row.get('graph', [])
                if isinstance(graph, str):
                    try:
                        graph = json.loads(graph)
                    except:
                        try:
                            graph = ast.literal_eval(graph)
                        except:
                            graph = []
                
                # Decode generated paths
                generated_paths = []
                for j in range(generated.size(1)):
                    path_ids = generated[i, j].tolist()
                    path_rels = self.decode_path(path_ids)
                    if path_rels:
                        generated_paths.append(path_rels)
                
                # Get ground truth relation paths
                gt_relation_paths = []
                for path in gt_paths:
                    if isinstance(path, dict) and 'relations' in path:
                        gt_relation_paths.append(path['relations'])
                    elif isinstance(path, list):
                        gt_relation_paths.append(path)
                
                # Compute Hits@K for path retrieval
                for k in [1, 5, 10]:
                    if compute_hits_at_k(generated_paths, gt_relation_paths, k):
                        hits[k] += 1
                
                # Compute relation-level Hits@1
                if compute_relation_hits_at_1(generated_paths, gt_relation_paths):
                    relation_hits_1 += 1
                
                # Extract answer entities from each generated path
                all_predicted_answers = set()
                for path_rels in generated_paths[:5]:  # Use top 5 paths
                    answers = extract_answer_entities_from_path(
                        path_rels, topic_entities, graph
                    )
                    all_predicted_answers.update(answers)
                
                # Compute QA metrics
                f1 = compute_f1(all_predicted_answers, gt_answers)
                em = compute_exact_match(all_predicted_answers, gt_answers)
                f1_scores.append(f1)
                em_scores.append(em)
                
                # Store detailed result
                results.append({
                    'id': sample_id,
                    'question': raw_row.get('question', ''),
                    'gt_paths': gt_relation_paths[:3],
                    'generated_paths': generated_paths[:5],
                    'gt_answers': list(gt_answers),
                    'predicted_answers': list(all_predicted_answers)[:10],
                    'f1': f1,
                    'em': em,
                    'hits_at_1': compute_hits_at_k(generated_paths, gt_relation_paths, 1),
                    'hits_at_10': compute_hits_at_k(generated_paths, gt_relation_paths, 10),
                })
                
                total_samples += 1
                
                if self.args.max_samples > 0 and total_samples >= self.args.max_samples:
                    break
            
            if self.args.max_samples > 0 and total_samples >= self.args.max_samples:
                break
        
        # Compute final metrics
        metrics = {
            'dataset': self.args.dataset_name,
            'total_samples': total_samples,
            'hits_at_1': hits[1] / total_samples if total_samples > 0 else 0,
            'hits_at_5': hits[5] / total_samples if total_samples > 0 else 0,
            'hits_at_10': hits[10] / total_samples if total_samples > 0 else 0,
            'relation_hits_at_1': relation_hits_1 / total_samples if total_samples > 0 else 0,
            'f1_score': np.mean(f1_scores) if f1_scores else 0,
            'exact_match': np.mean(em_scores) if em_scores else 0,
        }
        
        # Save results
        self._save_results(results, metrics)
        
        # Print summary
        self._print_summary(metrics)
        
        return metrics
    
    def _save_results(self, results: List[Dict], metrics: Dict):
        """Save evaluation results to files."""
        output_dir = Path(self.args.output_dir)
        
        # Save detailed results
        results_path = output_dir / 'detailed_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved detailed results to: {results_path}")
        
        # Save metrics summary
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")
        
        # Save as CSV too
        metrics_csv_path = output_dir / 'metrics.csv'
        pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)
        print(f"Saved metrics CSV to: {metrics_csv_path}")
    
    def _print_summary(self, metrics: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS - {self.args.dataset_name.upper()} Test Set")
        print("=" * 60)
        print(f"Samples Evaluated: {metrics['total_samples']}")
        print()
        print("PATH RETRIEVAL METRICS:")
        print(f"  Hits@1 (prefix):     {metrics['hits_at_1']*100:.2f}%")
        print(f"  Hits@5 (prefix):     {metrics['hits_at_5']*100:.2f}%")
        print(f"  Hits@10 (prefix):    {metrics['hits_at_10']*100:.2f}%")
        print(f"  Relation Hits@1:     {metrics['relation_hits_at_1']*100:.2f}%")
        print()
        print("QUESTION ANSWERING METRICS:")
        print(f"  F1 Score:     {metrics['f1_score']*100:.2f}%")
        print(f"  Exact Match:  {metrics['exact_match']*100:.2f}%")
        print("=" * 60)


def main():
    args = parse_args()
    runner = EvaluationRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
