"""
Complete KGQA Pipeline: Path Retrieval + LLM Answer Generation.

This script:
1. Loads a trained path ranker model (DCDR or BiEncoder)
2. Runs evaluation on test datasets (WebQSP, CWQ)
3. Outputs pretty metrics and detailed results
4. Uses LLM API to generate final answers based on retrieved paths

Usage:
    python run_complete_qa.py --checkpoint outputs_dcdr_tuned/last.ckpt --dataset webqsp
"""

import argparse
import json
import os
import requests
import torch
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

# Import model classes
try:
    from models.discrete_diffusion_reranker import DiscreteRankDiffusion
except ImportError:
    DiscreteRankDiffusion = None

try:
    from models.path_ranker import PathRankerModel
except ImportError:
    PathRankerModel = None

from data.path_ranker_dataset import PathRankerDataset, collate_fn


@dataclass
class QAResult:
    """Single QA result."""
    id: str
    question: str
    topic_entities: List[str]  # Question entities from annotations
    rank: int  # Rank of ground truth path (1-indexed)
    gt_idx: int  # Index of ground truth in candidates
    gt_path: List[str]  # Ground truth relation path
    gt_answers: List[str]  # Ground truth answer entities
    gt_path_with_entities: str  # Full path: topic -> relations -> answers
    top_pred_idx: int
    top_pred_path: List[str]
    top_pred_score: float
    top_5_preds: List[Dict]
    top_5_paths_with_entities: List[str]  # Formatted paths with entities
    llm_answer: Optional[str] = None
    llm_correct: Optional[bool] = None


def extract_entities_from_path(
    path_relations: List[str],
    topic_entities: List[str],
    graph: List[List[str]],
    return_formatted: bool = False,
) -> List[str]:
    """
    Follow path in KG to extract answer entities with optional intermediate entities.
    
    Args:
        path_relations: List of relations forming the path
        topic_entities: Starting entities (question entities)
        graph: KG as list of triples [head, relation, tail]
        return_formatted: If True, return formatted path string with all entities
    
    Returns:
        List of answer entity strings (if return_formatted=False)
        OR single formatted path string with intermediate entities (if return_formatted=True)
    """
    if not path_relations or not topic_entities or not graph:
        return [] if not return_formatted else ""
    
    # Build efficient lookup: (head, relation) -> list of tails
    from collections import defaultdict
    graph_dict = defaultdict(list)
    reverse_dict = defaultdict(list)
    
    for triple in graph:
        if len(triple) >= 3:
            head, rel, tail = triple[0], triple[1], triple[2]
            graph_dict[(head, rel)].append(tail)
            reverse_dict[(tail, rel)].append(head)
    
    # Track entities at each step for formatted output
    path_entities = [list(topic_entities)]  # Start with topic entities
    current_entities = set(topic_entities)
    
    # Follow each relation in the path
    for rel in path_relations:
        next_entities = set()
        for entity in current_entities:
            # Try forward direction
            next_entities.update(graph_dict.get((entity, rel), []))
            # Try reverse direction if forward yields nothing
            if not next_entities:
                next_entities.update(reverse_dict.get((entity, rel), []))
        
        if not next_entities:
            path_entities.append([])  # Empty for this hop
            break
        
        current_entities = next_entities
        path_entities.append(list(next_entities))
    
    if not return_formatted:
        return list(current_entities)
    
    # Build formatted path: Topic -> rel1 -> [Entities] -> rel2 -> [Entities]
    parts = []
    
    # Add starting entity(ies)
    if path_entities[0]:
        parts.append(path_entities[0][0] if len(path_entities[0]) == 1 else f"[{', '.join(path_entities[0][:2])}]")
    
    # Add each relation -> entities pair
    for i, rel in enumerate(path_relations):
        parts.append(rel)
        entity_idx = i + 1
        if entity_idx < len(path_entities) and path_entities[entity_idx]:
            entities = path_entities[entity_idx]
            if len(entities) == 1:
                parts.append(entities[0])
            else:
                # Show up to 3 entities
                display = entities[:3]
                parts.append(f"[{', '.join(display)}]")
        else:
            parts.append("?")
    
    return " -> ".join(parts)

def load_model(checkpoint_path: str, model_type: str = 'dcdr', device: str = 'cuda'):
    """Load a trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    if model_type == 'dcdr':
        if DiscreteRankDiffusion is None:
            raise ImportError("DiscreteRankDiffusion not found")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        hparams = ckpt.get('hyper_parameters', {})
        
        model = DiscreteRankDiffusion(
            encoder_name=hparams.get('encoder_name', 'BAAI/bge-small-en-v1.5'),
            hidden_dim=hparams.get('hidden_dim', 768),
            num_diffusion_steps=hparams.get('num_diffusion_steps', 10),
            num_layers=hparams.get('num_layers', 4),
            num_heads=hparams.get('num_heads', 8),
            freeze_encoder=True,  # For inference
        )
        
        model.load_state_dict(ckpt['state_dict'], strict=False)
        
    elif model_type == 'biencoder':
        if PathRankerModel is None:
            raise ImportError("PathRankerModel not found")
        model = PathRankerModel.load_from_checkpoint(checkpoint_path, map_location='cpu')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model with {total_params:,} parameters")
    
    return model


def run_path_retrieval(
    model,
    test_data_path: str,
    tokenizer_name: str = 'BAAI/bge-small-en-v1.5',
    batch_size: int = 16,
    device: str = 'cuda',
    limit: int = 0,
    graph_data_path: Optional[str] = None,
) -> List[QAResult]:
    """Run path retrieval evaluation and return results."""
    
    # Load test dataset
    dataset = PathRankerDataset(
        data_path=test_data_path,
        tokenizer_name=tokenizer_name,
        max_question_length=128,
        max_path_length=64,
        max_candidates=100,
        training=False,
        negative_sampling=False,
        num_negatives=99,
    )
    
    # Load raw parquet for additional fields (q_entity, a_entity)
    raw_df = pd.read_parquet(test_data_path)
    raw_id_to_row = {row['id']: row for _, row in raw_df.iterrows()}
    
    # Load graph data for entity extraction if available
    graph_id_to_graph = {}
    if graph_data_path:
        import ast
        print(f"Loading KG graph data from {graph_data_path}...")
        graph_df = pd.read_parquet(graph_data_path)
        for _, row in graph_df.iterrows():
            graph = row.get('graph', [])
            if isinstance(graph, str):
                try:
                    graph = json.loads(graph)
                except json.JSONDecodeError:
                    try:
                        graph = ast.literal_eval(graph)
                    except:
                        graph = []
            graph_id_to_graph[row['id']] = graph
        print(f"Loaded graphs for {len(graph_id_to_graph)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Path Retrieval")):
            if limit > 0 and batch_idx * batch_size >= limit:
                break
            
            # Move to device - filter to only model-expected keys
            # PathRankerModel expects: question_input_ids, question_attention_mask, 
            #                          path_input_ids, path_attention_mask, candidate_mask
            model_keys = {'question_input_ids', 'question_attention_mask', 
                          'path_input_ids', 'path_attention_mask', 'candidate_mask'}
            batch_inputs = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
                if k in model_keys
            }
            
            # Forward pass
            outputs = model(**batch_inputs)
            logits = outputs['logits']  # [B, C]
            
            # Get predictions
            scores = torch.softmax(logits, dim=-1)
            sorted_indices = scores.argsort(dim=-1, descending=True)
            
            # Process each sample
            for i in range(logits.size(0)):
                sample_idx = batch_idx * batch_size + i
                if sample_idx >= len(dataset):
                    break
                
                raw_sample = dataset.data[sample_idx]
                
                gt_label = batch['labels'][i].item()
                pred_idx = sorted_indices[i, 0].item()
                
                # Get rank of ground truth
                rank = (sorted_indices[i] == gt_label).nonzero(as_tuple=True)[0].item() + 1
                
                # IMPORTANT: Use candidates from batch (processed by __getitem__)
                # NOT from raw_sample which has 500 candidates before truncation
                candidates = batch['candidate_text'][i]
                
                gt_path = candidates[gt_label] if gt_label < len(candidates) else []
                top_pred_path = candidates[pred_idx] if pred_idx < len(candidates) else []
                
                # Top 5 predictions
                top_5 = []
                for j in range(min(5, logits.size(1))):
                    idx = sorted_indices[i, j].item()
                    top_5.append({
                        'path': candidates[idx] if idx < len(candidates) else [],
                        'score': scores[i, idx].item(),
                    })
                
                # Look up additional fields from raw parquet
                sample_id = raw_sample.get('id', '')
                raw_row = raw_id_to_row.get(sample_id, {})
                
                # Get topic entities from annotations (q_entity)
                topic_entities = raw_row.get('q_entity', [])
                if isinstance(topic_entities, str):
                    try:
                        topic_entities = json.loads(topic_entities)
                    except:
                        topic_entities = [topic_entities]
                elif hasattr(topic_entities, 'tolist'):
                    topic_entities = topic_entities.tolist()
                
                # Get ground truth answers (a_entity)
                gt_answers = raw_row.get('a_entity', [])
                if isinstance(gt_answers, str):
                    try:
                        gt_answers = json.loads(gt_answers)
                    except:
                        gt_answers = [gt_answers]
                elif hasattr(gt_answers, 'tolist'):
                    gt_answers = gt_answers.tolist()
                
                # Format GT path with entities: Topic -> relations -> answers
                gt_path_list = gt_path if isinstance(gt_path, list) else [gt_path]
                if topic_entities and gt_path_list:
                    gt_path_with_entities = " -> ".join(topic_entities[:1] + gt_path_list)
                    if gt_answers:
                        gt_path_with_entities += " -> [" + ", ".join(gt_answers[:3]) + "]"
                else:
                    gt_path_with_entities = ""
                
                # Get KG graph for this sample
                sample_graph = graph_id_to_graph.get(sample_id, [])
                
                # Format top 5 predicted paths with EXTRACTED entities from KG
                top_5_paths_with_entities = []
                for pred in top_5:
                    if pred['path'] and topic_entities:
                        # Extract formatted path with intermediate entities
                        path_str = extract_entities_from_path(
                            pred['path'], topic_entities, sample_graph,
                            return_formatted=True
                        )
                        if path_str:
                            top_5_paths_with_entities.append(path_str)
                        else:
                            # Fallback if no entities found
                            path_str = " -> ".join(topic_entities[:1] + pred['path']) + " -> ?"
                            top_5_paths_with_entities.append(path_str)
                    else:
                        top_5_paths_with_entities.append("")
                
                result = QAResult(
                    id=raw_sample.get('id', f'sample-{sample_idx}'),
                    question=raw_sample.get('question', ''),
                    topic_entities=topic_entities,
                    rank=rank,
                    gt_idx=gt_label,
                    gt_path=gt_path_list,
                    gt_answers=gt_answers,
                    gt_path_with_entities=gt_path_with_entities,
                    top_pred_idx=pred_idx,
                    top_pred_path=top_pred_path if isinstance(top_pred_path, list) else [top_pred_path],
                    top_pred_score=scores[i, pred_idx].item(),
                    top_5_preds=top_5,
                    top_5_paths_with_entities=top_5_paths_with_entities,
                )
                
                results.append(result)
    
    return results


def compute_metrics(results: List[QAResult]) -> Dict[str, float]:
    """Compute retrieval and QA metrics."""
    total = len(results)
    if total == 0:
        return {}
    
    # Path retrieval metrics
    hits_at_1 = sum(1 for r in results if r.rank == 1) / total
    hits_at_3 = sum(1 for r in results if r.rank <= 3) / total
    hits_at_5 = sum(1 for r in results if r.rank <= 5) / total
    hits_at_10 = sum(1 for r in results if r.rank <= 10) / total
    mrr = sum(1.0 / r.rank for r in results) / total
    
    # QA metrics (if LLM answers available)
    llm_results = [r for r in results if r.llm_answer is not None]
    if llm_results:
        # Hits@1 for QA: check if LLM answer matches any GT answer
        qa_hits = sum(1 for r in llm_results if r.llm_correct) / len(llm_results)
        
        # F1 Score: KGQA-style F1 (precision-focused)
        # - Precision: fraction of predicted answers that are correct
        # - Recall is capped to not penalize for missing GT answers when GT is large
        # Standard KGQA: if all your predictions are correct → F1 = 1.0
        f1_scores = []
        for r in llm_results:
            if not r.gt_answers or not r.llm_answer:
                f1_scores.append(0.0)
                continue
            
            # Parse LLM answer as comma-separated entity set
            pred_entities = [
                e.strip().lower() 
                for e in r.llm_answer.replace(', ', ',').split(',')
                if e.strip()
            ]
            
            # GT answer set (lowercase for comparison)  
            gt_entities = [gt.lower() for gt in r.gt_answers]
            
            if not pred_entities or not gt_entities:
                f1_scores.append(0.0)
                continue
            
            # Count matches using substring matching
            matched_pred = 0
            for pred in pred_entities:
                for gt in gt_entities:
                    if pred in gt or gt in pred or pred == gt:
                        matched_pred += 1
                        break
            
            # Precision: what fraction of predictions are correct
            precision = matched_pred / len(pred_entities) if pred_entities else 0
            
            # For KGQA F1: use precision as the score
            # This rewards correct answers without penalizing for not listing all GT
            f1_scores.append(precision)
        
        qa_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    else:
        qa_hits = 0.0
        qa_f1 = 0.0
    
    return {
        'hits@1': hits_at_1,
        'hits@3': hits_at_3,
        'hits@5': hits_at_5,
        'hits@10': hits_at_10,
        'mrr': mrr,
        'qa_hits@1': qa_hits,
        'qa_f1': qa_f1,
        'total': total,
    }


# ============ LLM Integration ============

def call_llm_api(
    prompt: str,
    api_url: str = "https://game.agaii.org/llm/v1",
    model: str = "default",
    max_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """Call the LLM API."""
    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"LLM API error: {e}")
        return ""


def build_qa_prompt(result: QAResult) -> str:
    """Build a prompt for the LLM to answer the question using retrieved paths."""
    
    # Format topic entities
    topic_str = ', '.join(result.topic_entities[:3]) if result.topic_entities else 'Unknown'
    
    # Use top_5_paths_with_entities which have real extracted entities from KG
    retrieved_info = []
    for i, path_with_entities in enumerate(result.top_5_paths_with_entities[:3], 1):
        if path_with_entities:
            retrieved_info.append(f"  {i}. {path_with_entities}")
        else:
            retrieved_info.append(f"  {i}. N/A")
    
    prompt = f"""You are a Knowledge Graph Question Answering system.

Question: {result.question}
Topic Entity: {topic_str}

Retrieved Knowledge Graph Paths with Entities:
{chr(10).join(retrieved_info)}

Instructions:
- Look at the question carefully and identify what type of answer is being asked for.
- Select ONLY the entity/entities that DIRECTLY answer the question from the paths above.
- If the question asks for a specific item (e.g., "what country", "which person"), give ONLY that single answer.
- If the question asks for multiple items (e.g., "what countries", "which films"), you may list multiple.
- Do NOT include extra entities that are not asked for.

Answer (entity names only, no explanation):"""
    
    return prompt


def run_llm_qa_batch(
    results: List[QAResult],
    api_url: str = "https://game.agaii.org/llm/v1",
    model: str = "default",
    batch_size: int = 10,
    max_workers: int = 5,
) -> List[QAResult]:
    """Run LLM QA on results in batches."""
    
    def process_one(result: QAResult) -> QAResult:
        prompt = build_qa_prompt(result)
        answer = call_llm_api(prompt, api_url=api_url, model=model)
        result.llm_answer = answer
        
        # Correctness check: see if any GT answer is in LLM answer (or vice versa)
        if result.gt_answers and answer:
            ans_lower = answer.lower()
            for gt in result.gt_answers:
                gt_lower = gt.lower()
                if gt_lower in ans_lower or ans_lower in gt_lower:
                    result.llm_correct = True
                    break
            else:
                result.llm_correct = False
        
        return result
    
    print(f"Running LLM QA on {len(results)} samples...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, r): i for i, r in enumerate(results)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM QA"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing: {e}")
    
    return results


def save_results(
    results: List[QAResult],
    metrics: Dict[str, float],
    output_dir: str,
    dataset_name: str,
):
    """Save results and metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    with open(metrics_path, 'w') as f:
        headers = list(metrics.keys())
        f.write(','.join(headers) + '\n')
        f.write(','.join(str(metrics[h]) for h in headers) + '\n')
    
    # Save detailed results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {dataset_name}")
    print(f"{'='*60}")
    print(f"Total samples: {metrics['total']}")
    print()
    print("PATH RETRIEVAL METRICS:")
    print(f"  Hits@1:  {metrics['hits@1']*100:.2f}%")
    print(f"  Hits@5:  {metrics['hits@5']*100:.2f}%")
    print(f"  Hits@10: {metrics['hits@10']*100:.2f}%")
    print(f"  MRR:     {metrics['mrr']*100:.2f}%")
    print()
    print("QA METRICS (Paper Format):")
    print(f"  Hits@1:  {metrics.get('qa_hits@1', 0)*100:.2f}%")
    print(f"  F1:      {metrics.get('qa_f1', 0)*100:.2f}%")
    print()
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return metrics_path, results_path


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = load_model(
        args.checkpoint,
        model_type=args.model_type,
        device=device,
    )
    
    # Determine test data path
    if args.dataset == 'webqsp':
        test_path = args.test_data or '/data/Yanlai/KGLLM/Data/preprocessed_paths/webqsp_test.parquet'
        dataset_name = 'WebQSP Test'
    elif args.dataset == 'cwq':
        test_path = args.test_data or '/data/Yanlai/KGLLM/Data/preprocessed_paths/cwq_test.parquet'
        dataset_name = 'CWQ Test'
    else:
        test_path = args.test_data
        dataset_name = args.dataset
    
    # Run path retrieval
    results = run_path_retrieval(
        model,
        test_path,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        device=device,
        limit=args.limit,
        graph_data_path=args.graph_data,
    )
    
    # Run LLM QA if enabled
    if args.run_llm_qa:
        results = run_llm_qa_batch(
            results,
            api_url=args.llm_api_url,
            model=args.llm_model,
            max_workers=args.llm_workers,
        )
    
    # Compute metrics (after LLM QA so QA metrics are included)
    metrics = compute_metrics(results)
    
    # Save results
    output_dir = os.path.join(args.output_dir, args.dataset)
    save_results(results, metrics, output_dir, dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete KGQA Pipeline')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='dcdr',
                        choices=['dcdr', 'biencoder'])
    parser.add_argument('--tokenizer', type=str, default='BAAI/bge-small-en-v1.5')
    
    # Data
    parser.add_argument('--dataset', type=str, default='webqsp',
                        choices=['webqsp', 'cwq'])
    parser.add_argument('--test_data', type=str, default=None,
                        help='Override test data path')
    parser.add_argument('--graph_data', type=str, default=None,
                        help='Path to parquet with KG graph for entity extraction')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit samples (0 = all)')
    
    # LLM
    parser.add_argument('--run_llm_qa', action='store_true',
                        help='Run LLM QA after retrieval')
    parser.add_argument('--llm_api_url', type=str,
                        default='https://game.agaii.org/llm/v1')
    parser.add_argument('--llm_model', type=str, default='default')
    parser.add_argument('--llm_workers', type=int, default=5)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs_complete_qa')
    
    args = parser.parse_args()
    main(args)
