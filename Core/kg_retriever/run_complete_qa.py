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
import re
import time
import threading
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

try:
    from models.hop_colbert_reranker import HopColBERTReranker
except ImportError:
    HopColBERTReranker = None

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
    top_k_preds: List[Dict]
    top_k_paths_with_entities: List[str]  # Formatted paths with entities
    top_k_answer_entities: List[List[str]]  # Extracted answers per path
    llm_answer: Optional[str] = None
    llm_correct: Optional[bool] = None


def build_graph_index(graph: List[List[str]]):
    """Build lookup tables for fast path traversal."""
    from collections import defaultdict

    graph_dict = defaultdict(list)
    reverse_dict = defaultdict(list)
    for triple in graph:
        if len(triple) >= 3:
            head, rel, tail = triple[0], triple[1], triple[2]
            graph_dict[(head, rel)].append(tail)
            reverse_dict[(tail, rel)].append(head)
    return graph_dict, reverse_dict


def walk_path_entities(
    path_relations: List[str],
    topic_entities: List[str],
    graph_index,
) -> Tuple[List[str], List[List[str]]]:
    """Follow a relation path and return final entities + per-hop entities."""
    if not path_relations or not topic_entities:
        return [], []

    graph_dict, reverse_dict = graph_index
    current_entities = set(topic_entities)
    path_entities = [list(topic_entities)]

    for rel in path_relations:
        next_entities = set()
        for entity in current_entities:
            next_entities.update(graph_dict.get((entity, rel), []))
            if not next_entities:
                next_entities.update(reverse_dict.get((entity, rel), []))

        if not next_entities:
            path_entities.append([])
            break

        current_entities = next_entities
        path_entities.append(list(next_entities))

    return list(current_entities), path_entities


def format_path_with_entities(
    path_relations: List[str],
    path_entities: List[List[str]],
) -> str:
    """Format path with intermediate entities for LLM context."""
    if not path_relations or not path_entities:
        return ""

    parts = []
    if path_entities[0]:
        parts.append(
            path_entities[0][0]
            if len(path_entities[0]) == 1
            else f"[{', '.join(path_entities[0][:2])}]"
        )

    for i, rel in enumerate(path_relations):
        parts.append(rel)
        entity_idx = i + 1
        if entity_idx < len(path_entities) and path_entities[entity_idx]:
            entities = path_entities[entity_idx]
            if len(entities) == 1:
                parts.append(entities[0])
            else:
                display = entities[:100]
                parts.append(f"[{', '.join(display)}]")
        else:
            parts.append("?")

    return " -> ".join(parts)


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
    
    graph_index = build_graph_index(graph)
    final_entities, path_entities = walk_path_entities(
        path_relations, topic_entities, graph_index
    )

    if not return_formatted:
        return list(final_entities)

    return format_path_with_entities(path_relations, path_entities)

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
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        hparams = ckpt.get('hyper_parameters', {})
        model = PathRankerModel(
            encoder_name=hparams.get('encoder_name', 'BAAI/bge-base-en-v1.5'),
            hidden_dim=hparams.get('hidden_dim', 768),
            dropout=hparams.get('dropout', 0.1),
            learning_rate=hparams.get('learning_rate', 2e-5),
            weight_decay=hparams.get('weight_decay', 0.01),
            warmup_steps=hparams.get('warmup_steps', 1000),
            max_steps=hparams.get('max_steps', 50000),
            freeze_encoder=hparams.get('freeze_encoder', False),
            temperature=hparams.get('temperature', 0.05),
            max_candidates=hparams.get('max_candidates', 100),
            max_path_length=hparams.get('max_path_length', 8),
        )
        model.load_state_dict(ckpt['state_dict'], strict=False)
    elif model_type == 'hop_colbert':
        if HopColBERTReranker is None:
            raise ImportError("HopColBERTReranker not found")
        # Use load_from_checkpoint for LightningModule
        # strict=False to handle minor architecture changes if any
        model = HopColBERTReranker.load_from_checkpoint(checkpoint_path, map_location='cpu', strict=False)
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
    top_k: int = 10,
    return_graph_lookup: bool = False,
) -> Any:
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
        for _, row in tqdm(graph_df.iterrows(), total=len(graph_df), desc="Loading Graphs"):
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
            # PathRankerModel and HopColBERTReranker both expect these primary keys
            model_keys = {
                'question_input_ids', 'question_attention_mask', 
                'path_input_ids', 'path_attention_mask', 'candidate_mask',
                'hop_boundaries', 'hop_mask'  # HopColBERT specific
            }
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
                
                # Top K predictions
                top_k_preds_list = []
                for j in range(min(top_k, logits.size(1))):
                    idx = sorted_indices[i, j].item()
                    top_k_preds_list.append({
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
                graph_index = build_graph_index(sample_graph) if sample_graph else None
                
                # Format top K predicted paths with EXTRACTED entities from KG
                top_k_paths_with_entities = []
                top_k_answer_entities = []
                for pred in top_k_preds_list:
                    path_str = ""
                    final_entities = []
                    if pred['path'] and topic_entities:
                        if graph_index:
                            final_entities, path_entities = walk_path_entities(
                                pred['path'], topic_entities, graph_index
                            )
                            path_str = format_path_with_entities(pred['path'], path_entities)
                        if not path_str:
                            path_str = " -> ".join(topic_entities[:1] + pred['path']) + " -> ?"
                    top_k_paths_with_entities.append(path_str)
                    top_k_answer_entities.append(final_entities)
                
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
                    top_k_preds=top_k_preds_list,
                    top_k_paths_with_entities=top_k_paths_with_entities,
                    top_k_answer_entities=top_k_answer_entities,
                )
                
                results.append(result)
    
    if return_graph_lookup:
        return results, graph_id_to_graph
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
        
        # F1 Score: KGQA-style F1 (precision + recall on answer sets)
        f1_scores = []
        for r in llm_results:
            pred_entities = []
            if r.llm_answer:
                pred_entities = [
                    e.strip().lower()
                    for e in r.llm_answer.replace(', ', ',').split(',')
                    if e.strip()
                ]

            gt_entities = [gt.lower() for gt in r.gt_answers] if r.gt_answers else []

            if not pred_entities and not gt_entities:
                f1_scores.append(1.0)
                continue
            if not pred_entities or not gt_entities:
                f1_scores.append(0.0)
                continue

            matched_pred = 0
            matched_gt = set()
            for pred in pred_entities:
                for idx, gt in enumerate(gt_entities):
                    if idx in matched_gt:
                        continue
                    if pred in gt or gt in pred or pred == gt:
                        matched_pred += 1
                        matched_gt.add(idx)
                        break

            precision = matched_pred / len(pred_entities) if pred_entities else 0.0
            recall = matched_pred / len(gt_entities) if gt_entities else 0.0
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
        
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

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "for", "to", "from", "by", "with",
    "and", "or", "as", "is", "was", "were", "be", "been", "being", "do", "does",
    "did", "have", "has", "had", "that", "this", "these", "those", "it", "its",
    "their", "his", "her", "he", "she", "they", "them", "we", "you", "i", "me",
    "my", "our", "your", "who", "what", "which", "where", "when", "why",
}

_TOKEN_SYNONYMS = {
    "born": {"birth"},
    "birth": {"born"},
    "died": {"death", "die"},
    "death": {"died", "die"},
    "inspire": {"influence", "influenced", "inspired"},
    "inspired": {"influence", "influenced", "inspire"},
    "influence": {"inspired", "inspire", "influenced"},
    "spouse": {"married", "marriage", "wife", "husband"},
    "married": {"spouse", "marriage"},
    "wife": {"spouse", "married"},
    "husband": {"spouse", "married"},
    "author": {"writer", "novelist"},
    "writer": {"author", "novelist"},
    "novelist": {"author", "writer"},
    "actor": {"actress", "cast", "starring"},
    "actress": {"actor", "cast", "starring"},
    "director": {"filmmaker"},
    "composer": {"music", "musician"},
    "singer": {"vocalist"},
    "population": {"people", "inhabitant", "resident"},
    "area": {"size", "square"},
    "height": {"elevation", "altitude"},
    "elevation": {"height", "altitude"},
    "length": {"distance", "long"},
    "currency": {"money"},
    "language": {"tongue"},
    "university": {"college"},
    "college": {"university"},
    "company": {"corporation", "firm"},
    "country": {"nation"},
    "city": {"town"},
    "state": {"province"},
    "airport": {"airfield"},
    "film": {"movie"},
    "movie": {"film"},
    "song": {"track"},
    "album": {"record"},
    "utc": {"offset", "timezone"},
    "offset": {"utc", "timezone"},
    "timezone": {"utc", "offset"},
}

_TIMEZONE_OFFSETS = {
    "central time zone": -6,
    "mountain time zone": -7,
    "eastern time zone": -5,
    "pacific time zone": -8,
    "alaska time zone": -9,
    "hawaii-aleutian time zone": -10,
    "atlantic time zone": -4,
    "greenwich mean time": 0,
    "utc": 0,
}

_GENERIC_ENTITY_TERMS = {
    "country", "countries", "city", "cities", "state", "states", "province", "provinces",
    "nation", "language", "languages", "sport", "sports", "team", "teams", "person", "people",
    "organization", "organisation", "company", "corporation", "film", "movie", "album", "song",
    "book", "novel", "river", "mountain", "lake", "capital", "population", "timezone", "time zone",
}

_LLM_BACKOFF_LOCK = threading.Lock()
_LLM_BACKOFF_UNTIL = 0.0


def wait_for_llm_backoff() -> None:
    with _LLM_BACKOFF_LOCK:
        until = _LLM_BACKOFF_UNTIL
    now = time.time()
    if until > now:
        time.sleep(until - now)


def trigger_llm_backoff(delay_sec: float) -> None:
    if delay_sec <= 0:
        return
    with _LLM_BACKOFF_LOCK:
        global _LLM_BACKOFF_UNTIL
        _LLM_BACKOFF_UNTIL = max(_LLM_BACKOFF_UNTIL, time.time() + delay_sec)

def normalize_entity(text: str) -> str:
    """Normalize entity text for matching."""
    return re.sub(r'[^a-z0-9 ]+', ' ', text.lower()).strip()


def normalize_token(token: str) -> str:
    token = re.sub(r'[^a-z0-9]+', '', token.lower())
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def tokenize_text(text: str) -> set:
    """Tokenize text into normalized word set."""
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
    tokens = set()
    for tok in raw_tokens:
        tok = normalize_token(tok)
        if tok and tok not in _STOPWORDS:
            tokens.add(tok)
    return tokens


def expand_tokens(tokens: set) -> set:
    expanded = set(tokens)
    for tok in list(tokens):
        expanded.update(_TOKEN_SYNONYMS.get(tok, set()))
    return expanded


def token_overlap_score(tokens_a: set, tokens_b: set) -> int:
    score = 0
    for ta in tokens_a:
        for tb in tokens_b:
            if ta == tb or ta in tb or tb in ta:
                score += 1
                break
    return score


def is_topic_entity(entity: str, topic_entities: Optional[List[str]]) -> bool:
    if not entity or not topic_entities:
        return False
    key = normalize_entity(entity)
    return any(key == normalize_entity(t) for t in topic_entities if t)


def is_generic_entity(entity: str) -> bool:
    key = normalize_entity(entity)
    return key in _GENERIC_ENTITY_TERMS


def question_expects_multiple(question: str) -> bool:
    q = question.lower()
    if any(h in q for h in ("how many", "number of", "count of", "total number")):
        return False
    if any(h in q for h in ("list", "name", "give me", "what are", "what were", "who are", "who were",
                            "which are", "which were", "which ones", "what kinds of", "all ")):
        return True
    plural_targets = (
        "countries", "states", "cities", "rivers", "mountains", "lakes", "languages",
        "films", "movies", "songs", "albums", "books", "teams", "players", "actors",
        "members", "children", "parents", "spouses", "awards", "winners",
    )
    if any(h in q for h in ("which", "what", "who")) and any(t in q for t in plural_targets):
        return True
    return False


def split_answer_items(text: str) -> List[str]:
    cleaned = re.sub(r'(?i)^final answer\\s*:\\s*', '', text).strip()
    if not cleaned:
        return []
    lines = [ln.strip() for ln in re.split(r'[\\n;]+', cleaned) if ln.strip()]
    items: List[str] = []
    for line in lines:
        line = re.sub(r'^\\s*[-*\\d.]+\\s*', '', line).strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',') if p.strip()]
        items.extend(parts)
    return items


def extract_year(text: str) -> Optional[int]:
    match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)
    return int(match.group(1)) if match else None


def relation_tokens(rel: str) -> set:
    """Tokenize a relation string for relevance matching."""
    cleaned = re.sub(r'[^a-z0-9_\\.]+', ' ', rel.lower())
    tokens = re.split(r'[._\\s]+', cleaned)
    normalized = set()
    for tok in tokens:
        tok = normalize_token(tok)
        if tok and tok not in _STOPWORDS:
            normalized.add(tok)
    return normalized


def is_time_candidate(text: str) -> bool:
    if extract_year(text) is not None:
        return True
    return bool(re.search(r'\b(century|season|month|day|year)\b', text.lower()))


def is_number_candidate(text: str) -> bool:
    return bool(re.search(r'\d', text))


def extract_numeric_values(text: str) -> List[float]:
    values = []
    for match in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", text):
        raw = match.group(0).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        tail = text[match.end():match.end() + 20].lower()
        if "billion" in tail:
            value *= 1e9
        elif "million" in tail:
            value *= 1e6
        elif "thousand" in tail:
            value *= 1e3
        values.append(value)
    return values


def detect_comparator(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(at least|no less than|not less than|more than|over|above|minimum)\b", q):
        return "ge"
    if re.search(r"\b(at most|no more than|not more than|less than|under|below|maximum)\b", q):
        return "le"
    return "eq"


def is_year_like(value: float) -> bool:
    return float(value).is_integer() and 1000 <= value <= 2100


def detect_superlative_direction(question: str) -> Optional[str]:
    q = question.lower()
    if any(k in q for k in ("largest", "biggest", "highest", "most", "longest", "latest", "newest", "greatest", "maximum", "max")):
        return "max"
    if any(k in q for k in ("smallest", "lowest", "least", "shortest", "earliest", "oldest", "minimum", "min")):
        return "min"
    return None


def split_question_clauses(question: str) -> List[set]:
    text = f" {question.lower()} "
    for sep in (" and ", " also ", " both ", " as well as ", " along with "):
        text = text.replace(sep, " | ")
    parts = [p.strip() for p in text.split("|") if p.strip()]
    return [expand_tokens(tokenize_text(p)) for p in parts if p]


def build_entity_relation_map(graph: List[List[str]]) -> Tuple[Dict[str, List[Tuple[str, Any]]], Dict[str, List[Tuple[str, Any]]]]:
    from collections import defaultdict
    head_map = defaultdict(list)
    tail_map = defaultdict(list)
    for triple in graph:
        if len(triple) >= 3:
            head, rel, tail = triple[0], triple[1], triple[2]
            head_map[head].append((rel, tail))
            tail_map[tail].append((rel, head))
    return head_map, tail_map


def infer_numeric_relation_keywords(question: str) -> List[str]:
    q = question.lower()
    hints = []
    if "population" in q or "people" in q or "inhabitant" in q or "resident" in q:
        hints.extend(["population", "population_total", "population_density"])
    if "density" in q:
        hints.extend(["population_density", "density"])
    if "area" in q or "size" in q or "square" in q:
        hints.extend(["area", "area_total", "land_area"])
    if "elevation" in q or "height" in q or "altitude" in q or "tall" in q:
        hints.extend(["elevation", "height", "altitude"])
    if "length" in q or "distance" in q or "long" in q:
        hints.extend(["length", "distance"])
    if "gdp" in q or "gross domestic product" in q:
        hints.extend(["gdp", "gdp_nominal", "gdp_real", "gdp_per_capita"])
    if "budget" in q or "cost" in q or "spending" in q:
        hints.extend(["budget", "cost"])
    if "box office" in q or "gross revenue" in q:
        hints.extend(["box_office", "gross_revenue", "box_office_revenue"])
    if "rating" in q or "score" in q:
        hints.extend(["rating", "score", "rating_value"])
    if "rank" in q or "ranking" in q or "ranked" in q:
        hints.extend(["rank", "ranking"])
    if "episode" in q:
        hints.extend(["number_of_episodes", "episodes"])
    if "season" in q:
        hints.extend(["number_of_seasons", "seasons"])
    if "capacity" in q or "seats" in q or "seating" in q:
        hints.extend(["capacity", "seating_capacity", "seats"])
    if "runtime" in q or "duration" in q or "running time" in q or "running_time" in q:
        hints.extend(["runtime", "running_time", "duration"])
    if "utc" in q or "offset" in q:
        hints.extend(["utc_offset", "utc_offset_seconds", "utc_offset_minutes", "offset"])
    return list(dict.fromkeys(hints))
def extract_date_strings(text: str) -> List[str]:
    return re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)


def extract_year_strings(text: str) -> List[str]:
    return re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)


def date_relation_keywords(question: str) -> List[str]:
    q = question.lower()
    keywords = []
    if any(k in q for k in ("born", "birth")):
        keywords.extend(["date_of_birth", "birth_date"])
    if any(k in q for k in ("died", "death", "passed away", "dead")):
        keywords.extend(["date_of_death", "death_date"])
    if any(k in q for k in ("founded", "established", "formed", "inception", "created")):
        keywords.extend(["date_founded", "foundation_date", "inception"])
    if any(k in q for k in ("released", "premiered", "publication", "published", "opened", "debut")):
        keywords.extend(["release_date", "initial_release_date", "publication_date", "date_of_first_publication"])
    if any(k in q for k in ("started", "began", "begin", "first", "earliest", "latest")):
        keywords.extend(["start_date", "end_date", "date_started", "date_ended"])
    return keywords


def filter_candidates_by_date(
    question: str,
    candidates: List[Dict[str, Any]],
    graph: List[List[str]],
) -> List[Dict[str, Any]]:
    if not candidates or not graph:
        return []
    date_strings = extract_date_strings(question)
    year_strings = extract_year_strings(question)
    time_hint = bool(re.search(r'\b(when|year|date|born|birth|death|died|founded|established|released|premiered)\b', question.lower()))
    if not date_strings and not year_strings and not time_hint:
        return []

    rel_keywords = date_relation_keywords(question)
    q_tokens = expand_tokens(tokenize_text(question))
    head_map, tail_map = build_entity_relation_map(graph)

    def rel_matches(rel: str) -> bool:
        if rel_keywords and any(k in rel for k in rel_keywords):
            return True
        return token_overlap_score(q_tokens, relation_tokens(rel)) > 0

    matches = []
    candidate_years: Dict[str, List[int]] = {}
    for c in candidates:
        entity = c['entity']
        years = []
        rels = head_map.get(entity, [])
        for rel, tail in rels:
            if not rel_matches(rel):
                continue
            tail_str = str(tail)
            if date_strings and any(d in tail_str for d in date_strings):
                matches.append(c)
                break
            years.extend(int(y) for y in extract_year_strings(tail_str))
        else:
            for rel, head in tail_map.get(entity, []):
                if not rel_matches(rel):
                    continue
                head_str = str(head)
                years.extend(int(y) for y in extract_year_strings(head_str))
        if years:
            candidate_years[normalize_entity(entity)] = years

        if year_strings and years:
            if any(y in str(v) for y in year_strings for v in years):
                matches.append(c)

    if matches:
        return matches

    direction = detect_superlative_direction(question)
    if direction and candidate_years:
        best_value = None
        best_key = None
        for key, years in candidate_years.items():
            value = max(years) if direction == "max" else min(years)
            if best_value is None or (direction == "max" and value > best_value) or (direction == "min" and value < best_value):
                best_value = value
                best_key = key
        if best_key is not None:
            for c in candidates:
                if normalize_entity(c['entity']) == best_key:
                    return [c]

    return []


def extract_numeric_constraints(question: str) -> List[Dict[str, Any]]:
    q = question.lower()
    constraints = []
    phrase_map = {
        "long term unemployment rate": ["long_term_unemployment_rate", "unemployment_rate"],
        "unemployment rate": ["unemployment_rate", "long_term_unemployment_rate"],
        "calling code": ["calling_code", "dialing_code", "dialling_code", "phone_number", "telephone_code"],
        "dialing code": ["dialing_code", "calling_code", "phone_number", "telephone_code"],
        "dialling code": ["dialling_code", "calling_code", "phone_number", "telephone_code"],
        "phone code": ["phone_number", "calling_code", "dialing_code", "telephone_code"],
        "area code": ["area_code", "phone_number", "calling_code", "dialing_code"],
        "postal code": ["postal_code", "postal_codes", "zip_code"],
        "zip code": ["zip_code", "postal_code", "postal_codes"],
        "population": ["population"],
        "area": ["area", "area_total", "land_area"],
        "elevation": ["elevation", "height", "altitude"],
        "height": ["height", "elevation"],
        "length": ["length"],
        "gdp": ["gdp"],
        "budget": ["budget"],
        "box office": ["box_office", "gross_revenue", "box_office_revenue"],
        "rating": ["rating", "score", "rating_value"],
        "score": ["score", "rating"],
        "rank": ["rank", "ranking"],
        "number of episodes": ["number_of_episodes", "episodes"],
        "number of seasons": ["number_of_seasons", "seasons"],
    }
    comparator = detect_comparator(question)

    for phrase, rel_keywords in phrase_map.items():
        if phrase not in q:
            continue
        pattern = re.compile(rf"{re.escape(phrase)}[^0-9]{{0,20}}(?P<num>\d+(?:\.\d+)?)")
        for match in pattern.finditer(q):
            value = float(match.group("num"))
            constraints.append({
                "value": value,
                "relation_keywords": rel_keywords,
                "comparator": comparator,
            })
        pattern_rev = re.compile(rf"(?P<num>\d+(?:\.\d+)?)\D{{0,20}}{re.escape(phrase)}")
        for match in pattern_rev.finditer(q):
            value = float(match.group("num"))
            constraints.append({
                "value": value,
                "relation_keywords": rel_keywords,
                "comparator": comparator,
            })

    if not constraints:
        rel_keywords = infer_numeric_relation_keywords(question)
        values = extract_numeric_values(question)
        time_hint = bool(re.search(r'\b(when|year|date|born|birth|death|died)\b', q))
        for value in values:
            if time_hint and is_year_like(value):
                continue
            if rel_keywords:
                constraints.append({
                    "value": value,
                    "relation_keywords": rel_keywords,
                    "comparator": comparator,
                })

    return constraints


def extract_numbers_from_value(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    text = str(value).replace(",", "")
    return [float(m.group(0)) for m in re.finditer(r"-?\d+(?:\.\d+)?", text)]


def matches_numeric_constraint(value: float, constraint: Dict[str, Any]) -> bool:
    target = constraint["value"]
    comparator = constraint.get("comparator", "eq")
    if comparator == "ge":
        return value >= target
    if comparator == "le":
        return value <= target
    tol = max(0.05, 0.01 * max(abs(target), 1.0))
    return abs(value - target) <= tol


def filter_candidates_by_numeric(
    question: str,
    candidates: List[Dict[str, Any]],
    graph: List[List[str]],
) -> List[Dict[str, Any]]:
    if not candidates or not graph:
        return []
    constraints = extract_numeric_constraints(question)
    if not constraints:
        return []

    q_tokens = expand_tokens(tokenize_text(question))
    head_map, tail_map = build_entity_relation_map(graph)

    def rel_matches(rel: str, rel_keywords: List[str]) -> bool:
        if rel_keywords and any(k in rel for k in rel_keywords):
            return True
        return token_overlap_score(q_tokens, relation_tokens(rel)) > 0

    matches = []
    for c in candidates:
        entity = c["entity"]
        rels = head_map.get(entity, [])
        satisfied_all = True
        for constraint in constraints:
            rel_keywords = constraint.get("relation_keywords", [])
            found = False
            for rel, tail in rels:
                if not rel_matches(rel, rel_keywords):
                    continue
                for num in extract_numbers_from_value(tail):
                    if matches_numeric_constraint(num, constraint):
                        found = True
                        break
                if found:
                    break
            if not found:
                for rel, head in tail_map.get(entity, []):
                    if not rel_matches(rel, rel_keywords):
                        continue
                    for num in extract_numbers_from_value(head):
                        if matches_numeric_constraint(num, constraint):
                            found = True
                            break
                    if found:
                        break
            if not found:
                satisfied_all = False
                break
        if satisfied_all:
            matches.append(c)

    return matches


def filter_candidates_by_clauses(
    result: QAResult,
    candidates: List[Dict[str, Any]],
    graph: Optional[List[List[str]]] = None,
) -> List[Dict[str, Any]]:
    if not candidates or not result.top_k_preds:
        return []
    clauses = split_question_clauses(result.question)
    if len(clauses) < 2:
        return []
    topic_tokens = set()
    for ent in result.topic_entities or []:
        topic_tokens.update(tokenize_text(ent))

    clause_tokens = []
    for tokens in clauses:
        tokens = {t for t in tokens if t not in topic_tokens}
        if tokens:
            clause_tokens.append(tokens)

    if len(clause_tokens) < 2:
        return []

    graph_index = build_graph_index(graph) if graph else None
    clause_candidate_sets = []
    for tokens in clause_tokens:
        matched = set()
        for i, pred in enumerate(result.top_k_preds):
            path_relations = pred.get("path", []) if i < len(result.top_k_preds) else []
            if not path_relations:
                continue
            path_tokens = set()
            for rel in path_relations:
                path_tokens.update(relation_tokens(rel))
            if not (tokens & path_tokens):
                continue
            # Collect entities for this path
            if graph_index and result.topic_entities:
                for topic in result.topic_entities:
                    final_entities, _ = walk_path_entities(path_relations, [topic], graph_index)
                    for ent in final_entities:
                        matched.add(normalize_entity(ent))
            else:
                answer_lists = getattr(result, "top_k_answer_entities", []) or []
                if i < len(answer_lists):
                    for ent in answer_lists[i]:
                        matched.add(normalize_entity(ent))
        if matched:
            clause_candidate_sets.append(matched)

    if len(clause_candidate_sets) < 2:
        return []

    intersection = set.intersection(*clause_candidate_sets)
    if not intersection:
        return []

    filtered = []
    for c in candidates:
        if normalize_entity(c["entity"]) in intersection:
            filtered.append(c)
    return filtered


def filter_candidates_by_superlative(
    question: str,
    candidates: List[Dict[str, Any]],
    graph: List[List[str]],
) -> List[Dict[str, Any]]:
    direction = detect_superlative_direction(question)
    if not direction or not candidates or not graph:
        return []
    if re.search(r"\b(when|year|date|born|birth|died|death)\b", question.lower()):
        return []

    q_tokens = expand_tokens(tokenize_text(question))
    rel_keywords = infer_numeric_relation_keywords(question)
    head_map, _ = build_entity_relation_map(graph)

    candidate_values: Dict[str, float] = {}
    for c in candidates:
        entity = c["entity"]
        values = []
        for rel, tail in head_map.get(entity, []):
            rel_tokens = relation_tokens(rel)
            if rel_keywords:
                if not any(k in rel for k in rel_keywords) and token_overlap_score(q_tokens, rel_tokens) == 0:
                    continue
            elif token_overlap_score(q_tokens, rel_tokens) == 0:
                continue
            values.extend(extract_numbers_from_value(tail))
        if values:
            candidate_values[normalize_entity(entity)] = max(values) if direction == "max" else min(values)

    if not candidate_values:
        return []

    best_value = max(candidate_values.values()) if direction == "max" else min(candidate_values.values())
    best_keys = {k for k, v in candidate_values.items() if v == best_value}
    return [c for c in candidates if normalize_entity(c["entity"]) in best_keys]


def filter_candidates_by_timezone_offset(
    question: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    q = question.lower()
    if "utc" not in q and "offset" not in q and "time zone" not in q and "timezone" not in q:
        return []
    values = extract_numeric_values(question)
    if not values:
        return []
    matches = []
    for c in candidates:
        key = normalize_entity(c["entity"])
        offset = _TIMEZONE_OFFSETS.get(key)
        if offset is None:
            continue
        if any(abs(offset - v) <= 0.1 for v in values):
            matches.append(c)
    return matches


_WIKIDATA_CACHE_PATH = os.path.join(os.path.dirname(__file__), "wikidata_cache.json")
_WIKIDATA_CACHE = None
_WIKIDATA_LOCK = threading.Lock()


def load_wikidata_cache() -> Dict[str, str]:
    global _WIKIDATA_CACHE
    if _WIKIDATA_CACHE is not None:
        return _WIKIDATA_CACHE
    if os.path.exists(_WIKIDATA_CACHE_PATH):
        try:
            with open(_WIKIDATA_CACHE_PATH, "r") as f:
                _WIKIDATA_CACHE = json.load(f)
        except Exception:
            _WIKIDATA_CACHE = {}
    else:
        _WIKIDATA_CACHE = {}
    return _WIKIDATA_CACHE


def save_wikidata_cache(cache: Dict[str, str]) -> None:
    try:
        with open(_WIKIDATA_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def get_wikidata_description(entity: str) -> str:
    cache = load_wikidata_cache()
    key = normalize_entity(entity)
    if not key:
        return ""
    if key in cache:
        return cache[key]
    with _WIKIDATA_LOCK:
        cache = load_wikidata_cache()
        if key in cache:
            return cache[key]
        try:
            resp = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": entity,
                    "language": "en",
                    "format": "json",
                    "limit": 1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            description = ""
            if data.get("search"):
                description = data["search"][0].get("description", "") or ""
            cache[key] = description
            save_wikidata_cache(cache)
            time.sleep(0.1)
            return description
        except Exception:
            cache[key] = ""
            save_wikidata_cache(cache)
            return ""


def detect_type_hint(question: str) -> Optional[str]:
    q = question.lower()
    patterns = [
        ("actor", ["actor", "actress", "cast member", "starred"]),
        ("singer", ["singer", "vocalist"]),
        ("musician", ["musician", "band", "composer"]),
        ("author", ["author", "writer", "novelist", "poet"]),
        ("director", ["director"]),
        ("producer", ["producer"]),
        ("politician", ["politician", "president", "prime minister", "governor", "senator", "king", "queen", "monarch"]),
        ("athlete", ["athlete", "player", "footballer", "soccer", "basketball", "baseball", "tennis"]),
        ("company", ["company", "corporation", "firm", "enterprise"]),
        ("organization", ["organization", "organisation", "association", "agency"]),
        ("city", ["what city", "which city"]),
        ("country", ["what country", "which country", "what nation", "which nation"]),
        ("state", ["what state", "which state", "what province", "which province"]),
        ("film", ["what film", "which film", "what movie", "which movie"]),
        ("album", ["what album", "which album"]),
        ("song", ["what song", "which song", "what track", "which track"]),
        ("book", ["what book", "which book", "what novel", "which novel"]),
        ("river", ["what river", "which river"]),
        ("mountain", ["what mountain", "which mountain", "what mount", "which mount"]),
        ("lake", ["what lake", "which lake"]),
        ("language", ["what language", "which language"]),
        ("currency", ["what currency", "which currency"]),
        ("team", ["what team", "which team", "what club", "which club"]),
        ("university", ["what university", "which university", "what college", "which college"]),
        ("airport", ["what airport", "which airport"]),
        ("person", ["who", "which person", "what person"]),
    ]
    for type_key, phrases in patterns:
        if any(p in q for p in phrases):
            return type_key
    return None


def filter_candidates_by_type(
    question: str,
    candidates: List[Dict[str, Any]],
    max_candidates: int = 50,
) -> List[Dict[str, Any]]:
    type_hint = detect_type_hint(question)
    if not type_hint:
        return []

    desc_keywords = {
        "actor": ["actor", "actress"],
        "singer": ["singer", "vocalist"],
        "musician": ["musician", "band", "composer", "instrumentalist"],
        "author": ["author", "writer", "novelist", "poet"],
        "director": ["director", "film director"],
        "producer": ["producer"],
        "politician": ["politician", "president", "prime minister", "governor", "senator", "king", "queen", "monarch"],
        "athlete": ["athlete", "footballer", "basketball", "baseball", "tennis", "soccer", "player"],
        "company": ["company", "corporation", "firm", "business"],
        "organization": ["organization", "organisation", "agency", "association", "institution"],
        "city": ["city", "town", "municipality"],
        "country": ["country", "sovereign state", "nation"],
        "state": ["state", "province", "territory"],
        "film": ["film", "movie"],
        "album": ["album"],
        "song": ["song"],
        "book": ["book", "novel"],
        "river": ["river"],
        "mountain": ["mountain", "mount"],
        "lake": ["lake"],
        "language": ["language"],
        "currency": ["currency"],
        "team": ["team", "club", "sports team"],
        "university": ["university", "college", "school"],
        "airport": ["airport"],
        "person": ["person", "human"],
    }
    keywords = desc_keywords.get(type_hint, [])
    if not keywords:
        return []

    matches = []
    for c in candidates[:max_candidates]:
        if is_generic_entity(c["entity"]):
            continue
        if normalize_entity(c["entity"]) == type_hint:
            continue
        desc = get_wikidata_description(c["entity"]).lower()
        if any(k in desc for k in keywords):
            matches.append(c)
    return matches


def prepare_candidates(
    result: QAResult,
    graph: Optional[List[List[str]]] = None,
    max_candidates: int = 120,
) -> List[Dict[str, Any]]:
    """Collect and score answer candidates from top-K paths."""
    candidate_stats: Dict[str, Dict[str, Any]] = {}

    def update_candidate(
        entity: str,
        path_score: float,
        path_tokens: set,
        path_key: tuple,
        topic: Optional[str] = None,
        path_relations: Optional[List[str]] = None,
    ):
        key = normalize_entity(entity)
        if not key:
            return
        stats = candidate_stats.get(key)
        if not stats:
            stats = {
                'entity': entity,
                'path_score': path_score,
                'path_tokens': set(path_tokens),
                'path_set': set(),
                'path_token_sets': {},
                'topic_set': set(),
                'best_path': list(path_relations) if path_relations else [],
                'best_path_score': path_score,
            }
            candidate_stats[key] = stats
        if path_score > stats['path_score']:
            stats['path_score'] = path_score
        stats['path_tokens'].update(path_tokens)
        if path_key:
            stats['path_set'].add(path_key)
            stats['path_token_sets'].setdefault(path_key, set(path_tokens))
        if topic:
            stats['topic_set'].add(topic)
        if path_relations and path_score >= stats.get('best_path_score', 0.0):
            stats['best_path_score'] = path_score
            stats['best_path'] = list(path_relations)

    if graph and result.topic_entities:
        graph_index = build_graph_index(graph)
        for i, pred in enumerate(result.top_k_preds):
            path_relations = pred.get('path', []) if i < len(result.top_k_preds) else []
            if not path_relations:
                continue
            path_score = pred.get('score', 0.0)
            path_token_set = set()
            for rel in path_relations:
                path_token_set.update(relation_tokens(rel))
            path_key = tuple(path_relations)
            for topic in result.topic_entities:
                final_entities, _ = walk_path_entities(path_relations, [topic], graph_index)
                for entity in final_entities:
                    update_candidate(entity, path_score, path_token_set, path_key, topic, path_relations)
    else:
        answer_lists = getattr(result, 'top_k_answer_entities', []) or []
        for i, entities in enumerate(answer_lists):
            if not entities:
                continue
            path_score = result.top_k_preds[i]['score'] if i < len(result.top_k_preds) else 0.0
            path_relations = result.top_k_preds[i].get('path', []) if i < len(result.top_k_preds) else []
            path_token_set = set()
            for rel in path_relations or []:
                path_token_set.update(relation_tokens(rel))
            path_key = tuple(path_relations) if path_relations else ()
            for entity in entities:
                update_candidate(entity, path_score, path_token_set, path_key, None, path_relations)

    candidates = []
    for stats in candidate_stats.values():
        stats['path_count'] = len(stats['path_set']) if stats['path_set'] else 1
        stats['topic_coverage'] = len(stats['topic_set']) if stats['topic_set'] else 1
        candidates.append(stats)

    if not candidates:
        return []

    q_tokens = expand_tokens(tokenize_text(result.question))
    q_lower = result.question.lower()
    time_related = any(h in q_lower for h in ("when", "what year", "year", "date", "season", "month", "day"))
    count_related = any(h in q_lower for h in ("how many", "number of", "count of", "total number"))
    conjunction = any(h in q_lower for h in (" and ", " also ", " both ", " as well as ", " along with "))
    focus_keywords = {
        normalize_token(k) for k in (
            "near", "born", "birth", "death", "died", "language", "country", "city",
            "movie", "film", "album", "song", "team", "river", "mountain", "capital",
            "president", "actor", "author", "book", "novel", "state", "university",
            "college", "party", "religion", "currency", "population", "area", "governor",
            "spouse", "child", "children", "parent", "married", "winner", "championship",
            "superbowl", "season", "year", "date", "first", "last", "largest", "smallest",
        )
    }
    focus_tokens = q_tokens & focus_keywords
    clause_tokens_list = split_question_clauses(result.question) if conjunction else []

    head_map, tail_map = build_entity_relation_map(graph) if graph else ({}, {})
    numeric_filtered = filter_candidates_by_numeric(result.question, candidates, graph) if graph else []
    date_filtered = filter_candidates_by_date(result.question, candidates, graph) if graph else []
    superlative_filtered = filter_candidates_by_superlative(result.question, candidates, graph) if graph else []
    timezone_filtered = filter_candidates_by_timezone_offset(result.question, candidates)

    numeric_keys = {normalize_entity(c['entity']) for c in numeric_filtered}
    date_keys = {normalize_entity(c['entity']) for c in date_filtered}
    superlative_keys = {normalize_entity(c['entity']) for c in superlative_filtered}
    timezone_keys = {normalize_entity(c['entity']) for c in timezone_filtered}

    for c in candidates:
        ent_tokens = tokenize_text(c['entity'])
        ent_overlap = token_overlap_score(q_tokens, ent_tokens)
        path_overlap = token_overlap_score(q_tokens, c.get('path_tokens', set()))
        focus_overlap = token_overlap_score(focus_tokens, c.get('path_tokens', set()))
        generic_penalty = -0.4 if is_generic_entity(c['entity']) else 0.0
        overlap_penalty = 0.0
        if ent_tokens and ent_tokens.issubset(q_tokens) and len(ent_tokens) <= 2:
            if not is_number_candidate(c['entity']) and not is_time_candidate(c['entity']):
                overlap_penalty = -0.2
        length_bonus = 0.02 * min(len(ent_tokens), 5)
        time_bonus = 0.15 if time_related and is_time_candidate(c['entity']) else 0.0
        time_penalty = -0.1 if (not time_related and is_time_candidate(c['entity'])) else 0.0
        count_bonus = 0.1 if count_related and is_number_candidate(c['entity']) else 0.0
        count_penalty = -0.05 if (not count_related and is_number_candidate(c['entity'])) else 0.0
        path_overlap_score = min(path_overlap, 4)
        focus_overlap_score = min(focus_overlap, 3)
        path_penalty = -0.2 if path_overlap == 0 else 0.0
        path_count_bonus = 0.08 * max(0, c.get('path_count', 1) - 1)
        topic_bonus = 0.2 * max(0, c.get('topic_coverage', 1) - 1) if conjunction else 0.0

        context_overlap = 0
        topic_graph_coverage = 0
        if head_map:
            context_tokens = set()
            for rel, _ in head_map.get(c['entity'], []):
                context_tokens.update(relation_tokens(rel))
            for rel, _ in tail_map.get(c['entity'], []):
                context_tokens.update(relation_tokens(rel))
            context_overlap = token_overlap_score(q_tokens, context_tokens)
            if result.topic_entities:
                topic_links = set()
                for rel, other in head_map.get(c['entity'], []):
                    if other in result.topic_entities:
                        topic_links.add(other)
                for rel, other in tail_map.get(c['entity'], []):
                    if other in result.topic_entities:
                        topic_links.add(other)
                topic_graph_coverage = len(topic_links)

        clause_bonus = 0.0
        if clause_tokens_list and c.get('path_token_sets'):
            clause_hits = 0
            for clause_tokens in clause_tokens_list:
                if any(token_overlap_score(clause_tokens, pts) > 0 for pts in c.get('path_token_sets', {}).values()):
                    clause_hits += 1
            if clause_hits >= 2:
                clause_bonus = 0.2

        key = normalize_entity(c['entity'])
        constraint_bonus = 0.0
        if key in numeric_keys:
            constraint_bonus += 0.35
        if key in date_keys:
            constraint_bonus += 0.35
        if key in superlative_keys:
            constraint_bonus += 0.3
        if key in timezone_keys:
            constraint_bonus += 0.35
        constraint_bonus = min(constraint_bonus, 0.8)
        context_bonus = 0.08 * min(context_overlap, 3)
        topic_graph_bonus = 0.2 * max(0, topic_graph_coverage - 1) if conjunction else 0.0
        topic_penalty = -0.35 if is_topic_entity(c['entity'], result.topic_entities) else 0.0

        c['ent_overlap'] = ent_overlap
        c['path_overlap'] = path_overlap
        c['focus_overlap'] = focus_overlap
        c['context_overlap'] = context_overlap
        c['is_topic'] = topic_penalty < 0
        c['rank_score'] = (
            c['path_score']
            + 0.35 * path_overlap_score
            + 0.2 * focus_overlap_score
            + 0.05 * ent_overlap
            + context_bonus
            + path_penalty
            + time_bonus
            + time_penalty
            + count_bonus
            + count_penalty
            + length_bonus
            + path_count_bonus
            + topic_bonus
            + topic_graph_bonus
            + clause_bonus
            + constraint_bonus
            + topic_penalty
            + generic_penalty
            + overlap_penalty
        )

    candidates.sort(key=lambda c: c['rank_score'], reverse=True)
    if max_candidates and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    return candidates


def select_best_candidate(
    question: str,
    candidates: List[Dict[str, Any]],
    topic_entities: Optional[List[str]] = None,
) -> str:
    """Pick a reasonable fallback candidate based on question cues."""
    if not candidates:
        return ""

    filtered = [c for c in candidates if not is_topic_entity(c['entity'], topic_entities)]
    candidates = filtered or candidates

    q = question.lower()
    earliest_hints = ("first", "earliest", "oldest", "lowest", "smallest")
    latest_hints = ("last", "latest", "most recent", "newest", "highest", "largest", "biggest")
    when_hints = ("when", "what year", "year", "date", "season")
    count_hints = ("how many", "number of", "count of", "total number")

    year_candidates = []
    number_candidates = []
    for c in candidates:
        year = extract_year(c['entity'])
        if year is not None:
            year_candidates.append((year, c))
        if re.search(r'\d', c['entity']):
            number_candidates.append(c)

    if year_candidates and any(h in q for h in earliest_hints):
        return min(year_candidates, key=lambda x: x[0])[1]['entity']
    if year_candidates and any(h in q for h in latest_hints):
        return max(year_candidates, key=lambda x: x[0])[1]['entity']
    if year_candidates and any(h in q for h in when_hints):
        return max(year_candidates, key=lambda x: x[1].get('rank_score', 0.0))[1]['entity']
    if number_candidates and any(h in q for h in count_hints):
        return max(number_candidates, key=lambda x: x.get('rank_score', 0.0))['entity']

    return max(candidates, key=lambda x: x.get('rank_score', 0.0))['entity']


def parse_llm_answer(
    raw_response: str,
    candidates: List[Dict[str, Any]],
    question: str,
    prefer_llm: bool = False,
    allow_index: bool = False,
    allow_multiple: bool = False,
    topic_entities: Optional[List[str]] = None,
) -> str:
    """Normalize LLM output into a single candidate entity."""
    if not raw_response:
        return select_best_candidate(question, candidates, topic_entities)

    text = raw_response.strip()
    if not text:
        return select_best_candidate(question, candidates, topic_entities)
    if "Final Answer:" in text:
        text = text.split("Final Answer:")[-1].strip()
    if not text:
        return select_best_candidate(question, candidates, topic_entities)
    text_line = text.splitlines()[0].strip()

    best_candidate = max(candidates, key=lambda x: x.get('rank_score', 0.0)) if candidates else None
    best_non_topic = None
    best_non_generic = None
    if candidates:
        non_topic = [c for c in candidates if not is_topic_entity(c['entity'], topic_entities)]
        if non_topic:
            best_non_topic = max(non_topic, key=lambda x: x.get('rank_score', 0.0))
        non_generic = [c for c in candidates if not is_generic_entity(c['entity'])]
        if non_generic:
            best_non_generic = max(non_generic, key=lambda x: x.get('rank_score', 0.0))

    if candidates and allow_index:
        if text_line.lstrip().startswith("#"):
            idxs = [int(n) - 1 for n in re.findall(r'\d{1,3}', text_line)]
            picked = [candidates[i]['entity'] for i in idxs if 0 <= i < len(candidates)]
            if picked:
                if allow_multiple and len(picked) > 1:
                    deduped = list(dict.fromkeys(picked))
                    return ", ".join(deduped)
                return picked[0]
        else:
            num_match = re.match(r'^\s*(?:answer[:\s]*)?#?\s*(\d{1,3})\s*[\).]?\s*$', text_line, re.IGNORECASE)
            if num_match:
                idx = int(num_match.group(1)) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]['entity']

    if candidates:
        items = split_answer_items(text)
        if allow_multiple and len(items) == 1 and " and " in items[0]:
            and_parts = [p.strip() for p in items[0].split(" and ") if p.strip()]
            if len(and_parts) > 1:
                and_hits = 0
                for part in and_parts:
                    part_norm = normalize_entity(part)
                    if not part_norm:
                        continue
                    for c in candidates:
                        c_norm = normalize_entity(c['entity'])
                        if c_norm and (part_norm == c_norm or part_norm in c_norm or c_norm in part_norm):
                            and_hits += 1
                            break
                if and_hits >= 2:
                    items = and_parts
        matches = []
        for item in items:
            item_norm = normalize_entity(item)
            if not item_norm:
                continue
            for c in candidates:
                c_norm = normalize_entity(c['entity'])
                if not c_norm:
                    continue
                if item_norm == c_norm or item_norm in c_norm or c_norm in item_norm:
                    matches.append(c)
                    break
        if not matches:
            norm_text = normalize_entity(text)
            for c in candidates:
                c_norm = normalize_entity(c['entity'])
                if c_norm and c_norm in norm_text:
                    matches.append(c)
        if allow_multiple and len(matches) > 1:
            ordered = []
            seen = set()
            for item in items:
                item_norm = normalize_entity(item)
                for c in matches:
                    if normalize_entity(c['entity']) == item_norm or item_norm in normalize_entity(c['entity']):
                        if c['entity'] not in seen and not is_topic_entity(c['entity'], topic_entities):
                            ordered.append(c['entity'])
                            seen.add(c['entity'])
                        break
            if ordered:
                return ", ".join(ordered)
            ranked = [
                c['entity'] for c in sorted(matches, key=lambda x: x.get('rank_score', 0.0), reverse=True)
                if not is_topic_entity(c['entity'], topic_entities)
            ]
            if ranked:
                return ", ".join(ranked)

        if matches:
            chosen = max(matches, key=lambda x: x.get('rank_score', 0.0))
            if is_topic_entity(chosen['entity'], topic_entities) and best_non_topic:
                chosen = best_non_topic
            if is_generic_entity(chosen['entity']) and best_non_generic:
                chosen = best_non_generic
            if prefer_llm or not best_candidate:
                return chosen['entity']
            if best_candidate:
                score_gap = best_candidate.get('rank_score', 0.0) - chosen.get('rank_score', 0.0)
                if score_gap > 0.2 or (chosen.get('ent_overlap', 0) == 0 and best_candidate.get('ent_overlap', 0) > 0):
                    return best_candidate['entity']
            return chosen['entity']

    parts = [p.strip() for p in re.split(r'[;,]', text_line) if p.strip()]
    if len(parts) > 1:
        if allow_multiple:
            return ", ".join(parts)
        return parts[0]

    if best_candidate:
        chosen = best_candidate
        if is_topic_entity(chosen['entity'], topic_entities) and best_non_topic:
            chosen = best_non_topic
        if is_generic_entity(chosen['entity']) and best_non_generic:
            chosen = best_non_generic
        return chosen['entity']
    return select_best_candidate(question, candidates, topic_entities) or text_line


def call_llm_api(
    prompt: str,
    api_url: str = "https://game.agaii.org/llm/v1",
    model: str = "default",
    max_tokens: int = 128,
    temperature: float = 0.1,
    retries: int = 5,
    backoff_sec: float = 3.0,
) -> str:
    """Call the LLM API."""
    for attempt in range(retries):
        wait_for_llm_backoff()
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
        except requests.HTTPError as e:
            print(f"LLM API error: {e}")
            status = getattr(e.response, "status_code", None)
            if status and status >= 500 and attempt < retries - 1:
                trigger_llm_backoff(backoff_sec * (attempt + 1))
                time.sleep(backoff_sec * (attempt + 1))
                continue
            return ""
        except Exception as e:
            print(f"LLM API error: {e}")
            if attempt < retries - 1:
                trigger_llm_backoff(backoff_sec * (attempt + 1))
                time.sleep(backoff_sec * (attempt + 1))
                continue
            return ""


def build_qa_prompt(
    result: QAResult,
    graph: Optional[List[List[str]]] = None,
    max_paths: int = 10,
    include_candidates: bool = False,
    max_candidates_list: int = 15,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build a prompt for the LLM to answer using retrieved paths and candidates."""
    topic_str = ', '.join(result.topic_entities[:3]) if result.topic_entities else 'Unknown'

    retrieved_info = []
    for i, path_with_entities in enumerate(result.top_k_paths_with_entities[:max_paths], 1):
        if path_with_entities:
            retrieved_info.append(f"  {i}. {path_with_entities}")
        else:
            retrieved_info.append(f"  {i}. N/A")

    candidates = prepare_candidates(result, graph=graph, max_candidates=120)
    candidate_block = ""
    if include_candidates and candidates:
        lines = []
        for i, c in enumerate(candidates[:max_candidates_list], 1):
            lines.append(f"  {i}. {c['entity']}")
        candidate_block = "\n".join(lines)

    format_block = "Format:\nFinal Answer: <entity>"
    if candidate_block:
        format_block += "\nFinal Answer: #<number> (if selecting from the candidate list)"

    prompt = (
        "You are a Knowledge Graph Question Answering system. Use the retrieved paths.\n\n"
        f"Question: {result.question}\n"
        f"Topic Entity: {topic_str}\n\n"
        "Retrieved Paths (with entities):\n"
        f"{chr(10).join(retrieved_info)}\n\n"
        + ("Candidate Answers (choose ONE):\n" + candidate_block + "\n\n" if candidate_block else "")
        + "Rules:\n"
        "1. Apply all constraints in the question.\n"
        "2. Answer with the END entity of a path.\n"
        "3. Copy answers exactly as shown in paths/candidates.\n"
        "4. If the question asks for multiple, return a comma-separated list.\n"
        "5. Do not answer with the topic entity unless the question asks for it.\n"
        "6. Do not introduce entities not shown in paths/candidates.\n\n"
        + format_block
    )

    return prompt, candidates



def run_llm_qa_batch(
    results: List[QAResult],
    api_url: str = "https://game.agaii.org/llm/v1",
    model: str = "default",
    batch_size: int = 10,
    max_workers: int = 5,
    dataset: str = "webqsp",
    graph_lookup: Optional[Dict[str, List[List[str]]]] = None,
    include_candidates: bool = False,
    max_candidates_list: int = 15,
) -> List[QAResult]:
    """Run LLM QA on results in batches."""
    
    def process_one(result: QAResult) -> QAResult:
        graph = graph_lookup.get(result.id, []) if graph_lookup else []
        graph_for_candidates = graph if graph else None
        max_paths = len(result.top_k_paths_with_entities) or 10
        prompt, candidates = build_qa_prompt(
            result,
            graph=graph_for_candidates,
            max_paths=max_paths,
            include_candidates=include_candidates,
            max_candidates_list=max_candidates_list,
        )
        raw_response = call_llm_api(prompt, api_url=api_url, model=model)

        expects_multiple = question_expects_multiple(result.question)
        fallback_answer = ""
        if result.top_k_answer_entities:
            first_path_answers = result.top_k_answer_entities[0]
            if first_path_answers:
                fallback_answer = first_path_answers[0]
        if not fallback_answer:
            fallback_answer = select_best_candidate(
                result.question,
                candidates,
                topic_entities=result.topic_entities,
            )

        if not raw_response:
            answer = fallback_answer
        else:
            answer = parse_llm_answer(
                raw_response,
                candidates,
                result.question,
                prefer_llm=(dataset == "cwq"),
                allow_index=include_candidates,
                allow_multiple=expects_multiple,
                topic_entities=result.topic_entities,
            )

        if answer and is_topic_entity(answer, result.topic_entities):
            non_topic_candidates = [c for c in candidates if not is_topic_entity(c['entity'], result.topic_entities)]
            if non_topic_candidates:
                answer = non_topic_candidates[0]['entity']

        if answer and is_generic_entity(answer):
            non_generic_candidates = [c for c in candidates if not is_generic_entity(c['entity'])]
            if non_generic_candidates:
                best_non_generic = max(non_generic_candidates, key=lambda x: x.get('rank_score', 0.0))
                answer = best_non_generic['entity']

        if expects_multiple and (not answer or "," not in answer):
            multi_answers: List[str] = []
            if result.top_k_answer_entities:
                top_answers = [
                    a for a in result.top_k_answer_entities[0]
                    if a and not is_topic_entity(a, result.topic_entities)
                ]
                if len(top_answers) > 1:
                    multi_answers = top_answers[:5]
            if not multi_answers and candidates:
                multi_answers = [
                    c['entity'] for c in candidates
                    if not is_topic_entity(c['entity'], result.topic_entities)
                ][:5]
            if multi_answers:
                answer = ", ".join(multi_answers)

        if graph and answer and ("," not in answer):
            timezone_filtered = filter_candidates_by_timezone_offset(result.question, candidates)
            if timezone_filtered:
                normalized_answer = normalize_entity(answer)
                if not any(normalize_entity(c['entity']) == normalized_answer for c in timezone_filtered):
                    answer = max(timezone_filtered, key=lambda x: x.get('rank_score', 0.0))['entity']

            date_filtered = filter_candidates_by_date(result.question, candidates, graph)
            if date_filtered:
                normalized_answer = normalize_entity(answer)
                if not any(normalize_entity(c['entity']) == normalized_answer for c in date_filtered):
                    answer = max(date_filtered, key=lambda x: x.get('rank_score', 0.0))['entity']

            numeric_filtered = filter_candidates_by_numeric(result.question, candidates, graph)
            if numeric_filtered:
                normalized_answer = normalize_entity(answer)
                if not any(normalize_entity(c['entity']) == normalized_answer for c in numeric_filtered):
                    answer = max(numeric_filtered, key=lambda x: x.get('rank_score', 0.0))['entity']

            clause_filtered = filter_candidates_by_clauses(result, candidates, graph)
            if clause_filtered:
                normalized_answer = normalize_entity(answer)
                if not any(normalize_entity(c['entity']) == normalized_answer for c in clause_filtered):
                    answer = max(clause_filtered, key=lambda x: x.get('rank_score', 0.0))['entity']

            superlative_filtered = filter_candidates_by_superlative(result.question, candidates, graph)
            if superlative_filtered:
                normalized_answer = normalize_entity(answer)
                if not any(normalize_entity(c['entity']) == normalized_answer for c in superlative_filtered):
                    answer = max(superlative_filtered, key=lambda x: x.get('rank_score', 0.0))['entity']

            type_filtered = filter_candidates_by_type(result.question, candidates)
            if type_filtered:
                normalized_answer = normalize_entity(answer)
                if not any(normalize_entity(c['entity']) == normalized_answer for c in type_filtered):
                    answer = max(type_filtered, key=lambda x: x.get('rank_score', 0.0))['entity']

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
    retrieval_output = run_path_retrieval(
        model,
        test_path,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        device=device,
        limit=args.limit,
        graph_data_path=args.graph_data,
        top_k=args.llm_top_k,
        return_graph_lookup=args.run_llm_qa,
    )
    if args.run_llm_qa:
        results, graph_lookup = retrieval_output
    else:
        results = retrieval_output
        graph_lookup = {}
    
    # Run LLM QA if enabled
    if args.run_llm_qa:
        results = run_llm_qa_batch(
            results,
            api_url=args.llm_api_url,
            model=args.llm_model,
            max_workers=args.llm_workers,
            dataset=args.dataset,
            graph_lookup=graph_lookup,
            include_candidates=args.llm_include_candidates,
            max_candidates_list=args.llm_max_candidates_list,
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
    parser.add_argument('--model_type', type=str, default='hop_colbert',
                        choices=['dcdr', 'biencoder', 'hop_colbert'])
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
    parser.add_argument('--llm_workers', type=int, default=20)
    parser.add_argument('--llm_top_k', type=int, default=15,
                        help='Number of top paths to use for LLM context')
    parser.add_argument('--llm_include_candidates', action='store_true',
                        help='Include candidate list in LLM prompt')
    parser.add_argument('--llm_max_candidates_list', type=int, default=15,
                        help='Max candidate list length when included in prompt')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs_complete_qa')
    
    args = parser.parse_args()
    main(args)
