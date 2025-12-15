"""
Evaluation script with Entity Retrieval Integration.

This script:
1. Uses the model to generate relation paths
2. Uses FastEntityIdentifier to identify topic entities from questions
3. Uses EntityRetriever to retrieve answer entities from KG using generated relations
4. Computes end-to-end evaluation metrics
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import entity components
from entity_identifier import FastEntityIdentifier
from entity_retriever import EntityRetriever, MultiHopEntityRetriever


def load_evaluation_results(path: str) -> Dict:
    """Load pre-computed evaluation results with predicted relations."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_parquet_data(path: str) -> List[Dict]:
    """Load test data from parquet file."""
    import pandas as pd
    df = pd.read_parquet(path)
    samples = []
    for _, row in df.iterrows():
        # Parse graph if it's a string
        graph = row.get('graph', [])
        if isinstance(graph, str):
            try:
                graph = json.loads(graph)
            except:
                graph = []
        
        # Parse paths if it's a string
        paths = row.get('paths', [])
        if isinstance(paths, str):
            try:
                paths = json.loads(paths)
            except:
                paths = []
        
        sample = {
            'id': row.get('id', ''),
            'question': row.get('question', ''),
            'answer': row.get('answer', []),
            'q_entity': row.get('q_entity', []),
            'a_entity': row.get('a_entity', []),
            'graph': graph,
            'paths': paths
        }
        samples.append(sample)
    return samples


def parse_relation_chain(relation_chain: str) -> List[str]:
    """Parse a relation chain string into list of relations."""
    if not relation_chain:
        return []
    return [r.strip() for r in relation_chain.split(" -> ") if r.strip()]


def calculate_entity_hits_at_k(
    retrieved_entities_list: List[Set[str]], 
    ground_truth_entities: List[Set[str]], 
    k: int = 1
) -> float:
    """
    Calculate Hits@K for entity retrieval.
    
    A hit occurs if any of the top-k retrieved entities matches any ground truth entity.
    """
    hits = 0
    total = 0
    
    for retrieved, gt in zip(retrieved_entities_list, ground_truth_entities):
        if not gt:
            continue
        total += 1
        
        # Check if any retrieved entity matches ground truth
        retrieved_list = list(retrieved)[:k] if len(retrieved) > k else list(retrieved)
        if any(r.lower() in {g.lower() for g in gt} for r in retrieved_list):
            hits += 1
    
    return hits / total if total > 0 else 0.0


def calculate_entity_recall(
    retrieved_entities_list: List[Set[str]], 
    ground_truth_entities: List[Set[str]]
) -> Dict[str, float]:
    """Calculate entity recall metrics."""
    total_gt = 0
    total_recalled = 0
    exact_matches = 0
    partial_matches = 0
    samples_counted = 0
    
    for retrieved, gt in zip(retrieved_entities_list, ground_truth_entities):
        if not gt:
            continue
        samples_counted += 1
        total_gt += len(gt)
        
        # Normalize for comparison
        gt_lower = {g.lower() for g in gt}
        retrieved_lower = {r.lower() for r in retrieved}
        
        # Count recalled entities
        recalled = len(gt_lower & retrieved_lower)
        total_recalled += recalled
        
        # Check match type
        if gt_lower == retrieved_lower:
            exact_matches += 1
        elif recalled > 0:
            partial_matches += 1
    
    return {
        'entity_recall': total_recalled / total_gt if total_gt > 0 else 0.0,
        'exact_match_rate': exact_matches / samples_counted if samples_counted > 0 else 0.0,
        'partial_match_rate': partial_matches / samples_counted if samples_counted > 0 else 0.0,
        'total_samples': samples_counted
    }


def retrieve_entities_with_relations(
    question: str,
    predicted_relations: List[str],
    graph_triples: List,
    entity_identifier: FastEntityIdentifier,
    entity_retriever: EntityRetriever,
    vocab: Dict[str, str] = None
) -> Set[str]:
    """
    Retrieve answer entities using predicted relations.
    
    Args:
        question: Natural language question
        predicted_relations: List of predicted relation chains
        graph_triples: Knowledge graph triples for this question
        entity_identifier: Entity identifier instance
        entity_retriever: Entity retriever instance
        vocab: Optional vocabulary for entity matching
        
    Returns:
        Set of retrieved entity names
    """
    retrieved_entities = set()
    
    # Build index for the graph
    entity_retriever.build_index(graph_triples)
    
    # Get topic entities from question
    topic_entities = entity_identifier.identify(question)
    
    # If no topic entity identified, try using the first entity in the graph
    if not topic_entities:
        # Extract unique subject entities from graph
        subjects = set()
        for triple in graph_triples:
            if isinstance(triple, dict):
                subj = triple.get('subject', triple.get('h', ''))
            elif isinstance(triple, (list, tuple)) and len(triple) >= 1:
                subj = triple[0]
            else:
                continue
            if subj:
                subjects.add(subj)
        
        # Use first subject as topic entity
        if subjects:
            topic_entities = [list(subjects)[0]]
    
    # For each topic entity and predicted relation chain
    for topic_entity in topic_entities:
        for relation_chain in predicted_relations:
            relations = parse_relation_chain(relation_chain)
            if not relations:
                continue
            
            # Strategy 1: Try the full relation chain
            entities = entity_retriever.retrieve(topic_entity, relations)
            retrieved_entities.update(entities)
            
            # Strategy 2: Try progressively shorter prefixes of the relation chain
            # This handles cases where the model generates longer paths
            for prefix_len in range(len(relations) - 1, 0, -1):
                prefix_relations = relations[:prefix_len]
                prefix_entities = entity_retriever.retrieve(topic_entity, prefix_relations)
                retrieved_entities.update(prefix_entities)
            
            # Strategy 3: Try each individual relation (single-hop)
            for rel in relations:
                single_hop_entities = entity_retriever.retrieve(topic_entity, [rel])
                retrieved_entities.update(single_hop_entities)
            
            # Strategy 4: Try combinations skipping intermediate relations
            if len(relations) >= 2:
                # Try first and last relation
                skip_entities = entity_retriever.retrieve(topic_entity, [relations[0], relations[-1]])
                retrieved_entities.update(skip_entities)
            
            # Limit retrieved entities to avoid noise
            if len(retrieved_entities) > 50:
                break
        
        if len(retrieved_entities) > 50:
            break
    
    # Filter out obviously wrong entities (like m.xxx IDs without labels)
    filtered_entities = set()
    for entity in retrieved_entities:
        # Skip mid-style identifiers that are just IDs
        if entity.startswith('m.') and len(entity) < 15:
            continue
        # Skip very long strings (likely not proper entities)
        if len(entity) > 100:
            continue
        filtered_entities.add(entity)
    
    return filtered_entities if filtered_entities else retrieved_entities


def evaluate_with_entity_retrieval(
    eval_results_path: str,
    test_data_path: str = None,
    vocab_path: str = None,
    max_examples: int = None
) -> Dict[str, Any]:
    """
    Evaluate model predictions with entity retrieval.
    
    Args:
        eval_results_path: Path to evaluation results JSON with predictions
        test_data_path: Path to test data parquet file (for graph triples)
        vocab_path: Path to vocabulary JSON file
        max_examples: Maximum examples to evaluate
        
    Returns:
        Dictionary with evaluation metrics and examples
    """
    print("=" * 70)
    print("EVALUATION WITH ENTITY RETRIEVAL")
    print("=" * 70)
    
    # Load evaluation results with predictions
    print(f"\nLoading evaluation results from: {eval_results_path}")
    eval_data = load_evaluation_results(eval_results_path)
    examples = eval_data.get('examples', [])
    
    # Load test data for graph triples
    if test_data_path:
        print(f"Loading test data from: {test_data_path}")
        test_samples = load_parquet_data(test_data_path)
        # Create lookup by question
        test_lookup = {s['question']: s for s in test_samples}
    else:
        test_lookup = {}
    
    # Load vocabulary if provided
    vocab = None
    if vocab_path and os.path.exists(vocab_path):
        print(f"Loading vocabulary from: {vocab_path}")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
    
    # Initialize components
    print("\nInitializing entity identifier and retriever...")
    entity_identifier = FastEntityIdentifier()
    # Load vocabulary into identifier if available
    if vocab_path and os.path.exists(vocab_path):
        entity_identifier.load_vocabulary(vocab_path)
    entity_retriever = MultiHopEntityRetriever()
    
    # Limit examples if needed
    if max_examples:
        examples = examples[:max_examples]
    
    print(f"\nEvaluating {len(examples)} examples...")
    print("=" * 70)
    
    # Collect metrics
    all_retrieved = []
    all_ground_truth = []
    all_results = []
    
    # Performance tracking
    total_time = 0
    entity_id_time = 0
    retrieval_time = 0
    
    for i, example in enumerate(examples):
        question = example.get('question', '')
        predicted = example.get('predicted', [])
        gt_answers = example.get('answer', [])
        gt_entities = example.get('answer_entities', gt_answers)
        
        # Normalize ground truth entities
        if isinstance(gt_entities, str):
            gt_entities = [gt_entities]
        gt_entity_set = set(gt_entities) if gt_entities else set()
        
        # Get graph triples from test data
        test_sample = test_lookup.get(question, {})
        graph_triples = test_sample.get('graph', [])
        
        if not graph_triples and 'graph' in example:
            graph_triples = example.get('graph', [])
        
        # Time entity identification
        t0 = time.time()
        topic_entities = entity_identifier.identify(question)
        entity_id_time += time.time() - t0
        
        # Time entity retrieval
        t0 = time.time()
        retrieved = retrieve_entities_with_relations(
            question=question,
            predicted_relations=predicted,
            graph_triples=graph_triples,
            entity_identifier=entity_identifier,
            entity_retriever=entity_retriever,
            vocab=vocab
        )
        retrieval_time += time.time() - t0
        
        all_retrieved.append(retrieved)
        all_ground_truth.append(gt_entity_set)
        
        # Get ground truth relation paths from the example
        gt_relations = example.get('ground_truth', [])
        gt_relations_full = example.get('ground_truth_full', [])
        
        # Store result for examples
        result = {
            'question': question,
            'topic_entities': topic_entities,
            'predicted_relations': predicted[:5],  # Top 5 relations
            'ground_truth_relations': gt_relations[:5] if gt_relations else [],  # GT relation paths
            'ground_truth_full_paths': gt_relations_full[:3] if gt_relations_full else [],  # Full paths with entities
            'retrieved_entities': list(retrieved)[:10],  # Top 10 retrieved
            'ground_truth_entities': list(gt_entity_set),
            'match': bool(retrieved & gt_entity_set) if gt_entity_set else None
        }
        all_results.append(result)
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(examples)} examples...")
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("COMPUTING METRICS")
    print("=" * 70)
    
    # Entity retrieval metrics
    entity_hits_1 = calculate_entity_hits_at_k(all_retrieved, all_ground_truth, k=1)
    entity_hits_5 = calculate_entity_hits_at_k(all_retrieved, all_ground_truth, k=5)
    entity_hits_10 = calculate_entity_hits_at_k(all_retrieved, all_ground_truth, k=10)
    entity_recall = calculate_entity_recall(all_retrieved, all_ground_truth)
    
    # Original relation metrics (from eval_data)
    original_metrics = eval_data.get('metrics', {})
    
    metrics = {
        # Entity retrieval metrics
        'entity_hits_at_1': entity_hits_1,
        'entity_hits_at_5': entity_hits_5,
        'entity_hits_at_10': entity_hits_10,
        'entity_recall': entity_recall['entity_recall'],
        'entity_exact_match': entity_recall['exact_match_rate'],
        'entity_partial_match': entity_recall['partial_match_rate'],
        
        # Original relation metrics
        'relation_hits_at_1': original_metrics.get('hits_at_1', 0),
        'relation_hits_at_5': original_metrics.get('hits_at_5', 0),
        
        # Timing
        'avg_entity_id_time_ms': (entity_id_time / len(examples)) * 1000,
        'avg_retrieval_time_ms': (retrieval_time / len(examples)) * 1000,
        'total_samples': len(examples)
    }
    
    # Print results
    print(f"\n{'ENTITY RETRIEVAL METRICS':=^70}")
    print(f"  Entity Hits@1:  {entity_hits_1:.4f}")
    print(f"  Entity Hits@5:  {entity_hits_5:.4f}")
    print(f"  Entity Hits@10: {entity_hits_10:.4f}")
    print(f"  Entity Recall:  {entity_recall['entity_recall']:.4f}")
    print(f"  Exact Match:    {entity_recall['exact_match_rate']:.4f}")
    print(f"  Partial Match:  {entity_recall['partial_match_rate']:.4f}")
    
    print(f"\n{'RELATION PREDICTION METRICS':=^70}")
    print(f"  Relation Hits@1: {original_metrics.get('hits_at_1', 0):.4f}")
    print(f"  Relation Hits@5: {original_metrics.get('hits_at_5', 0):.4f}")
    
    print(f"\n{'PERFORMANCE':=^70}")
    print(f"  Avg Entity ID Time:   {metrics['avg_entity_id_time_ms']:.3f} ms")
    print(f"  Avg Retrieval Time:   {metrics['avg_retrieval_time_ms']:.3f} ms")
    print(f"  Total Samples:        {len(examples)}")
    
    # Show examples
    print(f"\n{'EXAMPLE OUTPUTS':=^70}")
    for i, result in enumerate(all_results[:10]):
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {result['question']}")
        print(f"Topic Entities: {result['topic_entities']}")
        print(f"Predicted Relations: {result['predicted_relations'][:3]}")
        print(f"Retrieved Entities: {result['retrieved_entities'][:5]}")
        print(f"Ground Truth: {result['ground_truth_entities']}")
        print(f"Match: {'✓' if result['match'] else '✗'}" if result['match'] is not None else "Match: N/A")
    
    return {
        'metrics': metrics,
        'examples': all_results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate with Entity Retrieval')
    parser.add_argument('--eval_results', type=str, 
                       default='webqsp_final_evaluation.json',
                       help='Path to evaluation results JSON')
    parser.add_argument('--test_data', type=str,
                       default='../Data/webqsp_final/test.parquet',
                       help='Path to test data parquet')
    parser.add_argument('--vocab', type=str,
                       default='outputs_multipath_relation_only_8_freeze_question_encoder/vocab.json',
                       help='Path to vocabulary JSON')
    parser.add_argument('--output', type=str,
                       default='entity_retrieval_evaluation.json',
                       help='Output path for results')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Maximum examples to evaluate')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_with_entity_retrieval(
        eval_results_path=args.eval_results,
        test_data_path=args.test_data,
        vocab_path=args.vocab,
        max_examples=args.max_examples
    )
    
    # Save results
    print(f"\n{'SAVING RESULTS':=^70}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
