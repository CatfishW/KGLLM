"""
Test script for Entity Identifier and Entity Retriever.

Tests the components using the evaluation data format from webqsp_final_evaluation.json
and the actual training data with graph triples.
"""

import json
import pandas as pd
import time
from typing import List, Dict, Tuple
from collections import defaultdict

from entity_identifier import EntityIdentifier, FastEntityIdentifier
from entity_retriever import EntityRetriever, MultiHopEntityRetriever


def load_evaluation_data(path: str, max_samples: int = None) -> List[Dict]:
    """Load evaluation data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = data.get("examples", [])
    if max_samples:
        examples = examples[:max_samples]
    
    return examples


def load_training_data(path: str, max_samples: int = None) -> pd.DataFrame:
    """Load training data with graph triples from parquet file."""
    df = pd.read_parquet(path)
    if max_samples:
        df = df.head(max_samples)
    return df


def test_entity_identifier(examples: List[Dict]) -> Dict:
    """
    Test entity identifier on evaluation examples.
    
    Returns metrics comparing identified entities against question entities
    (if available) or ground truth answers.
    """
    print("\n" + "="*70)
    print("TESTING ENTITY IDENTIFIER")
    print("="*70)
    
    # Initialize identifiers
    print("\nInitializing EntityIdentifier (with SpaCy)...")
    identifier = EntityIdentifier(use_spacy=True)
    
    print("Initializing FastEntityIdentifier (regex-only)...")
    fast_identifier = FastEntityIdentifier()
    
    results = {
        "spacy": {"total": 0, "found_any": 0, "examples": []},
        "fast": {"total": 0, "found_any": 0, "examples": []}
    }
    
    print(f"\nProcessing {len(examples)} examples...\n")
    
    # Test each example
    start_time = time.time()
    for i, example in enumerate(examples[:20]):  # Test first 20 for demo
        question = example.get("question", "")
        answer = example.get("answer", [])
        
        # SpaCy identifier
        spacy_entities = identifier.identify(question)
        results["spacy"]["total"] += 1
        if spacy_entities:
            results["spacy"]["found_any"] += 1
        
        # Fast identifier
        fast_entities = fast_identifier.identify(question)
        results["fast"]["total"] += 1
        if fast_entities:
            results["fast"]["found_any"] += 1
        
        # Store example results
        results["spacy"]["examples"].append({
            "question": question,
            "identified": spacy_entities,
            "answer": answer[:3]  # First 3 answers for brevity
        })
        results["fast"]["examples"].append({
            "question": question,
            "identified": fast_entities,
            "answer": answer[:3]
        })
    
    elapsed = time.time() - start_time
    
    # Print results
    print("-" * 70)
    print("ENTITY IDENTIFIER RESULTS")
    print("-" * 70)
    
    print(f"\nSpaCy-based Identifier:")
    print(f"  Total questions: {results['spacy']['total']}")
    print(f"  Found entities:  {results['spacy']['found_any']} ({100*results['spacy']['found_any']/results['spacy']['total']:.1f}%)")
    
    print(f"\nFast (regex) Identifier:")
    print(f"  Total questions: {results['fast']['total']}")
    print(f"  Found entities:  {results['fast']['found_any']} ({100*results['fast']['found_any']/results['fast']['total']:.1f}%)")
    
    print(f"\nProcessing time: {elapsed:.3f}s ({elapsed/len(examples)*1000:.2f}ms per question)")
    
    # Show sample outputs
    print("\n" + "-" * 70)
    print("SAMPLE OUTPUTS (SpaCy)")
    print("-" * 70)
    
    for ex in results["spacy"]["examples"][:10]:
        print(f"\nQ: {ex['question']}")
        print(f"   Identified: {ex['identified']}")
        print(f"   Answers:    {ex['answer']}")
    
    return results


def test_entity_retriever(df: pd.DataFrame) -> Dict:
    """
    Test entity retriever on training data with graph triples.
    
    For each sample:
    1. Build graph index from triples
    2. Get question entities
    3. Traverse predicted paths to retrieve answer candidates
    4. Compare with actual answers
    """
    print("\n" + "="*70)
    print("TESTING ENTITY RETRIEVER")
    print("="*70)
    
    results = {
        "total": 0,
        "has_graph": 0,
        "has_paths": 0,
        "found_answers": 0,
        "examples": []
    }
    
    print(f"\nProcessing {len(df)} samples with graph triples...\n")
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        results["total"] += 1
        
        # Get graph triples
        graph = row.get("graph", [])
        if graph is None or len(graph) == 0:
            continue
        
        results["has_graph"] += 1
        
        # Get paths
        paths = row.get("paths", [])
        if isinstance(paths, str):
            paths = json.loads(paths)
        
        if not paths:
            continue
        
        results["has_paths"] += 1
        
        # Build retriever for this sample
        retriever = EntityRetriever()
        try:
            if isinstance(graph, str):
                graph = json.loads(graph)
            retriever.build_index(graph)
        except Exception as e:
            continue
        
        # Get question entities
        q_entities = row.get("q_entity", [])
        if isinstance(q_entities, str):
            q_entities = json.loads(q_entities)
        
        if not q_entities:
            continue
        
        # Get actual answers
        a_entities = row.get("a_entity", [])
        if isinstance(a_entities, str):
            a_entities = json.loads(a_entities)
        
        answer_set = set(str(a).lower() for a in a_entities)
        
        # Try to retrieve answers using paths
        all_retrieved = set()
        
        for path_info in paths[:5]:  # Try first 5 paths
            if isinstance(path_info, dict):
                relations = path_info.get("relations", [])
            else:
                continue
            
            for q_entity in q_entities:
                retrieved = retriever.retrieve(str(q_entity), relations)
                all_retrieved.update(retrieved)
        
        # Check if we found any answers
        retrieved_lower = set(r.lower() for r in all_retrieved)
        found = len(retrieved_lower & answer_set) > 0
        
        if found:
            results["found_answers"] += 1
        
        # Store example
        if results["total"] <= 10:
            results["examples"].append({
                "question": row.get("question", ""),
                "q_entities": q_entities[:3],
                "retrieved": list(all_retrieved)[:5],
                "actual_answers": a_entities[:3],
                "found": found
            })
        
        # Progress update
        if results["total"] % 100 == 0:
            print(f"  Processed {results['total']} samples...")
    
    elapsed = time.time() - start_time
    
    # Print results
    print("-" * 70)
    print("ENTITY RETRIEVER RESULTS")
    print("-" * 70)
    
    print(f"\nTotal samples:    {results['total']}")
    print(f"With graph:       {results['has_graph']} ({100*results['has_graph']/results['total']:.1f}%)")
    print(f"With paths:       {results['has_paths']} ({100*results['has_paths']/max(1,results['has_graph']):.1f}%)")
    print(f"Found answers:    {results['found_answers']} ({100*results['found_answers']/max(1,results['has_paths']):.1f}%)")
    
    print(f"\nProcessing time: {elapsed:.3f}s ({elapsed/max(1,results['total'])*1000:.2f}ms per sample)")
    
    # Show sample outputs
    print("\n" + "-" * 70)
    print("SAMPLE OUTPUTS")
    print("-" * 70)
    
    for ex in results["examples"]:
        print(f"\nQ: {ex['question']}")
        print(f"   Q Entities:  {ex['q_entities']}")
        print(f"   Retrieved:   {ex['retrieved']}")
        print(f"   Actual:      {ex['actual_answers']}")
        print(f"   Found: {'✓' if ex['found'] else '✗'}")
    
    return results


def test_integrated_pipeline(examples: List[Dict], df: pd.DataFrame = None) -> Dict:
    """
    Test the integrated pipeline: Question -> Entity Identification -> Path Retrieval -> Answer.
    
    This simulates the full KGQA pipeline.
    """
    print("\n" + "="*70)
    print("TESTING INTEGRATED PIPELINE")
    print("="*70)
    
    # Initialize components
    identifier = FastEntityIdentifier()
    
    results = {
        "total": 0,
        "identified_entity": 0,
        "predicted_matched_gt": 0,
        "examples": []
    }
    
    print(f"\nProcessing {len(examples)} evaluation examples...\n")
    
    for i, example in enumerate(examples[:30]):
        results["total"] += 1
        
        question = example.get("question", "")
        predicted_paths = example.get("predicted", [])
        ground_truth = example.get("ground_truth", [])
        answer = example.get("answer", [])
        
        # Step 1: Identify entities in question
        identified = identifier.identify(question)
        
        if identified:
            results["identified_entity"] += 1
        
        # Step 2: Check if any predicted path overlaps with ground truth
        pred_relations = set()
        for path in predicted_paths[:5]:  # Check top 5 predictions
            parts = [p.strip() for p in path.split('->')]
            pred_relations.update(parts)
        
        gt_relations = set()
        for path in ground_truth:
            parts = [p.strip() for p in path.split('->')]
            gt_relations.update(parts)
        
        has_overlap = len(pred_relations & gt_relations) > 0
        if has_overlap:
            results["predicted_matched_gt"] += 1
        
        # Store example
        if i < 10:
            results["examples"].append({
                "question": question,
                "identified_entities": identified,
                "predicted_paths": predicted_paths[:2],
                "ground_truth_paths": ground_truth[:2],
                "answer": answer[:2],
                "path_overlap": has_overlap
            })
    
    # Print results
    print("-" * 70)
    print("INTEGRATED PIPELINE RESULTS")
    print("-" * 70)
    
    print(f"\nTotal examples:     {results['total']}")
    print(f"Entity identified:  {results['identified_entity']} ({100*results['identified_entity']/results['total']:.1f}%)")
    print(f"Path overlap w/GT:  {results['predicted_matched_gt']} ({100*results['predicted_matched_gt']/results['total']:.1f}%)")
    
    # Show sample outputs
    print("\n" + "-" * 70)
    print("SAMPLE PIPELINE OUTPUTS")
    print("-" * 70)
    
    for ex in results["examples"]:
        print(f"\nQ: {ex['question']}")
        print(f"   Entities:    {ex['identified_entities']}")
        print(f"   Predicted:   {ex['predicted_paths']}")
        print(f"   GT Paths:    {ex['ground_truth_paths']}")
        print(f"   Answer:      {ex['answer']}")
        print(f"   Overlap: {'✓' if ex['path_overlap'] else '✗'}")
    
    return results


def main():
    """Main test function."""
    import os
    
    print("="*70)
    print("ENTITY IDENTIFIER & RETRIEVER TEST SUITE")
    print("="*70)
    
    # Find data files
    eval_path = "webqsp_final_evaluation.json"
    train_path = "../Data/webqsp_final/train.parquet"
    
    # Check if files exist
    if not os.path.exists(eval_path):
        print(f"\nWarning: {eval_path} not found. Skipping evaluation tests.")
        eval_data = []
    else:
        print(f"\nLoading evaluation data from {eval_path}...")
        eval_data = load_evaluation_data(eval_path, max_samples=100)
        print(f"Loaded {len(eval_data)} examples")
    
    if not os.path.exists(train_path):
        print(f"\nWarning: {train_path} not found. Skipping retriever tests with graph data.")
        train_df = None
    else:
        print(f"\nLoading training data from {train_path}...")
        train_df = load_training_data(train_path, max_samples=100)
        print(f"Loaded {len(train_df)} training samples")
    
    # Run tests
    if eval_data:
        test_entity_identifier(eval_data)
    
    if train_df is not None:
        test_entity_retriever(train_df)
    
    if eval_data:
        test_integrated_pipeline(eval_data, train_df)
    
    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
