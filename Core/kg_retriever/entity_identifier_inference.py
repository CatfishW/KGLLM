"""
Inference script for Entity Identifier.

Provides:
- Batch entity identification
- Entity index building
- Evaluation on test sets
- Demo with sample questions

Usage:
    # Demo
    python entity_identifier_inference.py --demo
    
    # Evaluate on test set
    python entity_identifier_inference.py \
        --checkpoint outputs_entity_identifier/best.ckpt \
        --data /data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/test.parquet \
        --eval
    
    # Build entity index from graph data
    python entity_identifier_inference.py \
        --checkpoint outputs_entity_identifier/best.ckpt \
        --build-index \
        --entity-data /path/to/entities.json \
        --output-index entity_embeddings/
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import torch
from tqdm import tqdm

from models.entity_identifier import EntityIdentifierModel, LinkedEntity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Optional[str] = None, device: str = "cuda") -> EntityIdentifierModel:
    """Load model from checkpoint or create fresh model."""
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading model from {checkpoint_path}")
        model = EntityIdentifierModel.load_from_checkpoint(checkpoint_path)
    else:
        logger.info("Creating fresh model (no checkpoint)")
        model = EntityIdentifierModel(
            encoder_name="BAAI/bge-small-en-v1.5",
            use_gliner=False,  # Disable for speed in demo
        )
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_on_dataset(
    model: EntityIdentifierModel,
    data_path: str,
    output_path: Optional[str] = None,
    max_samples: int = 0,
) -> Dict:
    """
    Evaluate entity identification on a test dataset.
    
    Metrics:
    - Recall@K: Fraction of ground truth entities found in top-K predictions
    - Hits@1: Fraction where top prediction is correct
    - MRR: Mean Reciprocal Rank
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    if max_samples > 0:
        df = df.head(max_samples)
    
    results = []
    hits_1 = 0
    hits_5 = 0
    total = 0
    mrr_sum = 0.0
    
    logger.info(f"Evaluating on {len(df)} samples...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row.get("question", "")
        
        # Get ground truth topic entities
        gt_entities = []
        if "topic_mid" in row:
            topic_mid = row["topic_mid"]
            if isinstance(topic_mid, str):
                gt_entities = [topic_mid]
            elif isinstance(topic_mid, list):
                gt_entities = topic_mid
        
        if not gt_entities or not question:
            continue
        
        # Get graph for candidate filtering
        graph = row.get("graph", [])
        if isinstance(graph, str):
            try:
                graph = json.loads(graph)
            except:
                graph = []
        
        # Run identification
        with torch.no_grad():
            linked = model.identify(question, graph=graph, top_k=5)
        
        # Get predicted entity IDs and names
        pred_entities = [e.entity_name for e in linked]
        pred_ids = [e.entity_id for e in linked]
        
        # Check hits
        gt_names_lower = [str(e).lower() for e in gt_entities]
        
        found_rank = None
        for i, (name, eid) in enumerate(zip(pred_entities, pred_ids)):
            name_lower = name.lower()
            eid_lower = eid.lower() if eid else ""
            
            if name_lower in gt_names_lower or eid_lower in gt_names_lower:
                if found_rank is None:
                    found_rank = i + 1
                break
            
            # Also check if prediction contains ground truth
            for gt in gt_names_lower:
                if gt in name_lower or name_lower in gt:
                    if found_rank is None:
                        found_rank = i + 1
                    break
        
        if found_rank:
            mrr_sum += 1.0 / found_rank
            if found_rank == 1:
                hits_1 += 1
            if found_rank <= 5:
                hits_5 += 1
        
        total += 1
        
        results.append({
            "question": question,
            "ground_truth": gt_entities,
            "predictions": [
                {"name": e.entity_name, "id": e.entity_id, "score": e.score}
                for e in linked[:5]
            ],
            "found_rank": found_rank,
        })
    
    # Compute metrics
    metrics = {
        "total": total,
        "hits@1": hits_1 / total if total > 0 else 0,
        "hits@5": hits_5 / total if total > 0 else 0,
        "mrr": mrr_sum / total if total > 0 else 0,
    }
    
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {metrics['total']}")
    logger.info(f"Hits@1:        {metrics['hits@1']:.4f} ({hits_1}/{total})")
    logger.info(f"Hits@5:        {metrics['hits@5']:.4f} ({hits_5}/{total})")
    logger.info(f"MRR:           {metrics['mrr']:.4f}")
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({"metrics": metrics, "results": results[:100]}, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return metrics


def build_entity_index(
    model: EntityIdentifierModel,
    entity_data_path: str,
    output_index_path: str,
):
    """
    Build FAISS entity index from entity data.
    
    Entity data format: JSON list of {"id": "m.xxx", "name": "Entity Name", "description": "..."}
    """
    logger.info(f"Loading entity data from {entity_data_path}")
    
    with open(entity_data_path, 'r') as f:
        entities = json.load(f)
    
    logger.info(f"Building index for {len(entities)} entities...")
    
    model.build_entity_index(entities)
    model.save_entity_index(output_index_path)
    
    logger.info(f"Entity index saved to {output_index_path}")


def extract_entities_from_graph(data_path: str, output_path: str):
    """
    Extract unique entities from graph data for index building.
    """
    logger.info(f"Extracting entities from {data_path}")
    
    df = pd.read_parquet(data_path)
    
    entities = {}
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        graph = row.get("graph", [])
        if isinstance(graph, str):
            try:
                graph = json.loads(graph)
            except:
                continue
        
        for triple in graph:
            if len(triple) >= 3:
                head, rel, tail = triple[0], triple[1], triple[2]
                
                # Skip MIDs and relations
                if not head.startswith("m.") and not "." in head:
                    if head not in entities:
                        entities[head] = {"id": "", "name": head, "description": ""}
                
                if not tail.startswith("m.") and not "." in tail:
                    if tail not in entities:
                        entities[tail] = {"id": "", "name": tail, "description": ""}
    
    entity_list = list(entities.values())
    
    with open(output_path, 'w') as f:
        json.dump(entity_list, f, indent=2)
    
    logger.info(f"Extracted {len(entity_list)} unique entities to {output_path}")


def demo(model: EntityIdentifierModel):
    """Run demo with sample questions."""
    questions = [
        "What language do Jamaican people speak?",
        "Who is Barack Obama's wife?",
        "Where was Albert Einstein born?",
        "What movies did Natalie Portman star in?",
        "Who founded Microsoft?",
        "What is the capital of France?",
        "When did World War II end?",
        "Who wrote Harry Potter?",
    ]
    
    print("\n" + "="*70)
    print("ENTITY IDENTIFIER DEMO")
    print("="*70)
    
    total_time = 0
    
    for question in questions:
        start = time.time()
        
        with torch.no_grad():
            linked = model.identify(question)
        
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"\nQ: {question}")
        print(f"   Time: {elapsed*1000:.1f}ms")
        
        if linked:
            for e in linked[:3]:
                print(f"   → {e.entity_name} (id={e.entity_id}, score={e.score:.3f}, label={e.label})")
        else:
            print("   → No entities found")
    
    print(f"\n{'='*70}")
    print(f"Average time per question: {total_time/len(questions)*1000:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Entity Identifier Inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--data", type=str, default=None, help="Data path for evaluation")
    parser.add_argument("--output", type=str, default=None, help="Output path for results")
    parser.add_argument("--build-index", action="store_true", help="Build entity index")
    parser.add_argument("--entity-data", type=str, default=None, help="Entity data for index building")
    parser.add_argument("--extract-entities", action="store_true", help="Extract entities from graph data")
    parser.add_argument("--output-index", type=str, default="entity_embeddings", help="Output index path")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    if args.extract_entities:
        if not args.data:
            logger.error("--data required for entity extraction")
            return
        extract_entities_from_graph(args.data, args.entity_data or "entities.json")
        return
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    if args.demo:
        demo(model)
    
    if args.eval:
        if not args.data:
            logger.error("--data required for evaluation")
            return
        evaluate_on_dataset(
            model,
            args.data,
            args.output,
            args.max_samples,
        )
    
    if args.build_index:
        if not args.entity_data:
            logger.error("--entity-data required for index building")
            return
        build_entity_index(model, args.entity_data, args.output_index)
    
    # Default: run demo
    if not args.demo and not args.eval and not args.build_index:
        demo(model)


if __name__ == "__main__":
    main()
