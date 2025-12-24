#!/usr/bin/env python
"""
Main entry point for the KGQA RAG System.
Run evaluation on WebQSP or CWQ datasets.

Usage:
    python run_rag.py --dataset webqsp --split test
    python run_rag.py --dataset cwq --split test --limit 100
"""
import argparse
import sys
import os
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from RAG.config import RAGConfig
from RAG.pipeline import RAGPipeline
from RAG.evaluator import RAGEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="KGQA RAG System - Retrieval and Reranking for Knowledge Graph QA"
    )
    
    # Dataset options
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="webqsp",
        choices=["webqsp", "cwq"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for debugging)"
    )
    
    # Retrieval options
    parser.add_argument(
        "--retriever",
        type=str,
        default="hybrid",
        choices=["bm25", "dense", "hybrid"],
        help="Retriever type"
    )
    parser.add_argument(
        "--top-k-retrieve",
        type=int,
        default=100,
        help="Number of candidates to retrieve"
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Weight for dense retrieval (0=pure BM25, 1=pure dense)"
    )
    
    # Reranker options
    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=10,
        help="Number of final results after reranking"
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=32,
        help="Batch size for reranker"
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable Flash Attention 2"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking (retrieval only mode)"
    )
    
    # Device options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("KGQA RAG System - State-of-the-Art Retrieval Pipeline")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print(f"Retriever: {args.retriever}")
    print(f"Top-K Retrieve: {args.top_k_retrieve}")
    print(f"Top-K Rerank: {args.top_k_rerank}")
    print("=" * 60)
    
    # Build config
    config = RAGConfig(
        dataset=args.dataset,
        retriever_type=args.retriever,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        hybrid_alpha=args.hybrid_alpha,
        reranker_batch_size=args.reranker_batch_size,
        use_flash_attention=not args.no_flash_attention,
        reranker_dtype=args.dtype,
        device=args.device,
    )
    
    # Create pipeline
    pipeline = RAGPipeline(config)
    
    # Build index
    print("\n[1/3] Building retrieval index...")
    pipeline.build_index()
    
    # Run evaluation
    print(f"\n[2/3] Running evaluation on {args.split} set...")
    
    if args.no_rerank:
        results = pipeline.retrieval_only(
            split=args.split,
            limit=args.limit,
            top_k=args.top_k_retrieve
        )
    else:
        results = pipeline.process_dataset(
            split=args.split,
            limit=args.limit,
        )
    
    # Evaluate
    print("\n[3/3] Computing metrics...")
    evaluator = RAGEvaluator(eval_ks=config.eval_ks)
    metrics = evaluator.evaluate(results)
    
    # Print results
    print("\n" + str(metrics))
    
    # Generate report
    if args.output_dir is None:
        output_dir = Path(f"/data/Yanlai/KGLLM/RAG/results/{args.dataset}_{args.split}")
    else:
        output_dir = Path(args.output_dir)
    
    report = evaluator.generate_report(
        results=results,
        metrics=metrics,
        output_path=output_dir / "report.md",
        include_examples=10,
    )
    
    evaluator.save_results(results, metrics, output_dir)
    
    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")
    
    # Cleanup
    pipeline.reranker.unload()
    
    return metrics


if __name__ == "__main__":
    main()
