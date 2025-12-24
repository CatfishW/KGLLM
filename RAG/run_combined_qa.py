#!/usr/bin/env python
"""
Main entry point for the Combined KGQA System.

Combines diffusion path generation with reranker-based RAG for
state-of-the-art question answering on knowledge graphs.

Usage:
    python run_combined_qa.py --dataset webqsp --split test
    python run_combined_qa.py --dataset cwq --split test --limit 100
    python run_combined_qa.py --dataset webqsp --no-diffusion --no-llm  # RAG only
"""
import argparse
import sys
import os
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from RAG.combined_pipeline import CombinedQAPipeline, CombinedConfig
from RAG.combined_evaluator import CombinedEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined KGQA System - Diffusion + RAG + LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline on WebQSP test set
    python run_combined_qa.py --dataset webqsp --split test

    # Quick test with 50 samples
    python run_combined_qa.py --dataset webqsp --split test --limit 50

    # CWQ dataset
    python run_combined_qa.py --dataset cwq --split test --limit 100

    # RAG only (no diffusion, no LLM answer)
    python run_combined_qa.py --dataset webqsp --no-diffusion --no-llm

    # Diffusion only (no RAG retrieval)
    python run_combined_qa.py --dataset webqsp --no-rag
        """
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
    
    # Pipeline components
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG retrieval (use diffusion only)"
    )
    parser.add_argument(
        "--no-diffusion",
        action="store_true",
        help="Disable diffusion generation (RAG only)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM answer generation"
    )
    
    # Retrieval options
    parser.add_argument(
        "--top-k-retrieve",
        type=int,
        default=100,
        help="Number of candidates to retrieve"
    )
    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=10,
        help="Number of final results after reranking"
    )
    
    # Diffusion options
    parser.add_argument(
        "--num-diffusion-paths",
        type=int,
        default=5,
        help="Number of paths to generate per question"
    )
    parser.add_argument(
        "--diffusion-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for diffusion"
    )
    parser.add_argument(
        "--diffusion-checkpoint",
        type=str,
        default="/data/Yanlai/KGLLM/Core/diffusion_100M/checkpoints/last.ckpt",
        help="Path to diffusion model checkpoint"
    )
    parser.add_argument(
        "--diffusion-vocab",
        type=str,
        default="/data/Yanlai/KGLLM/Core/diffusion_100M/vocab.json",
        help="Path to diffusion vocabulary"
    )
    
    # LLM options
    parser.add_argument(
        "--llm-api",
        type=str,
        default="https://game.agaii.org/llm/v1",
        help="LLM API URL"
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.7,
        help="LLM sampling temperature"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results"
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
    
    print("=" * 70)
    print("Combined KGQA System - Diffusion + RAG + LLM")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print(f"RAG Retrieval: {'Enabled' if not args.no_rag else 'Disabled'}")
    print(f"Diffusion Generation: {'Enabled' if not args.no_diffusion else 'Disabled'}")
    print(f"LLM Answer: {'Enabled' if not args.no_llm else 'Disabled'}")
    print(f"Top-K Retrieve: {args.top_k_retrieve}")
    print(f"Top-K Rerank: {args.top_k_rerank}")
    if not args.no_diffusion:
        print(f"Diffusion Paths: {args.num_diffusion_paths}")
        print(f"Diffusion Checkpoint: {args.diffusion_checkpoint}")
    print("=" * 70)
    
    # Check files exist
    if not args.no_diffusion:
        if not os.path.exists(args.diffusion_checkpoint):
            print(f"Error: Diffusion checkpoint not found: {args.diffusion_checkpoint}")
            return 1
        if not os.path.exists(args.diffusion_vocab):
            print(f"Error: Diffusion vocab not found: {args.diffusion_vocab}")
            return 1
    
    # Build config
    config = CombinedConfig(
        dataset=args.dataset,
        device=args.device,
        
        # RAG settings
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        use_rag=not args.no_rag,
        
        # Diffusion settings
        use_diffusion=not args.no_diffusion,
        diffusion_checkpoint=args.diffusion_checkpoint,
        diffusion_vocab=args.diffusion_vocab,
        num_diffusion_paths=args.num_diffusion_paths,
        diffusion_temperature=args.diffusion_temperature,
        
        # LLM settings
        use_llm_answer=not args.no_llm,
        llm_api_url=args.llm_api,
        llm_temperature=args.llm_temperature,
    )
    
    # Create pipeline
    pipeline = CombinedQAPipeline(config)
    
    # Build index
    print("\n[1/4] Building retrieval index...")
    pipeline.build_index()
    
    # Run evaluation
    print(f"\n[2/4] Processing {args.split} set...")
    results = pipeline.process_dataset(
        split=args.split,
        limit=args.limit,
    )
    
    # Evaluate
    print("\n[3/4] Computing metrics...")
    evaluator = CombinedEvaluator(eval_ks=[1, 3, 5, 10, 20, 50, 100])
    metrics = evaluator.evaluate(results)
    
    # Print results
    print("\n" + str(metrics))
    
    # Save results
    print("\n[4/4] Saving results...")
    if args.output_dir is None:
        mode_suffix = ""
        if args.no_rag:
            mode_suffix += "_diffusion_only"
        if args.no_diffusion:
            mode_suffix += "_rag_only"
        if args.no_llm:
            mode_suffix += "_no_llm"
        output_dir = Path(f"/data/Yanlai/KGLLM/RAG/results/combined_{args.dataset}_{args.split}{mode_suffix}")
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
    pipeline.unload()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
