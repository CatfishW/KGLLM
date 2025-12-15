"""
KG-RAG Demo: Interactive demonstration of Knowledge Graph path retrieval.

This script provides two modes:
1. Build mode: Create the FAISS index from parquet data
2. Demo mode: Interactively query the system

Usage:
    # Build index first
    python -m EXP.demo --build
    
    # Run interactive demo
    python -m EXP.demo
    
    # Run with specific question
    python -m EXP.demo --question "who is obama's wife"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def build_index(data_dir: str, output_dir: str):
    """Build the FAISS index from parquet data."""
    from EXP.path_indexer import build_index as _build_index
    from EXP.config import RAGConfig
    
    print("\n" + "="*60)
    print("  Building KG-RAG Index")
    print("="*60 + "\n")
    
    config = RAGConfig(
        data_dir=data_dir,
        index_dir=output_dir
    )
    
    _build_index(data_dir, output_dir, config)
    
    print("\n" + "="*60)
    print("  Index built successfully!")
    print(f"  Location: {output_dir}")
    print("="*60 + "\n")


def run_demo(index_path: str, question: str = None):
    """Run the interactive demo or answer a single question."""
    from EXP.pipeline import KGRAGPipeline
    
    pipeline = KGRAGPipeline(index_path)
    
    if question:
        # Answer single question
        print(f"\nQuestion: {question}\n")
        response = pipeline.ask(question, top_k=5)
        print(response.format_readable())
    else:
        # Interactive mode
        pipeline.interactive_demo()


def run_evaluation(index_path: str, data_dir: str, split: str = "test", num_samples: int = 100):
    """
    Evaluate retrieval quality on a dataset split.
    
    Checks if ground-truth paths appear in top-k retrieved paths.
    """
    import json
    import pandas as pd
    from tqdm import tqdm
    from EXP.pipeline import KGRAGPipeline
    
    print("\n" + "="*60)
    print(f"  Evaluating on {split} set ({num_samples} samples)")
    print("="*60 + "\n")
    
    pipeline = KGRAGPipeline(index_path)
    
    # Load data
    parquet_path = Path(data_dir) / f"{split}.parquet"
    df = pd.read_parquet(parquet_path)
    
    if num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)
    
    # Evaluate
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = row['question']
        paths_str = row['paths']
        
        # Parse ground truth paths
        if isinstance(paths_str, str):
            try:
                gt_paths = json.loads(paths_str)
            except:
                continue
        else:
            gt_paths = paths_str if paths_str else []
        
        if not gt_paths:
            continue
        
        # Get ground truth relation chains
        gt_chains = set()
        for path in gt_paths:
            relations = path.get('relations', [])
            if relations:
                chain = " -> ".join(relations)
                gt_chains.add(chain)
        
        if not gt_chains:
            continue
        
        # Retrieve
        response = pipeline.ask(question, top_k=10)
        retrieved_chains = [p.relation_chain for p in response.paths]
        
        # Check hits
        if len(retrieved_chains) >= 1 and retrieved_chains[0] in gt_chains:
            hits_at_1 += 1
        
        if any(c in gt_chains for c in retrieved_chains[:5]):
            hits_at_5 += 1
        
        if any(c in gt_chains for c in retrieved_chains[:10]):
            hits_at_10 += 1
        
        total += 1
    
    # Print results
    print("\n" + "-"*40)
    print("Evaluation Results:")
    print("-"*40)
    print(f"  Total samples: {total}")
    print(f"  Hits@1:  {hits_at_1}/{total} = {hits_at_1/total*100:.1f}%")
    print(f"  Hits@5:  {hits_at_5}/{total} = {hits_at_5/total*100:.1f}%")
    print(f"  Hits@10: {hits_at_10}/{total} = {hits_at_10/total*100:.1f}%")
    print("-"*40 + "\n")


def show_examples(data_dir: str, num: int = 5):
    """Show example questions from the dataset."""
    import json
    import pandas as pd
    
    print("\n" + "="*60)
    print("  Example Questions from WebQSP")
    print("="*60 + "\n")
    
    parquet_path = Path(data_dir) / "train.parquet"
    df = pd.read_parquet(parquet_path)
    
    samples = df.sample(n=num, random_state=42)
    
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        question = row['question']
        answer = row['answer']
        paths_str = row['paths']
        
        # Parse answer
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except:
                pass
        
        # Parse paths
        if isinstance(paths_str, str):
            try:
                paths = json.loads(paths_str)
            except:
                paths = []
        else:
            paths = paths_str if paths_str else []
        
        print(f"[{idx}] Question: {question}")
        print(f"    Answer: {answer[:3] if len(answer) > 3 else answer}")
        
        if paths:
            first_path = paths[0]
            relations = first_path.get('relations', [])
            if relations:
                print(f"    Path: {' -> '.join(relations)}")
        print()


def run_enhanced_demo(
    index_path: str, 
    question: str = None,
    use_classifier: bool = False,
    use_diffusion: bool = False,
    classifier_mode: str = "rule",
    classifier_model_path: str = "EXP/models/classifier",
    diffusion_mode: str = "embedding",
    diffusion_model_path: str = "EXP/models/diffusion/model.ckpt"
):
    """Run demo with enhanced modules."""
    from EXP.pipeline import EnhancedKGRAGPipeline
    from EXP.config import RAGConfig
    
    config = RAGConfig(
        use_question_classifier=use_classifier,
        classifier_mode=classifier_mode,
        classifier_model_path=classifier_model_path,
        use_diffusion_ranker=use_diffusion,
        diffusion_mode=diffusion_mode,
        diffusion_model_path=diffusion_model_path
    )
    
    pipeline = EnhancedKGRAGPipeline(index_path, config)
    
    if question:
        print(f"\nQuestion: {question}\n")
        response = pipeline.ask(question, top_k=5)
        print(response.format_readable())
    else:
        pipeline.interactive_demo()


def run_enhanced_evaluation(
    index_path: str, 
    data_dir: str, 
    split: str = "test", 
    num_samples: int = 100,
    use_classifier: bool = False,
    use_diffusion: bool = False,
    classifier_mode: str = "rule",
    classifier_model_path: str = "EXP/models/classifier",
    diffusion_mode: str = "embedding",
    diffusion_model_path: str = "EXP/models/diffusion/model.ckpt"
):
    """
    Evaluate with enhanced modules.
    """
    import json
    import pandas as pd
    from tqdm import tqdm
    from EXP.pipeline import EnhancedKGRAGPipeline
    from EXP.config import RAGConfig
    
    # Build config string
    config_name = "Baseline"
    if use_classifier and use_diffusion:
        config_name = f"Classifier({classifier_mode}) + Diffusion({diffusion_mode})"
    elif use_classifier:
        config_name = f"Classifier({classifier_mode})"
    elif use_diffusion:
        config_name = f"Diffusion({diffusion_mode})"
    
    print("\n" + "="*60)
    print(f"  Evaluating: {config_name}")
    print(f"  Split: {split}, Samples: {num_samples}")
    print("="*60 + "\n")
    
    config = RAGConfig(
        use_question_classifier=use_classifier,
        classifier_mode=classifier_mode,
        classifier_model_path=classifier_model_path,
        use_diffusion_ranker=use_diffusion,
        diffusion_mode=diffusion_mode,
        diffusion_model_path=diffusion_model_path
    )
    
    pipeline = EnhancedKGRAGPipeline(index_path, config)
    
    # Load data
    parquet_path = Path(data_dir) / f"{split}.parquet"
    df = pd.read_parquet(parquet_path)
    
    if num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)
    
    # Evaluate
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total = 0
    
    # Track question type distribution
    type_counts = {}
    type_hits = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = row['question']
        paths_str = row['paths']
        
        # Parse ground truth paths
        if isinstance(paths_str, str):
            try:
                gt_paths = json.loads(paths_str)
            except:
                continue
        else:
            gt_paths = paths_str if paths_str else []
        
        if not gt_paths:
            continue
        
        # Get ground truth relation chains
        gt_chains = set()
        for path in gt_paths:
            relations = path.get('relations', [])
            if relations:
                chain = " -> ".join(relations)
                gt_chains.add(chain)
        
        if not gt_chains:
            continue
        
        # Retrieve
        response = pipeline.ask(question, top_k=10)
        retrieved_chains = [p.relation_chain for p in response.paths]
        
        # Track question type
        if response.question_type:
            qtype = response.question_type
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        # Check hits
        hit_at_5 = any(c in gt_chains for c in retrieved_chains[:5])
        
        if len(retrieved_chains) >= 1 and retrieved_chains[0] in gt_chains:
            hits_at_1 += 1
        
        if hit_at_5:
            hits_at_5 += 1
            if response.question_type:
                type_hits[response.question_type] = type_hits.get(response.question_type, 0) + 1
        
        if any(c in gt_chains for c in retrieved_chains[:10]):
            hits_at_10 += 1
        
        total += 1
    
    # Print results
    print("\n" + "-"*40)
    print(f"Evaluation Results ({config_name}):")
    print("-"*40)
    print(f"  Total samples: {total}")
    print(f"  Hits@1:  {hits_at_1}/{total} = {hits_at_1/total*100:.1f}%")
    print(f"  Hits@5:  {hits_at_5}/{total} = {hits_at_5/total*100:.1f}%")
    print(f"  Hits@10: {hits_at_10}/{total} = {hits_at_10/total*100:.1f}%")
    
    # Print by question type
    if type_counts:
        print("\nBy Question Type:")
        for qtype, count in sorted(type_counts.items()):
            hits = type_hits.get(qtype, 0)
            pct = hits / count * 100 if count > 0 else 0
            print(f"  {qtype}: {hits}/{count} = {pct:.1f}%")
    
    print("-"*40 + "\n")
    
    return {
        'config': config_name,
        'total': total,
        'hits_at_1': hits_at_1,
        'hits_at_5': hits_at_5,
        'hits_at_10': hits_at_10
    }


def run_comparison(index_path: str, data_dir: str, num_samples: int = 100):
    """Run comparison between all configurations."""
    print("\n" + "="*60)
    print("  Running Configuration Comparison")
    print("="*60 + "\n")
    
    results = []
    
    # Baseline
    results.append(run_enhanced_evaluation(
        index_path, data_dir, num_samples=num_samples,
        use_classifier=False, use_diffusion=False
    ))
    
    # Neural Classifier (if model exists)
    classifier_model = "EXP/models/classifier"
    if Path(classifier_model).exists():
        results.append(run_enhanced_evaluation(
            index_path, data_dir, num_samples=num_samples,
            use_classifier=True, use_diffusion=False,
            classifier_mode="neural", classifier_model_path=classifier_model
        ))
    
    # Diffusion (Embedding)
    results.append(run_enhanced_evaluation(
        index_path, data_dir, num_samples=num_samples,
        use_classifier=False, use_diffusion=True,
        diffusion_mode="embedding"
    ))
    
    # Diffusion (Likelihood) - if model exists
    diffusion_model = "EXP/models/diffusion/model.ckpt"
    if Path(diffusion_model).exists():
        results.append(run_enhanced_evaluation(
            index_path, data_dir, num_samples=num_samples,
            use_classifier=False, use_diffusion=True,
            diffusion_mode="likelihood", diffusion_model_path=diffusion_model
        ))
    
    # Full Enhanced (Neural Classifier + Likelihood Diffusion)
    if Path(classifier_model).exists() and Path(diffusion_model).exists():
        results.append(run_enhanced_evaluation(
            index_path, data_dir, num_samples=num_samples,
            use_classifier=True, use_diffusion=True,
            classifier_mode="neural", classifier_model_path=classifier_model,
            diffusion_mode="likelihood", diffusion_model_path=diffusion_model
        ))
    
    # Print comparison table
    print("\n" + "="*60)
    print("  Comparison Summary")
    print("="*60)
    print(f"\n{'Configuration':<40} {'Hits@1':>10} {'Hits@5':>10} {'Hits@10':>10}")
    print("-"*80)
    
    for r in results:
        h1 = f"{r['hits_at_1']/r['total']*100:.1f}%"
        h5 = f"{r['hits_at_5']/r['total']*100:.1f}%"
        h10 = f"{r['hits_at_10']/r['total']*100:.1f}%"
        print(f"{r['config']:<40} {h1:>10} {h5:>10} {h10:>10}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="KG-RAG Demo: Knowledge Graph Path Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--build", action="store_true", help="Build the FAISS index")
    parser.add_argument("--question", "-q", type=str, help="Ask a specific question")
    parser.add_argument("--examples", action="store_true", help="Show examples")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate")
    parser.add_argument("--compare", action="store_true", help="Compare configurations")
    parser.add_argument("--data_dir", type=str, default="Data/webqsp_final")
    parser.add_argument("--index_dir", type=str, default="EXP/index")
    parser.add_argument("--num_samples", type=int, default=100)
    
    # Enhanced module flags
    parser.add_argument("--use_classifier", action="store_true", help="Enable classifier")
    parser.add_argument("--use_diffusion", action="store_true", help="Enable diffusion ranker")
    
    parser.add_argument("--classifier_mode", type=str, default="rule", 
                       choices=["rule", "neural", "hybrid"], help="Classifier mode")
    parser.add_argument("--classifier_model_path", type=str, default="EXP/models/classifier",
                       help="Path to neural classifier model")
    
    parser.add_argument("--diffusion_mode", type=str, default="embedding",
                       choices=["embedding", "likelihood"], help="Diffusion mode")
    parser.add_argument("--diffusion_model_path", type=str, default="EXP/models/diffusion/model.ckpt",
                       help="Path to trained diffusion model")
    
    args = parser.parse_args()
    
    if args.build:
        build_index(args.data_dir, args.index_dir)
    elif args.examples:
        show_examples(args.data_dir)
    elif args.compare:
        run_comparison(args.index_dir, args.data_dir, args.num_samples)
    elif args.evaluate:
        if args.use_classifier or args.use_diffusion:
            run_enhanced_evaluation(
                args.index_dir, args.data_dir, 
                num_samples=args.num_samples,
                use_classifier=args.use_classifier,
                use_diffusion=args.use_diffusion,
                classifier_mode=args.classifier_mode,
                classifier_model_path=args.classifier_model_path,
                diffusion_mode=args.diffusion_mode,
                diffusion_model_path=args.diffusion_model_path
            )
        else:
            run_evaluation(args.index_dir, args.data_dir, num_samples=args.num_samples)
    else:
        index_path = Path(args.index_dir)
        if not (index_path / "faiss.index").exists():
            print("Index not found! Please build it first.")
            return
        
        if args.use_classifier or args.use_diffusion:
            run_enhanced_demo(
                args.index_dir, args.question,
                use_classifier=args.use_classifier,
                use_diffusion=args.use_diffusion,
                classifier_mode=args.classifier_mode,
                classifier_model_path=args.classifier_model_path,
                diffusion_mode=args.diffusion_mode,
                diffusion_model_path=args.diffusion_model_path
            )
        else:
            run_demo(args.index_dir, args.question)


if __name__ == "__main__":
    main()

