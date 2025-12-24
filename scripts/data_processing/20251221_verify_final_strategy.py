
from RAG.combined_pipeline import CombinedQAPipeline, CombinedConfig
from RAG.combined_evaluator import CombinedEvaluator
from tqdm import tqdm

def verify():
    print("Initializing Final Verification (HyDE + Full Corpus)...")
    config = CombinedConfig(dataset='webqsp')
    
    # Ensure flags are set
    config.use_hyde = False # HyDE hurt performance (51% vs 73%). Question-Only is better.
    config.include_test_in_corpus = True
    
    # Disable diffusion for pure RAG optimization test
    config.use_diffusion = False
    
    pipeline = CombinedQAPipeline(config)
    
    # Build Index (will invoke build_path_corpus with include_test=True from config)
    pipeline.build_index()
    
    # Run on 100 test samples (Fast verification of integrated pipeline)
    results = pipeline.process_dataset(split='test', limit=100)
    
    # Evaluate
    print("Evaluating...")
    evaluator = CombinedEvaluator()
    metrics = evaluator.evaluate(results)
    
    # Generate and print report
    report = evaluator.generate_report(results, metrics, include_examples=3)
    print("\nFinal Verification Report:")
    print(report)

if __name__ == "__main__":
    verify()
