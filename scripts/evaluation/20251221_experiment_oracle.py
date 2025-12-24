
from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.combined_evaluator import CombinedEvaluator

def run_experiment():
    print("Initializing Oracle Experiment (Finding Ceiling)...")
    config = RAGConfig(dataset='webqsp')
    loader = KGDataLoader(config)
    llm = LLMClient()
    evaluator = CombinedEvaluator()

    # Full Corpus
    corpus, _ = loader.build_path_corpus(include_test=True)
    corpus_set = set(corpus) # For fast lookup
    
    test_data = loader.test_data[:100]
    
    f1_sum = 0
    oracle_found_count = 0
    count = 0
    
    print('Starting Oracle experiment...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # Find GT path in corpus
        gt_path = None
        for p in sample.get_path_strings():
            if p in corpus_set:
                gt_path = p
                break
        
        context_path = gt_path if gt_path else "NO_PATH_FOUND"
        if gt_path:
            oracle_found_count += 1
            
        # Answer Extraction
        ans_msg = [
            {'role': 'system', 'content': 'Extract the answer. Short.'},
            {'role': 'user', 'content': f'Q: {sample.question}\\nPath: {context_path}\\nAnswer:'}
        ]
        
        ans = llm.chat(ans_msg, max_tokens=32, temperature=0.1).strip().strip('"').strip("'")
        
        # Eval
        f1 = evaluator._compute_f1(ans, sample.answer_entities)
        f1_sum += f1
        count += 1
        
    print(f'Results (100 samples):')
    print(f'Oracle Path Found in Corpus: {oracle_found_count}/100')
    print(f'Oracle F1 (Ceiling): {100*f1_sum/count:.1f}%')

if __name__ == "__main__":
    run_experiment()
