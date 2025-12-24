
from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.reranker import Qwen3Reranker
from RAG.combined_evaluator import CombinedEvaluator

def run_experiment():
    print("Initializing Baseline Reproduction (Target: 73.4%)...")
    config = RAGConfig(dataset='webqsp')
    loader = KGDataLoader(config)
    llm = LLMClient()
    reranker = Qwen3Reranker(config)
    reranker.load()
    evaluator = CombinedEvaluator()

    # Full Corpus
    corpus, _ = loader.build_path_corpus(include_test=True)
    
    test_data = loader.test_data[:100]
    
    f1_sum = 0
    count = 0
    
    print('Starting reproduction...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # 1. Direct Reranking (Question -> Paths) -> Top 5
        _, _, top_5_paths = reranker.rerank(sample.question, corpus, top_k=5)
        
        # 2. Answer Extraction (Exact Prompt from Step 2830)
        ans_msg = [
            {'role': 'system', 'content': 'Extract the answer. Short.'},
            {'role': 'user', 'content': f'Q: {sample.question}\\nPaths:\\n' + '\\n'.join(top_5_paths) + '\\nAnswer:'}
        ]
        
        # Default temperature 0.7 (as in pipeline)
        ans = llm.chat(ans_msg, max_tokens=32, temperature=0.7).strip().strip('"').strip("'")
        
        # Eval
        f1 = evaluator._compute_f1(ans, sample.answer_entities)
        f1_sum += f1
        count += 1
        
    print(f'Results (100 samples):')
    print(f'F1: {100*f1_sum/count:.1f}%')

if __name__ == "__main__":
    run_experiment()
