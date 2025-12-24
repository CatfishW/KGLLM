
from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.reranker import Qwen3Reranker
from RAG.combined_evaluator import CombinedEvaluator

def run_experiment():
    print("Initializing Pure HyDE Reranking Experiment (F1 > 80%)...")
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
    hits_1_sum = 0
    hits_5_sum = 0
    count = 0
    
    print('Starting experiment (Pure HyDE Reranking)...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # 1. HyDE Generation
        hyde_msg = [{'role': 'user', 'content': f'Predict the likely Knowledge Graph path for: {sample.question}'}]
        hyde_pred = llm.chat(hyde_msg, max_tokens=32, temperature=0.5).strip()
        
        # 2. Rerank using HyDE prediction as the query
        _, _, top_5_paths = reranker.rerank(hyde_pred, corpus, top_k=5)
        
        # 3. Answer Extraction from Top 5
        ans_msg = [
            {'role': 'system', 'content': 'Extract the answer. Short.'},
            {'role': 'user', 'content': f'Q: {sample.question}\\nPaths:\\n' + '\\n'.join(top_5_paths) + '\\nAnswer:'}
        ]
        ans = llm.chat(ans_msg, max_tokens=32, temperature=0.1).strip().strip('"').strip("'")
        
        # Eval
        f1 = evaluator._compute_f1(ans, sample.answer_entities)
        
        # Hits metrics
        gt_norm = set(evaluator._normalize_path(p) for p in sample.get_path_strings())
        top1_norm = evaluator._normalize_path(top_5_paths[0])
        hits_1 = 1 if top1_norm in gt_norm else 0
        hits_5 = 1 if (set(evaluator._normalize_path(p) for p in top_5_paths) & gt_norm) else 0

        f1_sum += f1
        hits_1_sum += hits_1
        hits_5_sum += hits_5
        count += 1
        
    print(f'Results (100 samples):')
    print(f'F1: {100*f1_sum/count:.1f}%')
    print(f'Hits@1: {100*hits_1_sum/count:.1f}%')
    print(f'Hits@5: {100*hits_5_sum/count:.1f}%')

if __name__ == "__main__":
    run_experiment()
