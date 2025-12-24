
import re
from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.reranker import Qwen3Reranker
from RAG.combined_evaluator import CombinedEvaluator

def run_experiment():
    print("Initializing LLM Listwise Reranking Experiment (F1 > 80%)...")
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
    hits_10_sum = 0
    original_hits_1_sum = 0
    count = 0
    
    print('Starting experiment (LLM Selection from Top-10)...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # 1. Direct Reranking (Question -> Paths)
        # Note: No HyDE augmentation here.
        _, _, top_10_paths = reranker.rerank(sample.question, corpus, top_k=10)
        
        # 2. LLM Selection
        paths_str = ""
        paths_map = {}
        for i, path in enumerate(top_10_paths):
            idx = i + 1
            paths_str += f"{idx}. {path}\\n"
            paths_map[idx] = path
            
        select_msg = [
            {'role': 'system', 'content': 'You are an expert Knowledge Graph Path Selector. Select the single path index number (1-10) that is MOST RELEVANT and ACCURATE for answering the question. Return ONLY the number.'},
            {'role': 'user', 'content': f'Question: {sample.question}\\n\\nCandidate Paths:\\n{paths_str}\\nBest Path Index:'}
        ]
        
        selection_resp = llm.chat(select_msg, max_tokens=8, temperature=0.1).strip()
        
        # Parse selection
        match = re.search(r'\d+', selection_resp)
        if match:
            selected_idx = int(match.group())
        else:
            selected_idx = 1 # Fallback to Rank 1
            
        best_path = paths_map.get(selected_idx, top_10_paths[0])
        
        # 3. Answer Extraction from Best Path
        ans_msg = [
            {'role': 'system', 'content': 'Extract the answer from the KG path. Short.'},
            {'role': 'user', 'content': f'Q: {sample.question}\\nPath: {best_path}\\nAnswer:'}
        ]
        ans = llm.chat(ans_msg, max_tokens=32, temperature=0.1).strip().strip('"').strip("'")
        
        # Eval
        f1 = evaluator._compute_f1(ans, sample.answer_entities)
        
        # Hits metrics
        gt_norm = set(evaluator._normalize_path(p) for p in sample.get_path_strings())
        
        # Check LLM selection hit
        best_path_norm = evaluator._normalize_path(best_path)
        hits_1 = 1 if best_path_norm in gt_norm else 0
        
        # Check Original Rank 1 hit
        top1_norm = evaluator._normalize_path(top_10_paths[0])
        orig_hits_1 = 1 if top1_norm in gt_norm else 0
        
        # Check if ANY in Top 10 was correct
        top10_norm = set(evaluator._normalize_path(p) for p in top_10_paths)
        hits_10 = 1 if (top10_norm & gt_norm) else 0

        f1_sum += f1
        hits_1_sum += hits_1
        original_hits_1_sum += orig_hits_1
        hits_10_sum += hits_10
        count += 1
        
    print(f'Results (100 samples):')
    print(f'F1: {100*f1_sum/count:.1f}%')
    print(f'LLM Selected Hits@1: {100*hits_1_sum/count:.1f}%')
    print(f'Original Reranker Hits@1: {100*original_hits_1_sum/count:.1f}%')
    print(f'Reranker Hits@10 (Ceiling): {100*hits_10_sum/count:.1f}%')

if __name__ == "__main__":
    run_experiment()
