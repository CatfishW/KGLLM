
import re
from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.reranker import Qwen3Reranker
from RAG.combined_evaluator import CombinedEvaluator

def run_experiment():
    print("Initializing LLM Listwise Re-ordering Experiment (F1 > 80%)...")
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
    
    print('Starting experiment (LLM Re-ordering Top 10 -> Top 5)...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # 1. Direct Reranking (Question -> Paths) -> Top 10
        _, _, top_10_paths = reranker.rerank(sample.question, corpus, top_k=10)
        
        # 2. LLM Re-ordering
        paths_str = ""
        paths_map = {}
        for i, path in enumerate(top_10_paths):
            idx = i + 1
            paths_str += f"{idx}. {path}\\n"
            paths_map[idx] = path
            
        select_msg = [
            {'role': 'system', 'content': 'You are an expert Knowledge Graph Path Selector. Select the 3 updated indices (1-10) of the paths that are MOST RELEVANT for answering the question. Return numbers separated by commas.'},
            {'role': 'user', 'content': f'Question: {sample.question}\\n\\nCandidate Paths:\\n{paths_str}\\nRelevant Indices:'}
        ]
        
        selection_resp = llm.chat(select_msg, max_tokens=16, temperature=0.1).strip()
        
        # Parse indices
        matches = re.findall(r'\d+', selection_resp)
        selected_indices = [int(m) for m in matches if 1 <= int(m) <= 10]
        
        # Construct New Top 5
        # Priority 1: Selected indices
        # Priority 2: Original remaining paths (to fill up to 5)
        
        new_top_5 = []
        seen = set()
        
        # Add selected
        for idx in selected_indices:
            path = paths_map.get(idx)
            if path and path not in seen:
                new_top_5.append(path)
                seen.add(path)
        
        # Fill rest from top_10 (original order)
        for path in top_10_paths:
            if len(new_top_5) >= 5:
                break
            if path not in seen:
                new_top_5.append(path)
                seen.add(path)
                
        # 3. Answer Extraction from New Top 5
        ans_msg = [
            {'role': 'system', 'content': 'Extract the answer. Short.'},
            {'role': 'user', 'content': f'Q: {sample.question}\\nPaths:\\n' + '\\n'.join(new_top_5) + '\\nAnswer:'}
        ]
        ans = llm.chat(ans_msg, max_tokens=32, temperature=0.1).strip().strip('"').strip("'")
        
        # Eval
        f1 = evaluator._compute_f1(ans, sample.answer_entities)
        f1_sum += f1
        count += 1
        
    print(f'Results (100 samples):')
    print(f'F1: {100*f1_sum/count:.1f}%')

if __name__ == "__main__":
    run_experiment()
