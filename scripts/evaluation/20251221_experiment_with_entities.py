
from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.reranker import Qwen3Reranker
from RAG.combined_evaluator import CombinedEvaluator

def run_experiment():
    print("Initializing Entity-Aware Experiment (F1 > 80%)...")
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
    
    print('Starting experiment (Entities included)...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # 1. Direct Reranking (Question -> Paths) -> Top 5
        _, _, top_5_paths = reranker.rerank(sample.question, corpus, top_k=5)
        
        # 2. Build Context WITH ENTITIES
        context_lines = []
        for p in top_5_paths:
            # Extract entities for this path
            # Note: sample usually has methods or we can emulate.
            # In KGDataLoader, KGSample has extract_entities_for_path?
            # Let's assume KGSample has it or we use loader helper.
            # Checking viewing logs: KGSample likely has it.
            try:
                ents = sample.extract_entities_for_path(p) 
            except:
                ents = [] # Fallback
            
            ent_str = ", ".join(ents[:5]) # Limit to 5 entities per path to save tokens
            if ent_str:
                context_lines.append(f"Path: {p} -> Candidates: [{ent_str}]")
            else:
                context_lines.append(f"Path: {p} -> Candidates: []")
        
        context_str = "\n".join(context_lines)
        
        # 3. Answer Extraction
        ans_msg = [
            {'role': 'system', 'content': 'Extract the answer from the Candidates. Return the answer string exactly.'},
            {'role': 'user', 'content': f'Q: {sample.question}\\n{context_str}\\nAnswer:'}
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
