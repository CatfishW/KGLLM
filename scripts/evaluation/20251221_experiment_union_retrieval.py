
from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.reranker import Qwen3Reranker
from RAG.combined_evaluator import CombinedEvaluator

def run_experiment():
    print("Initializing Union Retrieval Experiment (F1 > 80%)...")
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
    
    print('Starting experiment (Union Q+HyDE with Entities)...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # 1. Question Rerank -> Top 10
        _, _, q_top_10 = reranker.rerank(sample.question, corpus, top_k=10)
        
        # 2. HyDE Generation
        hyde_msg = [{'role': 'user', 'content': f'Predict the likely Knowledge Graph path for: {sample.question}'}]
        hyde_pred = llm.chat(hyde_msg, max_tokens=32, temperature=0.5).strip()
        
        # 3. HyDE Rerank -> Top 10
        _, _, h_top_10 = reranker.rerank(hyde_pred, corpus, top_k=10)
        
        # 4. Union
        union_paths = list(dict.fromkeys(q_top_10 + h_top_10)) # unique stable order
        # Limit total to passed context (say 15 max)
        union_paths = union_paths[:15]
        
        # 5. Build Context WITH ENTITIES
        context_lines = []
        for p in union_paths:
            try:
                ents = sample.extract_entities_for_path(p) 
            except:
                ents = []
            
            ent_str = ", ".join(ents[:5])
            if ent_str:
                context_lines.append(f"Path: {p} -> Candidates: [{ent_str}]")
        
        # If no paths had entities, context is empty
        if not context_lines:
            context_str = "No candidates found."
        else:
            context_str = "\n".join(context_lines)
        
        # 6. Answer Extraction
        ans_msg = [
            {'role': 'system', 'content': 'You are an expert answer extractor. Review the candidate paths and their entities. Select the answer that best fits the question. Return the answer string exactly.'},
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
