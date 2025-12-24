
from tqdm import tqdm
from collections import Counter
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.reranker import Qwen3Reranker
from RAG.combined_evaluator import CombinedEvaluator

def run_experiment():
    print("Initializing Self-Consistency Voting Experiment (F1 > 80%)...")
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
    
    print('Starting experiment (Question-Only + 5-Vote SC, Temp 0.7)...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # 1. Direct Reranking (Question -> Paths) -> Top 5
        # Matches the 73.4% baseline logic
        _, _, top_5_paths = reranker.rerank(sample.question, corpus, top_k=5)
        
        # 2. Vote (5 times)
        answers = []
        context_str = '\\n'.join([f'- {p}' for p in top_5_paths])
        
        ans_msg = [
            {'role': 'system', 'content': 'Extract the answer from the KG paths. SHORT answers only (1-5 words).'},
            {'role': 'user', 'content': f'Q: {sample.question}\\nSelected Paths:\\n{context_str}\\nAnswer:'}
        ]
        
        for _ in range(5):
            # Use high temp for diversity
            ans = llm.chat(ans_msg, max_tokens=32, temperature=0.7).strip().strip('"').strip("'").rstrip('.')
            answers.append(evaluator._normalize_answer(ans))
            
        # Majority Vote
        counter = Counter(answers)
        final_answer = counter.most_common(1)[0][0]
        
        # Eval
        f1 = evaluator._compute_f1(final_answer, sample.answer_entities)
        f1_sum += f1
        count += 1
        
    print(f'Results (100 samples):')
    print(f'F1: {100*f1_sum/count:.1f}%')

if __name__ == "__main__":
    run_experiment()
