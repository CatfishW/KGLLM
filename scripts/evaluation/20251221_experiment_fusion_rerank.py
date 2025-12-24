
import time
import torch
import numpy as np
from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.reranker import Qwen3Reranker
from RAG.combined_evaluator import CombinedEvaluator

def run_experiment():
    print("Initializing Multi-View Reranking Experiment (F1 > 80%)...")
    config = RAGConfig(dataset='webqsp')
    loader = KGDataLoader(config)
    llm = LLMClient()
    reranker = Qwen3Reranker(config)
    reranker.load()
    evaluator = CombinedEvaluator()

    # Full Corpus
    corpus, _ = loader.build_path_corpus(include_test=True)
    print(f"Corpus Size: {len(corpus)}")

    test_data = loader.test_data[:100]
    
    results = []
    
    # Weights for fusion
    # Question weight vs HyDE weight
    W_Q = 0.5
    W_H = 0.5

    f1_sum = 0
    hits_1_sum = 0
    hits_5_sum = 0
    count = 0

    print(f'Starting experiment (W_Q={W_Q}, W_H={W_H})...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # 1. HyDE Generation
        hyde_msg = [{'role': 'user', 'content': f'Predict the likely Knowledge Graph path for: {sample.question}'}]
        hyde_pred = llm.chat(hyde_msg, max_tokens=32, temperature=0.5).strip()
        
        # 2. Score View 1: Question -> Paths
        # We need raw scores. reranker.rerank returns sorted list.
        # We need to access the internal scoring method or map back.
        # Efficient way: Rerank returns scores. We need to align them.
        
        # Let's batch compute manually to get aligned scores
        # Qwen3Reranker.compute_score(query, docs) ? No, use rerank but we need indices.
        # Actually Qwen3Reranker.rerank returns (reranked_paths, scores)
        # We want scores for ALL paths in FIXED order to sum them.
        
        # We will use the reranker model directly or a helper if available. 
        # Reranker.predict(pairs) is standard.
        # Let's see Qwen3Reranker implementation. 
        # Assuming we can use internal `model.compute_score` or similar.
        # For now, let's just use `rerank` which calls the model.
        # To fuse, we need the score for every path.
        
        # View 1
        _, scores_q, _ = reranker.rerank(sample.question, corpus, top_k=len(corpus))
        # Wait, rerank sorts them. We lose the index alignment unless we map back.
        # Actually `rerank` output is sorted. 
        # Map: Path -> Score
        score_map_q = {p: s for p, s in zip(_, scores_q)}
        
        # View 2
        _, scores_h, _ = reranker.rerank(hyde_pred, corpus, top_k=len(corpus))
        score_map_h = {p: s for p, s in zip(_, scores_h)}
        
        # 3. Fuse Scores
        fused_scores = {}
        for p in corpus:
            s_q = score_map_q.get(p, -100) # Should exist
            s_h = score_map_h.get(p, -100)
            fused_scores[p] = W_Q * s_q + W_H * s_h
            
        # Sort by fused score
        sorted_paths = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        top_paths = sorted_paths[:5]
        
        # 4. Answer Extraction
        ans_msg = [
            {'role': 'system', 'content': 'Extract the answer. Short.'},
            {'role': 'user', 'content': f'Q: {sample.question}\\nPaths:\\n' + '\\n'.join(top_paths) + '\\nAnswer:'}
        ]
        ans = llm.chat(ans_msg, max_tokens=32, temperature=0.1).strip().strip('"').strip("'")
        
        # Eval
        f1 = evaluator._compute_f1(ans, sample.answer_entities)
        
        # Hits
        gt_norm = set(evaluator._normalize_path(p) for p in sample.get_path_strings())
        top1_norm = evaluator._normalize_path(top_paths[0])
        hits_1 = 1 if top1_norm in gt_norm else 0
        
        top5_norm = set(evaluator._normalize_path(p) for p in top_paths)
        hits_5 = 1 if (top5_norm & gt_norm) else 0

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
