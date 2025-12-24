"""
CWQ Hybrid Retrieval: BM25 + Dense (Entity-Enhanced Paths)
Hypothesis: Dense alone is 57% Hits@1. Adding BM25 should help with lexical matching.
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from RAG.llm_client import LLMClient
from RAG.combined_evaluator import CombinedEvaluator


def parse_cwq_paths(shortest_gt_paths_str):
    if not shortest_gt_paths_str or shortest_gt_paths_str == '[]':
        return []
    try:
        fixed = shortest_gt_paths_str.replace('"s ', "'s ").replace('"', '"').replace('"', '"')
        paths = json.loads(fixed)
        result = []
        for p in paths:
            if 'entities' in p and 'relations' in p:
                entities = p['entities']
                relations = p['relations']
                
                parts = []
                for i in range(len(relations)):
                    if i < len(entities):
                        parts.append(entities[i])
                    parts.append(relations[i])
                if len(entities) > len(relations):
                    parts.append(entities[-1])
                
                full_path = " -> ".join(parts)
                result.append(full_path)
        return result
    except:
        return []

def extract_entities_from_path(path_str):
    if not path_str:
        return []
    parts = path_str.split(' -> ')
    return parts[::2]

def hybrid_retrieve(query, dense_retriever, dense_index, bm25, corpus, top_k=10, alpha=0.5):
    """
    Hybrid retrieval: alpha * dense + (1-alpha) * BM25
    alpha=0.5 means equal weight
    """
    # Dense retrieval
    q_emb = dense_retriever.encode([query], normalize_embeddings=True)
    dense_scores, dense_indices = dense_index.search(q_emb.astype('float32'), top_k*2)
    dense_scores = dense_scores[0]
    dense_indices = dense_indices[0]
    
    # BM25 retrieval
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Normalize scores to [0, 1]
    dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-9)
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
    
    # Combine scores
    combined_scores = {}
    for idx, score in zip(dense_indices, dense_norm):
        combined_scores[idx] = alpha * score
    
    for idx, score in enumerate(bm25_norm):
        if idx in combined_scores:
            combined_scores[idx] += (1 - alpha) * score
        else:
            combined_scores[idx] = (1 - alpha) * score
    
    # Sort by combined score
    sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]
    return [corpus[i] for i in sorted_indices]

def run_hybrid_evaluation():
    print("=" * 60)
    print("CWQ HYBRID Retrieval Evaluation")
    print("BM25 + Dense (alpha=0.5)")
    print("=" * 60)
    
    train = pd.read_parquet('../../Data/CWQ/shortest_paths/train.parquet')
    test = pd.read_parquet('../../Data/CWQ/shortest_paths/test.parquet')
    
    # Build corpus
    all_paths = set()
    for _, row in train.iterrows():
        for p in parse_cwq_paths(row['shortest_gt_paths']):
            if p: all_paths.add(p)
    for _, row in test.iterrows():
        for p in parse_cwq_paths(row['shortest_gt_paths']):
            if p: all_paths.add(p)
    
    corpus = list(all_paths)
    print(f"Corpus: {len(corpus)} paths")
    
    print("\nLoading dense retriever...")
    dense_retriever = SentenceTransformer('../models/finetuned_retriever_cwq_v2_fix', device='cuda')
    
    print("Building dense index...")
    embeddings = dense_retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    dense_index = faiss.IndexFlatIP(embeddings.shape[1])
    dense_index.add(embeddings.astype('float32'))
    
    print("Building BM25 index...")
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    llm = LLMClient()
    evaluator = CombinedEvaluator()
    
    f1_sum = em_sum = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    recall = {10: 0}
    count = 0
    
    # Test on 500 samples for comparison
    test_sample = test.head(500)
    print(f"\nEvaluating on {len(test_sample)} samples...")
    
    for _, row in tqdm(test_sample.iterrows(), total=len(test_sample)):
        question = row['question']
        answer_entities = row['a_entity'] if isinstance(row['a_entity'], list) else [row['a_entity']]
        
        if not answer_entities:
            continue
        
        # Hybrid Retrieve
        retrieved = hybrid_retrieve(question, dense_retriever, dense_index, bm25, corpus, top_k=10, alpha=0.5)
        
        # Hits
        gt_paths = set(p.lower() for p in parse_cwq_paths(row['shortest_gt_paths']))
        for k in [1, 3, 5, 10]:
            if any(p.lower() in gt_paths for p in retrieved[:k]):
                hits[k] += 1
        
        # Extract candidates
        retrieved_entities = []
        for p in retrieved:
            ents = extract_entities_from_path(p)
            for e in ents:
                if e not in retrieved_entities:
                    retrieved_entities.append(e)
        
        # Recall
        norm_answers = {evaluator._normalize_answer(a) for a in answer_entities}
        norm_candidates = {evaluator._normalize_answer(c) for c in retrieved_entities}
        if not norm_candidates.isdisjoint(norm_answers):
            recall[10] += 1
        
        # Context
        context_lines = []
        for i, path in enumerate(retrieved[:5]):
            context_lines.append(f"[Path {i+1}] {path}")
        context = "\n".join(context_lines)
        
        candidates = retrieved_entities[:20]
        
        if candidates:
            prompt = f"""Answer the question by selecting from the candidates.

Question: {question}

Knowledge Graph Paths:
{context}

Candidate Entities: {', '.join(candidates)}

Instructions:
- Select the ONE entity that best answers the question
- Only output the entity name

Answer:"""
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        try:
            answer = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
            answer = answer.strip().strip('"\'').split('\n')[0].rstrip('.')
        except:
            answer = ""
        
        f1 = evaluator._compute_f1(answer, answer_entities)
        em = evaluator._compute_exact_match(answer, answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
        
        if count % 100 == 0:
            print(f"  Step {count}: F1={f1_sum/count*100:.1f}%, Hits@1={hits[1]/count*100:.1f}%, Recall={recall[10]/count*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"CWQ HYBRID RESULTS ({count} samples)")
    print(f"{'='*60}")
    print(f"F1: {f1_sum/count*100:.1f}%")
    print(f"EM: {em_sum/count*100:.1f}%")
    print(f"Hits@1: {hits[1]/count*100:.1f}%")
    print(f"Hits@10: {hits[10]/count*100:.1f}%")
    print(f"Candidate Recall: {recall[10]/count*100:.1f}%")
    
    print(f"\nComparison:")
    print(f"  D-RAG:      70.3% Hits@1, 63.8% F1")
    print(f"  Dense only: 57.4% Hits@1, 60.3% F1")
    print(f"  Hybrid:     {hits[1]/count*100:.1f}% Hits@1, {f1_sum/count*100:.1f}% F1")
    
    # Save results
    with open('cwq_hybrid_results.json', 'w') as f:
        json.dump({
            'f1': f1_sum/count*100,
            'em': em_sum/count*100,
            'hits@1': hits[1]/count*100,
            'hits@10': hits[10]/count*100,
            'recall@10': recall[10]/count*100,
            'count': count,
            'alpha': 0.5,
            'method': 'BM25 + Dense (equal weight)'
        }, f, indent=2)
    print(f"\nResults saved to cwq_hybrid_results.json")

if __name__ == "__main__":
    run_hybrid_evaluation()
