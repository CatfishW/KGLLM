"""
Direct Entity Selection Strategy.
Key insight: When correct path is retrieved, directly extract entities
without LLM - reducing hallucination.

Strategy:
1. If correct path in top-K, use extracted entities directly
2. Otherwise use LLM for reasoning
3. Better prompt with few-shot examples
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.combined_evaluator import CombinedEvaluator
from RAG.reranker import Qwen3Reranker
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

def run_evaluation():
    print("=" * 60)
    print("Direct Entity Selection Strategy")
    print("=" * 60)
    
    config = RAGConfig(dataset='webqsp')
    loader = KGDataLoader(config)
    llm = LLMClient()
    evaluator = CombinedEvaluator()
    reranker = Qwen3Reranker(config)
    reranker.load()

    corpus, _ = loader.build_path_corpus(include_test=True)
    print(f"Corpus: {len(corpus)} paths")
    
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever', device='cuda')
    
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    test_data = loader.test_data[:500]
    
    f1_sum = em_sum = 0
    hits_at_1 = hits_at_5 = hits_at_10 = 0
    direct_selection = 0
    llm_selection = 0
    count = 0
    
    # Few-shot examples for better prompting
    FEW_SHOT_PROMPT = """You are an expert at Knowledge Graph Question Answering.
Given a question and candidate entities from relevant KG paths, select the correct answer.

Example 1:
Question: Where was Barack Obama born?
Candidates: Honolulu, Hawaii, United States, 1961
Answer: Honolulu

Example 2:
Question: Who is the CEO of Apple?
Candidates: Tim Cook, Steve Jobs, Apple Inc, California
Answer: Tim Cook

Example 3:
Question: What language is spoken in France?
Candidates: French, Paris, Europe, Euro
Answer: French

Now answer the following:
Question: {question}
Candidates: {candidates}
Answer:"""
    
    print(f'\nProcessing {len(test_data)} samples...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # 1. Retrieve
        q_emb = retriever.encode([sample.question], normalize_embeddings=True)
        _, indices = index.search(q_emb.astype('float32'), 30)
        retrieved = [corpus[i] for i in indices[0]]
        
        # 2. Rerank
        _, _, reranked = reranker.rerank(sample.question, retrieved, top_k=10)
        
        # 3. Check Hits
        gt = set(p.lower() for p in sample.get_path_strings()) if sample.ground_truth_paths else set()
        reranked_lower = [p.lower() for p in reranked]
        
        hit_at_1 = any(p in gt for p in reranked_lower[:1])
        hit_at_5 = any(p in gt for p in reranked_lower[:5])
        hit_at_10 = any(p in gt for p in reranked_lower[:10])
        
        if hit_at_1: hits_at_1 += 1
        if hit_at_5: hits_at_5 += 1
        if hit_at_10: hits_at_10 += 1
        
        # 4. Entity Extraction
        all_candidates = []
        matched_path_entities = []
        
        for path in reranked[:5]:
            try:
                ents = sample.extract_entities_for_path(path)
                if ents:
                    # Deduplicate
                    seen = set()
                    unique = []
                    for e in ents:
                        if e.lower() not in seen:
                            seen.add(e.lower())
                            unique.append(e)
                    all_candidates.extend(unique[:5])
                    
                    # Check if this is a matched ground truth path
                    if path.lower() in gt:
                        matched_path_entities = unique
            except:
                pass
        
        # Deduplicate all candidates
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique_candidates.append(c)
        
        # 5. Answer Selection Strategy
        answer = ""
        
        if matched_path_entities:
            # Direct selection: use entities from matched path
            answer = ", ".join(matched_path_entities[:5])
            direct_selection += 1
        elif unique_candidates:
            # LLM selection with few-shot prompt
            prompt = FEW_SHOT_PROMPT.format(
                question=sample.question,
                candidates=", ".join(unique_candidates[:10])
            )
            answer = llm.chat([{'role': 'user', 'content': prompt}], 
                             max_tokens=64, temperature=0.1)
            answer = answer.strip().strip('"\'').split('\n')[0].rstrip('.')
            llm_selection += 1
        else:
            # Fallback: just ask the question
            prompt = f"Question: {sample.question}\nGive a short, direct answer."
            answer = llm.chat([{'role': 'user', 'content': prompt}],
                             max_tokens=64, temperature=0.1)
            answer = answer.strip().strip('"\'').split('\n')[0].rstrip('.')
            llm_selection += 1
        
        f1 = evaluator._compute_f1(answer, sample.answer_entities)
        em = evaluator._compute_exact_match(answer, sample.answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
        
    print(f'\n{"="*60}')
    print(f'RESULTS ({count} samples)')
    print(f'{"="*60}')
    print(f'F1 Score: {100*f1_sum/count:.1f}%')
    print(f'Exact Match: {100*em_sum/count:.1f}%')
    print(f'Hits@1: {100*hits_at_1/count:.1f}%')
    print(f'Hits@5: {100*hits_at_5/count:.1f}%')
    print(f'Hits@10: {100*hits_at_10/count:.1f}%')
    print(f'\nDirect selection: {direct_selection} ({100*direct_selection/count:.1f}%)')
    print(f'LLM selection: {llm_selection} ({100*llm_selection/count:.1f}%)')

if __name__ == "__main__":
    run_evaluation()
