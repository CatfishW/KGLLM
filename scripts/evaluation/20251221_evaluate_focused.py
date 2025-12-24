"""
Optimized entity selection with LLM guidance.
Uses top-ranked path entities as primary candidates with better filtering.
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

def normalize_answer(answer: str) -> str:
    """Clean up LLM answer."""
    # Remove common prefixes
    answer = re.sub(r'^(The answer is:?|Answer:?)\s*', '', answer, flags=re.IGNORECASE)
    # Take first line
    answer = answer.strip().split('\n')[0]
    # Remove trailing punctuation
    answer = answer.strip().rstrip('.')
    # Remove quotes
    answer = answer.strip('"\'')
    return answer

def run_evaluation():
    print("=" * 60)
    print("Optimized Entity Selection with Better Prompting")
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
    count = 0
    
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
        
        if any(p in gt for p in reranked_lower[:1]): hits_at_1 += 1
        if any(p in gt for p in reranked_lower[:5]): hits_at_5 += 1
        if any(p in gt for p in reranked_lower[:10]): hits_at_10 += 1
        
        # 4. Extract entities from top-3 paths only (more focused)
        candidates_by_path = []
        all_candidates = []
        
        for path in reranked[:3]:  # Top 3 only
            try:
                ents = sample.extract_entities_for_path(path)
                if ents:
                    seen = set()
                    unique = []
                    for e in ents:
                        if e.lower() not in seen:
                            seen.add(e.lower())
                            unique.append(e)
                    unique = unique[:3]  # Max 3 per path
                    candidates_by_path.append({'path': path, 'entities': unique})
                    all_candidates.extend(unique)
            except:
                pass
        
        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique_candidates.append(c)
        
        # 5. Structured prompt for answer selection
        if unique_candidates:
            # Build structured context
            context_parts = []
            for item in candidates_by_path:
                context_parts.append(f"- {item['path']}: {', '.join(item['entities'])}")
            context = "\n".join(context_parts)
            
            prompt = f"""Question: {sample.question}

Knowledge Graph paths and their entities:
{context}

Available candidates: {', '.join(unique_candidates[:10])}

Based on the question and paths above, which candidate(s) answer the question?
Give ONLY the entity name(s), separated by comma if multiple. No explanation.

Answer:"""
        else:
            # Fallback
            prompt = f"""Question: {sample.question}

Paths: {', '.join(reranked[:3])}

Answer the question directly with the entity name(s).

Answer:"""
        
        answer = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
        answer = normalize_answer(answer)
        
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
    
    if f1_sum/count >= 0.85:
        print("\n🎉 TARGET ACHIEVED: F1 >= 85%!")
    else:
        gap = 0.85 - f1_sum/count
        print(f"\n📊 Gap to 85%: {gap*100:.1f}%")

if __name__ == "__main__":
    run_evaluation()
