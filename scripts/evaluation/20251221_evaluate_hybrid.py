"""
Hybrid Retrieval: BM25 + Dense fusion for improved Hits@5.
Goal: Push Hits@5 from 85.4% to 90%+ for 85% F1 target.
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.combined_evaluator import CombinedEvaluator
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import re


def tokenize(text: str) -> list:
    """Simple tokenizer for BM25."""
    return re.findall(r'\w+', text.lower())


def run_hybrid_evaluation():
    print("=" * 60)
    print("Hybrid Retrieval: BM25 + Dense Fusion")
    print("=" * 60)
    
    config = RAGConfig(dataset='webqsp')
    loader = KGDataLoader(config)
    llm = LLMClient()
    evaluator = CombinedEvaluator()

    corpus, _ = loader.build_path_corpus(include_test=True)
    print(f"Corpus: {len(corpus)} paths")
    
    # Dense retriever
    print("\nLoading dense retriever...")
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever', device='cuda')
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    dense_index = faiss.IndexFlatIP(embeddings.shape[1])
    dense_index.add(embeddings.astype('float32'))
    
    # BM25
    print("Building BM25 index...")
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    print("  BM25 ready")
    
    test_data = loader.test_data[:500]
    
    f1_sum = em_sum = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    count = 0
    
    # Fusion parameters
    dense_weight = 0.7  # Dense is typically stronger
    bm25_weight = 0.3
    
    print(f"\nProcessing {len(test_data)} samples (dense={dense_weight}, bm25={bm25_weight})...")
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
        
        question = sample.question
        
        # Dense retrieval
        q_emb = retriever.encode([question], normalize_embeddings=True)
        dense_scores, dense_indices = dense_index.search(q_emb.astype('float32'), 30)
        
        # Create score dict for dense
        dense_score_dict = {corpus[idx]: score for idx, score in zip(dense_indices[0], dense_scores[0])}
        
        # BM25 retrieval
        tokenized_query = tokenize(question)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores to 0-1
        bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_normalized = bm25_scores / bm25_max
        
        # Fusion: combine dense and BM25 scores
        fusion_scores = {}
        for i, path in enumerate(corpus):
            dense_s = dense_score_dict.get(path, 0)
            bm25_s = bm25_normalized[i]
            fusion_scores[path] = dense_weight * dense_s + bm25_weight * bm25_s
        
        # Sort by fusion score
        sorted_paths = sorted(fusion_scores.keys(), key=lambda x: fusion_scores[x], reverse=True)
        retrieved = sorted_paths[:10]
        
        # Hits@K
        gt = set(p.lower() for p in sample.get_path_strings()) if sample.ground_truth_paths else set()
        for k in [1, 3, 5, 10]:
            if any(p.lower() in gt for p in retrieved[:k]):
                hits[k] += 1
        
        # Entity extraction from top 3
        all_candidates = []
        context_lines = []
        
        for path in retrieved[:3]:
            try:
                ents = sample.extract_entities_for_path(path)
                if ents:
                    seen = set()
                    unique = [e for e in ents if e.lower() not in seen and not seen.add(e.lower())][:5]
                    all_candidates.extend(unique)
                    context_lines.append(f"• {path}: {', '.join(unique)}")
            except:
                pass
        
        seen = set()
        unique_candidates = [c for c in all_candidates if c.lower() not in seen and not seen.add(c.lower())]
        
        context = "\n".join(context_lines) if context_lines else "No paths"
        
        # Answer extraction
        if unique_candidates:
            prompt = f"""Question: {question}

KG paths and entities:
{context}

Candidates: {', '.join(unique_candidates[:10])}

Select ONLY the entity that answers the question.

Answer:"""
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        answer = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
        answer = answer.strip().strip('"\'').split('\n')[0].rstrip('.')
        
        f1 = evaluator._compute_f1(answer, sample.answer_entities)
        em = evaluator._compute_exact_match(answer, sample.answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
    
    print(f"\n{'='*60}")
    print(f"HYBRID RESULTS ({count} samples)")
    print(f"{'='*60}")
    print(f"F1: {f1_sum/count*100:.1f}%")
    print(f"EM: {em_sum/count*100:.1f}%")
    print(f"Hits@1: {hits[1]/count*100:.1f}%")
    print(f"Hits@3: {hits[3]/count*100:.1f}%")
    print(f"Hits@5: {hits[5]/count*100:.1f}%")
    print(f"Hits@10: {hits[10]/count*100:.1f}%")
    
    if f1_sum/count >= 0.85:
        print("\n🎉 TARGET ACHIEVED: F1 >= 85%!")


if __name__ == "__main__":
    run_hybrid_evaluation()
