"""
Test v2 retriever (10 epochs) without reranker.
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

def run_evaluation():
    print("=" * 60)
    print("V2 Retriever (10 epochs) - NO RERANKER")
    print("=" * 60)
    
    config = RAGConfig(dataset='webqsp')
    loader = KGDataLoader(config)
    llm = LLMClient()
    evaluator = CombinedEvaluator()

    corpus, _ = loader.build_path_corpus(include_test=True)
    print(f"Corpus: {len(corpus)} paths")
    
    # Use v2 retriever
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever_v2', device='cuda')
    
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    test_data = loader.test_data[:500]
    
    f1_sum = em_sum = 0
    hits_at_1 = hits_at_3 = hits_at_5 = hits_at_10 = 0
    count = 0
    
    print(f'\nProcessing {len(test_data)} samples...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # Direct retrieval - top 10
        q_emb = retriever.encode([sample.question], normalize_embeddings=True)
        _, indices = index.search(q_emb.astype('float32'), 10)
        retrieved = [corpus[i] for i in indices[0]]
        
        # Hits@K
        gt = set(p.lower() for p in sample.get_path_strings()) if sample.ground_truth_paths else set()
        retrieved_lower = [p.lower() for p in retrieved]
        
        if any(p in gt for p in retrieved_lower[:1]): hits_at_1 += 1
        if any(p in gt for p in retrieved_lower[:3]): hits_at_3 += 1
        if any(p in gt for p in retrieved_lower[:5]): hits_at_5 += 1
        if any(p in gt for p in retrieved_lower[:10]): hits_at_10 += 1
        
        # Extract entities from top 5 paths
        all_candidates = []
        context_lines = []
        
        for path in retrieved[:5]:
            try:
                ents = sample.extract_entities_for_path(path)
                if ents:
                    seen = set()
                    unique = []
                    for e in ents:
                        if e.lower() not in seen:
                            seen.add(e.lower())
                            unique.append(e)
                    unique = unique[:5]
                    all_candidates.extend(unique)
                    context_lines.append(f"• {path}: {', '.join(unique)}")
            except:
                pass
        
        # Deduplicate candidates
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique_candidates.append(c)
        
        context = "\n".join(context_lines) if context_lines else "No paths"
        
        # LLM answer selection
        if unique_candidates:
            prompt = f"""Question: {sample.question}

KG paths and entities:
{context}

Candidates: {', '.join(unique_candidates[:12])}

Select ONLY the entity name(s) that answer the question.

Answer:"""
        else:
            prompt = f"Q: {sample.question}\nAnswer:"
        
        answer = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
        answer = answer.strip().strip('"\'').split('\n')[0].rstrip('.')
        
        f1 = evaluator._compute_f1(answer, sample.answer_entities)
        em = evaluator._compute_exact_match(answer, sample.answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
        
    print(f'\n{"="*60}')
    print(f'RESULTS (V2 Retriever, {count} samples)')
    print(f'{"="*60}')
    print(f'F1 Score: {100*f1_sum/count:.1f}%')
    print(f'Exact Match: {100*em_sum/count:.1f}%')
    print(f'Hits@1: {100*hits_at_1/count:.1f}%')
    print(f'Hits@3: {100*hits_at_3/count:.1f}%')
    print(f'Hits@5: {100*hits_at_5/count:.1f}%')
    print(f'Hits@10: {100*hits_at_10/count:.1f}%')
    
    if f1_sum/count >= 0.85:
        print("\n🎉 TARGET ACHIEVED: F1 >= 85%!")
    else:
        gap = 0.85 - f1_sum/count
        print(f"\n📊 Gap to 85%: {gap*100:.1f}%")

if __name__ == "__main__":
    run_evaluation()
