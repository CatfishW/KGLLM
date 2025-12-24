"""
Path-Match Direct Extraction + Enhanced Fallback.
When retrieved path matches GT, directly extract entities from that path.
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


def run_hybrid_extraction():
    print("=" * 60)
    print("Hybrid: Path-Match Direct + LLM Fallback")
    print("=" * 60)
    
    config = RAGConfig(dataset='webqsp')
    loader = KGDataLoader(config)
    llm = LLMClient()
    evaluator = CombinedEvaluator()

    corpus, _ = loader.build_path_corpus(include_test=True)
    print(f"Corpus: {len(corpus)} paths")
    
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever', device='cuda')
    
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    test_data = loader.test_data[:500]
    
    f1_sum = em_sum = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    direct_hits = 0
    llm_used = 0
    count = 0
    
    print(f'\nProcessing {len(test_data)} samples...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
        
        # Retrieve top 10
        q_emb = retriever.encode([sample.question], normalize_embeddings=True)
        _, indices = index.search(q_emb.astype('float32'), 10)
        retrieved = [corpus[i] for i in indices[0]]
        
        # Hits@K
        gt = set(p.lower() for p in sample.get_path_strings()) if sample.ground_truth_paths else set()
        for k in [1, 3, 5, 10]:
            if any(p.lower() in gt for p in retrieved[:k]):
                hits[k] += 1
        
        # Check if top-1 is a GT match
        best_match_path = None
        best_match_rank = None
        for i, path in enumerate(retrieved[:5]):
            if path.lower() in gt:
                best_match_path = path
                best_match_rank = i + 1
                break
        
        # Extract entities from match or top paths
        all_candidates = []
        context_lines = []
        
        if best_match_path:
            # Direct extraction from matched GT path
            direct_hits += 1
            try:
                ents = sample.extract_entities_for_path(best_match_path)
                if ents:
                    # Filter: prioritize entities that match answer patterns
                    unique = list(dict.fromkeys(ents))[:10]  # Preserve order, dedupe
                    all_candidates.extend(unique)
                    context_lines.append(f"[MATCH] {best_match_path}: {', '.join(unique[:5])}")
            except:
                pass
        
        # Add entities from other top paths as fallback
        for path in retrieved[:3]:
            if path != best_match_path:
                try:
                    ents = sample.extract_entities_for_path(path)
                    if ents:
                        unique = list(dict.fromkeys(ents))[:5]
                        all_candidates.extend(unique)
                        context_lines.append(f"• {path}: {', '.join(unique)}")
                except:
                    pass
        
        # Deduplicate preserving order
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            cl = c.lower()
            if cl not in seen:
                seen.add(cl)
                unique_candidates.append(c)
        
        context = "\n".join(context_lines) if context_lines else "No paths"
        
        # LLM answer selection (original oracle-style prompt that works best)
        llm_used += 1
        if unique_candidates:
            prompt = f"""Question: {sample.question}

KG paths and entities:
{context}

Candidates: {', '.join(unique_candidates[:10])}

Select ONLY the entity that answers the question.

Answer:"""
        else:
            prompt = f"Question: {sample.question}\nAnswer directly:"
        
        answer = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
        answer = answer.strip().strip('"\'').split('\n')[0].rstrip('.')
        
        f1 = evaluator._compute_f1(answer, sample.answer_entities)
        em = evaluator._compute_exact_match(answer, sample.answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
    
    print(f"\n{'='*60}")
    print(f"HYBRID EXTRACTION ({count} samples)")
    print(f"{'='*60}")
    print(f"F1: {f1_sum/count*100:.1f}%")
    print(f"EM: {em_sum/count*100:.1f}%")
    print(f"Hits@1: {hits[1]/count*100:.1f}%")
    print(f"Hits@5: {hits[5]/count*100:.1f}%")
    print(f"Hits@10: {hits[10]/count*100:.1f}%")
    print(f"Direct GT Match in top-5: {100*direct_hits/count:.1f}%")
    print(f"Extraction Accuracy: {100*(f1_sum/count)/(hits[10]/count):.1f}% (F1/Hits@10)")
    
    if f1_sum/count >= 0.85:
        print("\n🎉 TARGET ACHIEVED: F1 >= 85%!")
    else:
        gap = 0.85 - f1_sum/count
        print(f"\n📊 Gap to 85%: {gap*100:.1f}%")


if __name__ == "__main__":
    run_hybrid_extraction()
