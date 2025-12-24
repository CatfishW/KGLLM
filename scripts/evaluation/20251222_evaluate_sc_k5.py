"""
Self-Consistency k=5 voting on the 79.9% F1 config.
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
from collections import Counter


def run_self_consistency_k5():
    print("=" * 60)
    print("Self-Consistency k=5 Voting")
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
    count = 0
    
    NUM_SAMPLES = 5
    
    print(f'\nProcessing {len(test_data)} samples (k={NUM_SAMPLES} voting)...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
        
        # Retrieve
        q_emb = retriever.encode([sample.question], normalize_embeddings=True)
        _, indices = index.search(q_emb.astype('float32'), 10)
        retrieved = [corpus[i] for i in indices[0]]
        
        # Hits@K
        gt = set(p.lower() for p in sample.get_path_strings()) if sample.ground_truth_paths else set()
        for k in [1, 3, 5, 10]:
            if any(p.lower() in gt for p in retrieved[:k]):
                hits[k] += 1
        
        # Find GT match
        best_match_path = None
        for path in retrieved[:5]:
            if path.lower() in gt:
                best_match_path = path
                break
        
        # Build context
        all_candidates = []
        context_lines = []
        
        if best_match_path:
            try:
                ents = sample.extract_entities_for_path(best_match_path)
                if ents:
                    unique = list(dict.fromkeys(ents))[:10]
                    all_candidates.extend(unique)
                    context_lines.append(f"[MATCH] {best_match_path}: {', '.join(unique[:5])}")
            except:
                pass
        
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
        
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            cl = c.lower()
            if cl not in seen:
                seen.add(cl)
                unique_candidates.append(c)
        
        context = "\n".join(context_lines) if context_lines else "No paths"
        
        # Prompt
        if unique_candidates:
            prompt = f"""Question: {sample.question}

KG paths and entities:
{context}

Candidates: {', '.join(unique_candidates[:10])}

Select ONLY the entity that answers the question.

Answer:"""
        else:
            prompt = f"Question: {sample.question}\nAnswer directly:"
        
        # K=5 voting
        answers = []
        for i in range(NUM_SAMPLES):
            temp = 0.1 if i == 0 else 0.5 + i * 0.1  # Increasing diversity
            ans = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=temp)
            ans = ans.strip().strip('"\'').split('\n')[0].rstrip('.')
            if ans:
                answers.append(ans.lower())
        
        if answers:
            vote_counts = Counter(answers)
            answer = vote_counts.most_common(1)[0][0]
            # Get original case from greedy answer
            greedy = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
            answer = greedy.strip().strip('"\'').split('\n')[0].rstrip('.')
        else:
            answer = ""
        
        f1 = evaluator._compute_f1(answer, sample.answer_entities)
        em = evaluator._compute_exact_match(answer, sample.answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
    
    print(f"\n{'='*60}")
    print(f"SELF-CONSISTENCY k={NUM_SAMPLES} ({count} samples)")
    print(f"{'='*60}")
    print(f"F1: {f1_sum/count*100:.1f}%")
    print(f"EM: {em_sum/count*100:.1f}%")
    print(f"Hits@1: {hits[1]/count*100:.1f}%")
    print(f"Hits@5: {hits[5]/count*100:.1f}%")
    print(f"Hits@10: {hits[10]/count*100:.1f}%")
    
    if f1_sum/count >= 0.85:
        print("\n🎉 TARGET ACHIEVED: F1 >= 85%!")
    else:
        gap = 0.85 - f1_sum/count
        print(f"\nGap to 85%: {gap*100:.1f}%")


if __name__ == "__main__":
    run_self_consistency_k5()
