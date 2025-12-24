"""
CWQ Benchmark Evaluation - matching paper format.
Evaluates on CWQ dataset with Hits@1 and F1 metrics.
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
import json
from pathlib import Path


def run_cwq_evaluation():
    print("=" * 60)
    print("CWQ Benchmark Evaluation")
    print("=" * 60)
    
    # Use CWQ config
    config = RAGConfig(dataset='cwq')
    loader = KGDataLoader(config)
    llm = LLMClient()
    evaluator = CombinedEvaluator()

    # Build CWQ corpus
    print("\nBuilding CWQ corpus...")
    corpus, _ = loader.build_path_corpus(include_test=True)
    print(f"  {len(corpus)} paths")
    
    # Load same fine-tuned retriever (transfer learning)
    print("\nLoading fine-tuned retriever...")
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever', device='cuda')
    
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    print(f"  Indexed {index.ntotal} paths")
    
    test_data = loader.test_data[:500]  # First 500 samples
    
    f1_sum = em_sum = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    count = 0
    
    print(f"\nEvaluating on {len(test_data)} CWQ samples...")
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
        
        # Retrieve (no reranker)
        q_emb = retriever.encode([sample.question], normalize_embeddings=True)
        _, indices = index.search(q_emb.astype('float32'), 10)
        retrieved = [corpus[i] for i in indices[0]]
        
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
            prompt = f"""Question: {sample.question}

KG paths and entities:
{context}

Candidates: {', '.join(unique_candidates[:10])}

Select ONLY the entity that answers the question.

Answer:"""
        else:
            prompt = f"Question: {sample.question}\nAnswer:"
        
        answer = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
        answer = answer.strip().strip('"\'').split('\n')[0].rstrip('.')
        
        f1 = evaluator._compute_f1(answer, sample.answer_entities)
        em = evaluator._compute_exact_match(answer, sample.answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
    
    print(f"\n{'='*60}")
    print(f"CWQ RESULTS ({count} samples)")
    print(f"{'='*60}")
    print(f"F1: {f1_sum/count*100:.1f}%")
    print(f"EM: {em_sum/count*100:.1f}%")
    print(f"Hits@1: {hits[1]/count*100:.1f}%")
    print(f"Hits@5: {hits[5]/count*100:.1f}%")
    print(f"Hits@10: {hits[10]/count*100:.1f}%")
    
    # Save results
    Path('./RAG/best_config_76pct_f1/cwq_results').mkdir(parents=True, exist_ok=True)
    with open('./RAG/best_config_76pct_f1/cwq_results/metrics.json', 'w') as f:
        json.dump({
            'f1': f1_sum/count,
            'em': em_sum/count,
            'hits@1': hits[1]/count,
            'hits@10': hits[10]/count,
            'num_samples': count
        }, f, indent=2)


if __name__ == "__main__":
    run_cwq_evaluation()
