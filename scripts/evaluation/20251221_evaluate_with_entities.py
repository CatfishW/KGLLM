"""
Final evaluation with proper entity extraction.
Uses KG graph traversal to get candidate entities for each path.
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

def run_evaluation():
    print("=" * 60)
    print("Full Pipeline with Entity Extraction from KG")
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
    entity_found = 0
    count = 0
    
    print(f'\nProcessing {len(test_data)} samples...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # 1. Retrieve top 30
        q_emb = retriever.encode([sample.question], normalize_embeddings=True)
        _, indices = index.search(q_emb.astype('float32'), 30)
        retrieved = [corpus[i] for i in indices[0]]
        
        # 2. Rerank with Qwen3
        _, _, reranked = reranker.rerank(sample.question, retrieved, top_k=10)
        
        # 3. Hits@K
        gt = set(p.lower() for p in sample.get_path_strings()) if sample.ground_truth_paths else set()
        reranked_lower = [p.lower() for p in reranked]
        
        if any(p in gt for p in reranked_lower[:1]): hits_at_1 += 1
        if any(p in gt for p in reranked_lower[:5]): hits_at_5 += 1
        if any(p in gt for p in reranked_lower[:10]): hits_at_10 += 1
        
        # 4. Extract entities from KG for each path
        all_candidates = []
        context_lines = []
        
        for path in reranked[:5]:
            try:
                ents = sample.extract_entities_for_path(path)
                if ents:
                    entity_found += 1
                    # Deduplicate while preserving order
                    seen = set()
                    unique_ents = []
                    for e in ents:
                        if e.lower() not in seen:
                            seen.add(e.lower())
                            unique_ents.append(e)
                    all_candidates.extend(unique_ents[:5])
                    context_lines.append(f"• {path}\n  Entities: {', '.join(unique_ents[:5])}")
            except:
                pass
        
        # Deduplicate all candidates
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique_candidates.append(c)
        
        # 5. Answer extraction with entity list
        if unique_candidates:
            candidate_str = ", ".join(unique_candidates[:15])
            context = "\n".join(context_lines) if context_lines else "No paths"
            
            prompt = f"""Question: {sample.question}

Candidate Entities from Knowledge Graph:
{candidate_str}

Path Context:
{context}

Select the entity that answers the question. Return ONLY the entity name(s).

Answer:"""
        else:
            # Fallback: just paths without entities
            context = "\n".join(reranked[:5])
            prompt = f"""Question: {sample.question}

Relevant paths: {context}

What is the answer? Give only the entity name(s).

Answer:"""
        
        ans = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
        ans = ans.strip().strip('"\'').split('\n')[0].rstrip('.')
        
        f1 = evaluator._compute_f1(ans, sample.answer_entities)
        em = evaluator._compute_exact_match(ans, sample.answer_entities)
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
    print(f'Entity Extraction Success: {entity_found} paths with entities')

if __name__ == "__main__":
    run_evaluation()
