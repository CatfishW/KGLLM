"""
Final evaluation with fine-tuned retriever + cross-encoder reranker + improved prompting.
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.combined_evaluator import CombinedEvaluator
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np

def run_evaluation():
    print("=" * 60)
    print("Optimized Pipeline: Fine-Tuned Retriever + Cross-Encoder")
    print("=" * 60)
    
    config = RAGConfig(dataset='webqsp')
    loader = KGDataLoader(config)
    llm = LLMClient()
    evaluator = CombinedEvaluator()

    corpus, _ = loader.build_path_corpus(include_test=True)
    print(f"Corpus: {len(corpus)} paths")
    
    print("\nLoading models...")
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever', device='cuda')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
    
    print("Building FAISS index...")
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
            
        # 1. Retrieve top 30
        q_emb = retriever.encode([sample.question], normalize_embeddings=True)
        _, indices = index.search(q_emb.astype('float32'), 30)
        retrieved = [corpus[i] for i in indices[0]]
        
        # 2. Cross-Encoder Rerank
        pairs = [[sample.question, path] for path in retrieved]
        scores = reranker.predict(pairs)
        sorted_idx = np.argsort(scores)[::-1]
        reranked = [retrieved[i] for i in sorted_idx[:10]]
        
        # 3. Hits@K
        gt = set(p.lower() for p in sample.get_path_strings()) if sample.ground_truth_paths else set()
        reranked_lower = [p.lower() for p in reranked]
        
        if any(p in gt for p in reranked_lower[:1]): hits_at_1 += 1
        if any(p in gt for p in reranked_lower[:5]): hits_at_5 += 1
        if any(p in gt for p in reranked_lower[:10]): hits_at_10 += 1
        
        # 4. Entity context (simplified)
        context_parts = []
        for p in reranked[:5]:
            try:
                ents = sample.extract_entities_for_path(p)[:5]
            except:
                ents = []
            if ents:
                context_parts.append(f"{p} → ENTITIES: {', '.join(ents)}")
            else:
                context_parts.append(p)
        context = "\n".join(context_parts) if context_parts else "No paths found"
        
        # 5. Improved Answer Extraction Prompt
        prompt = f"""Task: Extract the answer entity from Knowledge Graph paths.

Question: {sample.question}

Relevant KG Paths with Candidate Entities:
{context}

Instructions:
- Pick the entity that directly answers the question
- If multiple answers, list them separated by commas
- Give ONLY the entity name(s), no explanation

Answer:"""

        ans = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
        ans = ans.strip().strip('"\'').split('\n')[0].rstrip('.')
        
        f1 = evaluator._compute_f1(ans, sample.answer_entities)
        em = evaluator._compute_exact_match(ans, sample.answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
        
    print(f'\n{"="*60}')  
    print(f'FINAL RESULTS ({count} samples)')
    print(f'{"="*60}')
    print(f'F1 Score: {100*f1_sum/count:.1f}%')
    print(f'Exact Match: {100*em_sum/count:.1f}%')
    print(f'Hits@1: {100*hits_at_1/count:.1f}%')
    print(f'Hits@5: {100*hits_at_5/count:.1f}%')
    print(f'Hits@10: {100*hits_at_10/count:.1f}%')

if __name__ == "__main__":
    run_evaluation()
