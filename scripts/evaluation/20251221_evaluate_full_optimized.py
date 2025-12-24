"""
Evaluate full pipeline with both fine-tuned retriever and reranker.
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

def run_full_evaluation():
    print("=" * 60)
    print("Full Optimized Pipeline Evaluation")
    print("Fine-Tuned Retriever + Fine-Tuned Cross-Encoder Reranker")
    print("=" * 60)
    
    config = RAGConfig(dataset='webqsp')
    loader = KGDataLoader(config)
    llm = LLMClient()
    evaluator = CombinedEvaluator()

    # Full Corpus
    print("\nBuilding corpus...")
    corpus, _ = loader.build_path_corpus(include_test=True)
    print(f"  Corpus size: {len(corpus)}")
    
    # Load fine-tuned retriever
    print("\nLoading fine-tuned retriever...")
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever', device='cuda')
    
    # Load fine-tuned cross-encoder reranker
    print("Loading fine-tuned cross-encoder reranker...")
    reranker = CrossEncoder('./RAG/models/finetuned_reranker_ce', device='cuda')
    
    # Build FAISS index
    print("Building FAISS index...")
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    print(f"  Indexed {index.ntotal} paths")
    
    test_data = loader.test_data[:500]  # Larger eval
    
    f1_sum = 0
    em_sum = 0
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    count = 0
    
    print(f'\nRunning evaluation ({len(test_data)} samples)...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # 1. Dense Retrieval -> Top 30
        q_emb = retriever.encode([sample.question], normalize_embeddings=True)
        _, indices = index.search(q_emb.astype('float32'), 30)
        retrieved = [corpus[i] for i in indices[0]]
        
        # 2. Cross-Encoder Rerank on TOP 30
        pairs = [[sample.question, path] for path in retrieved]
        scores = reranker.predict(pairs)
        
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        reranked_top10 = [retrieved[i] for i in sorted_indices[:10]]
        
        # 3. Check Hits@K
        gt_paths = sample.get_path_strings() if sample.ground_truth_paths else []
        gt_norm = set(p.lower() for p in gt_paths)
        reranked_norm = [p.lower() for p in reranked_top10]
        
        if any(p in gt_norm for p in reranked_norm[:1]):
            hits_at_1 += 1
        if any(p in gt_norm for p in reranked_norm[:5]):
            hits_at_5 += 1
        if any(p in gt_norm for p in reranked_norm[:10]):
            hits_at_10 += 1
        
        # 4. Build Context WITH ENTITIES
        context_lines = []
        for p in reranked_top10[:5]:
            try:
                ents = sample.extract_entities_for_path(p) 
            except:
                ents = []
            
            ent_str = ", ".join(ents[:5])
            if ent_str:
                context_lines.append(f"Path: {p} -> Candidates: [{ent_str}]")
            else:
                context_lines.append(f"Path: {p}")
        
        if not context_lines:
            context_str = "No candidates found."
        else:
            context_str = "\n".join(context_lines)
        
        # 5. Answer Extraction  
        ans_msg = [
            {'role': 'system', 'content': 'You are an expert answer extractor. Review the paths and their candidate entities. Select the answer that best fits the question. Return ONLY the entity name(s), nothing else. Be concise.'},
            {'role': 'user', 'content': f'Question: {sample.question}\n\n{context_str}\n\nAnswer:'}
        ]
        
        ans = llm.chat(ans_msg, max_tokens=64, temperature=0.1).strip().strip('"').strip("'").rstrip('.')
        
        # Eval
        f1 = evaluator._compute_f1(ans, sample.answer_entities)
        em = evaluator._compute_exact_match(ans, sample.answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
        
    print(f'\n{"="*60}')
    print(f'RESULTS ({count} samples)')
    print(f'{"="*60}')
    print(f'F1: {100*f1_sum/count:.1f}%')
    print(f'Exact Match: {100*em_sum/count:.1f}%')
    print(f'Hits@1: {100*hits_at_1/count:.1f}%')
    print(f'Hits@5: {100*hits_at_5/count:.1f}%')
    print(f'Hits@10: {100*hits_at_10/count:.1f}%')

if __name__ == "__main__":
    run_full_evaluation()
