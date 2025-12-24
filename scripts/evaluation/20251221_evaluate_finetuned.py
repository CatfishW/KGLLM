"""
Quick evaluation script combining fine-tuned retriever with baseline approach.
Uses full corpus reranking like the original experiment_union_retrieval.py
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.reranker import Qwen3Reranker
from RAG.combined_evaluator import CombinedEvaluator
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def run_evaluation():
    print("=" * 60)
    print("Fine-Tuned Retriever + Full Corpus Reranking + Entity Context")
    print("=" * 60)
    
    config = RAGConfig(dataset='webqsp')
    loader = KGDataLoader(config)
    llm = LLMClient()
    reranker = Qwen3Reranker(config)
    reranker.load()
    evaluator = CombinedEvaluator()

    # Full Corpus
    print("\nBuilding corpus...")
    corpus, _ = loader.build_path_corpus(include_test=True)
    print(f"  Corpus size: {len(corpus)}")
    
    # Load fine-tuned retriever
    print("\nLoading fine-tuned retriever...")
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever', device='cuda')
    
    # Build FAISS index
    print("Building FAISS index...")
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    print(f"  Indexed {index.ntotal} paths")
    
    test_data = loader.test_data[:200]
    
    f1_sum = 0
    hits_at_5 = 0
    hits_at_10 = 0
    count = 0
    
    print('\nRunning evaluation (200 samples)...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
            
        # 1. Dense Retrieval -> Top 20
        q_emb = retriever.encode([sample.question], normalize_embeddings=True)
        _, indices = index.search(q_emb.astype('float32'), 20)
        retrieved = [corpus[i] for i in indices[0]]
        
        # 2. Rerank on TOP 20 retrieved 
        _, _, reranked_top10 = reranker.rerank(sample.question, retrieved, top_k=10)
        
        # 3. Check Hits@K
        gt_norm = set(p.lower() for paths in sample.ground_truth_paths for p in ([" -> ".join(paths)] if isinstance(paths, list) else [paths]))
        reranked_norm = [p.lower() for p in reranked_top10]
        
        if any(p in gt_norm for p in reranked_norm[:5]):
            hits_at_5 += 1
        if any(p in gt_norm for p in reranked_norm[:10]):
            hits_at_10 += 1
        
        # 4. Build Context WITH ENTITIES (like baseline)
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
            {'role': 'system', 'content': 'You are an expert answer extractor. Review the candidate paths and their entities. Select the answer that best fits the question. Return the answer string exactly. Be concise - give only the entity name(s).'},
            {'role': 'user', 'content': f'Q: {sample.question}\n{context_str}\nAnswer:'}
        ]
        
        ans = llm.chat(ans_msg, max_tokens=64, temperature=0.1).strip().strip('"').strip("'")
        
        # Eval
        f1 = evaluator._compute_f1(ans, sample.answer_entities)
        f1_sum += f1
        count += 1
        
    print(f'\n{"="*60}')
    print(f'RESULTS (200 samples)')
    print(f'{"="*60}')
    print(f'F1: {100*f1_sum/count:.1f}%')
    print(f'Hits@5: {100*hits_at_5/count:.1f}%')
    print(f'Hits@10: {100*hits_at_10/count:.1f}%')

if __name__ == "__main__":
    run_evaluation()
