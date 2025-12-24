"""
WebQSP Retrieval Evaluation - Entity Enhanced
Uses Entity-Enhanced Paths (E1 -> R -> E2) and CWQ Model.
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

def parse_webqsp_paths_enhanced(paths_data):
    """Parse paths into Entity-Enhanced format: E1 -> R -> E2."""
    result_chains = []
    
    if not paths_data:
        return []
        
    # Handle JSON string
    if isinstance(paths_data, str):
        try:
            paths_data = json.loads(paths_data)
        except json.JSONDecodeError:
            pass # Treat as single string path if not JSON
            
    paths_list = paths_data if isinstance(paths_data, list) else [paths_data]
    for p in paths_list:
        if isinstance(p, dict):
            # Construct E1 -> R -> E2
            parts = []
            ents = p.get('entities', [])
            rels = p.get('relations', [])
            
            # Interleave
            for i in range(len(rels)):
                if i < len(ents):
                    parts.append(ents[i])
                parts.append(rels[i])
            if len(ents) > len(rels):
                parts.append(ents[-1])
            
            chain = " -> ".join(parts)
            if chain:
                result_chains.append(chain)
        elif isinstance(p, str):
             result_chains.append(p)
             
    return result_chains


def run_retrieval_eval():
    print("=" * 60, flush=True)
    print(f"WebQSP Retrieval Evaluation (Entity Enhanced)", flush=True)
    print("Strategy: E1 -> R -> E2 format + CWQ Model", flush=True)
    print("Target: Hits@1 > 50%", flush=True)
    print("=" * 60, flush=True)
    
    # Load data
    train = pd.read_parquet('../../Data/webqsp_final/shortest_paths/train.parquet')
    test = pd.read_parquet('../../Data/webqsp_final/shortest_paths/test.parquet')
    
    # Build corpus
    train_chains = set()
    for _, row in train.iterrows():
        chains = parse_webqsp_paths_enhanced(row.get('shortest_gt_paths'))
        for c in chains:
            if c:
                train_chains.add(c)
    
    corpus = list(train_chains)
    print(f"Corpus: {len(corpus)} enhanced paths from training", flush=True)
    
    # Load new WebQSP enhanced retriever
    model_name = '../models/finetuned_retriever_webqsp_enhanced'
    print(f"\nLoading retriever: {model_name}...", flush=True)
    retriever = SentenceTransformer(model_name, device='cuda')
    
    print("Building index...", flush=True)
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    hits = {1: 0, 5: 0, 10: 0}
    count = 0
    
    print(f"\nEvaluating on {len(test)} samples...", flush=True)
    
    for _, row in tqdm(test.iterrows(), total=len(test)):
        question = row['question']
        
        gt_chains = parse_webqsp_paths_enhanced(row.get('shortest_gt_paths'))
        gt_chains_lower = set(c.lower() for c in gt_chains if c)
        
        if not gt_chains_lower:
            count += 1
            continue

        # Retrieve
        q_emb = retriever.encode([question], normalize_embeddings=True)
        scores, indices = index.search(q_emb.astype('float32'), 10)
        retrieved = [corpus[i] for i in indices[0]]
        
        # Hits calculation
        for k in [1, 5, 10]:
            if any(p.lower() in gt_chains_lower for p in retrieved[:k]):
                hits[k] += 1
        
        count += 1
        
        if count % 200 == 0:
            print(f"  Step {count}: Hits@1={hits[1]/count*100:.1f}%, Hits@5={hits[5]/count*100:.1f}%", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"WebQSP RETRIEVAL RESULTS (Enhanced)")
    print(f"{'='*60}")
    print(f"Hits@1: {hits[1]/count*100:.1f}%")
    print(f"Hits@5: {hits[5]/count*100:.1f}%")
    print(f"Hits@10: {hits[10]/count*100:.1f}%")
    
    # Save results
    results = {
        'hits@1': hits[1]/count*100,
        'hits@5': hits[5]/count*100,
        'hits@10': hits[10]/count*100,
        'count': count
    }
    
    with open('webqsp_retrieval_enhanced.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to webqsp_retrieval_enhanced.json", flush=True)
    
    return results

if __name__ == "__main__":
    run_retrieval_eval()
