"""
Diagnose CWQ retrieval bottleneck.
Analyze failure cases where Hits@1 = 0.
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


def parse_cwq_paths(shortest_gt_paths_str):
    if not shortest_gt_paths_str or shortest_gt_paths_str == '[]':
        return []
    try:
        fixed = shortest_gt_paths_str.replace('"s ', "'s ").replace('"', '"').replace('"', '"')
        paths = json.loads(fixed)
        result = []
        for p in paths:
            if 'entities' in p and 'relations' in p:
                entities = p['entities']
                relations = p['relations']
                
                parts = []
                for i in range(len(relations)):
                    if i < len(entities):
                        parts.append(entities[i])
                    parts.append(relations[i])
                if len(entities) > len(relations):
                    parts.append(entities[-1])
                
                full_path = " -> ".join(parts)
                result.append(full_path)
        return result
    except:
        return []


def diagnose():
    print("=" * 60)
    print("CWQ Retrieval Bottleneck Diagnosis")
    print("=" * 60)
    
    train = pd.read_parquet('Data/CWQ/shortest_paths/train.parquet')
    test = pd.read_parquet('Data/CWQ/shortest_paths/test.parquet')
    
    # Build corpus
    all_paths = set()
    for _, row in train.iterrows():
        for p in parse_cwq_paths(row['shortest_gt_paths']):
            if p: all_paths.add(p)
    for _, row in test.iterrows():
        for p in parse_cwq_paths(row['shortest_gt_paths']):
            if p: all_paths.add(p)
    
    corpus = list(all_paths)
    print(f"Corpus: {len(corpus)} paths")
    
    print("\nLoading retriever...")
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever_cwq_v2_fix', device='cuda')
    
    print("Building index...")
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    # Analyze first 100 samples
    test_sample = test.head(100)
    
    miss_at_1 = 0
    miss_at_10 = 0
    total = 0
    
    failure_cases = []
    
    for _, row in tqdm(test_sample.iterrows(), total=len(test_sample)):
        question = row['question']
        gt_paths = set(p.lower() for p in parse_cwq_paths(row['shortest_gt_paths']))
        
        if not gt_paths:
            continue
        
        # Retrieve
        q_emb = retriever.encode([question], normalize_embeddings=True)
        scores, indices = index.search(q_emb.astype('float32'), 10)
        retrieved = [corpus[i] for i in indices[0]]
        
        # Check hits
        hit_at_1 = any(p.lower() in gt_paths for p in retrieved[:1])
        hit_at_10 = any(p.lower() in gt_paths for p in retrieved[:10])
        
        if not hit_at_1:
            miss_at_1 += 1
            failure_cases.append({
                'question': question,
                'gt_paths': list(gt_paths)[:2],
                'retrieved_top3': retrieved[:3],
                'scores': scores[0][:3].tolist()
            })
        
        if not hit_at_10:
            miss_at_10 += 1
        
        total += 1
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS RESULTS ({total} samples)")
    print(f"{'='*60}")
    print(f"Miss@1: {miss_at_1}/{total} ({miss_at_1/total*100:.1f}%)")
    print(f"Miss@10: {miss_at_10}/{total} ({miss_at_10/total*100:.1f}%)")
    print(f"Hits@1: {(total-miss_at_1)/total*100:.1f}%")
    print(f"Hits@10: {(total-miss_at_10)/total*100:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"FAILURE CASES (Miss@1)")
    print(f"{'='*60}")
    
    for i, case in enumerate(failure_cases[:5]):
        print(f"\n**Case {i+1}:**")
        print(f"Q: {case['question']}")
        print(f"\nGT Paths:")
        for gt in case['gt_paths']:
            print(f"  - {gt[:150]}...")
        print(f"\nRetrieved (scores):")
        for r, s in zip(case['retrieved_top3'], case['scores']):
            print(f"  [{s:.3f}] {r[:150]}...")
    
    # Save diagnosis
    with open('cwq_diagnosis.json', 'w') as f:
        json.dump({
            'miss_at_1': miss_at_1,
            'miss_at_10': miss_at_10,
            'total': total,
            'hits@1': (total-miss_at_1)/total*100,
            'hits@10': (total-miss_at_10)/total*100,
            'failure_cases': failure_cases[:20]
        }, f, indent=2)
    
    print(f"\n\nDiagnosis saved to cwq_diagnosis.json")


if __name__ == "__main__":
    diagnose()
