"""
CWQ FULL Evaluation (All 3531 test samples).
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from RAG.llm_client import LLMClient
from RAG.combined_evaluator import CombinedEvaluator


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

def extract_entities_from_path(path_str):
    if not path_str:
        return []
    parts = path_str.split(' -> ')
    return parts[::2]

def run_full_evaluation():
    print("=" * 60)
    print("CWQ FULL TEST Evaluation (3531 samples)")
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
    
    print("Loading retriever...")
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever_cwq_v2_fix', device='cuda')
    
    print("Building index...")
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    llm = LLMClient()
    evaluator = CombinedEvaluator()
    
    f1_sum = em_sum = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    recall = {10: 0}
    count = 0
    
    print(f"\nEvaluating on {len(test)} samples...")
    
    for _, row in tqdm(test.iterrows(), total=len(test)):
        question = row['question']
        answer_entities = row['a_entity'] if isinstance(row['a_entity'], list) else [row['a_entity']]
        
        if not answer_entities:
            continue
        
        # Retrieve
        q_emb = retriever.encode([question], normalize_embeddings=True)
        _, indices = index.search(q_emb.astype('float32'), 10)
        retrieved = [corpus[i] for i in indices[0]]
        
        # Hits
        gt_paths = set(p.lower() for p in parse_cwq_paths(row['shortest_gt_paths']))
        for k in [1, 3, 5, 10]:
            if any(p.lower() in gt_paths for p in retrieved[:k]):
                hits[k] += 1
        
        # Extract candidates from retrieved paths
        retrieved_entities = []
        for p in retrieved:
            ents = extract_entities_from_path(p)
            for e in ents:
                if e not in retrieved_entities:
                    retrieved_entities.append(e)
        
        # Recall
        norm_answers = {evaluator._normalize_answer(a) for a in answer_entities}
        norm_candidates = {evaluator._normalize_answer(c) for c in retrieved_entities}
        if not norm_candidates.isdisjoint(norm_answers):
            recall[10] += 1
        
        # Context
        context_lines = []
        for i, path in enumerate(retrieved[:5]):
            context_lines.append(f"[Path {i+1}] {path}")
        context = "\n".join(context_lines)
        
        candidates = retrieved_entities[:20]
        
        if candidates:
            prompt = f"""Answer the question by selecting from the candidates.

Question: {question}

Knowledge Graph Paths:
{context}

Candidate Entities: {', '.join(candidates)}

Instructions:
- Select the ONE entity that best answers the question
- Only output the entity name

Answer:"""
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        try:
            answer = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
            answer = answer.strip().strip('"\'').split('\n')[0].rstrip('.')
        except:
            answer = ""
        
        f1 = evaluator._compute_f1(answer, answer_entities)
        em = evaluator._compute_exact_match(answer, answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
        
        if count % 500 == 0:
            print(f"  Step {count}: F1={f1_sum/count*100:.1f}%, Hits@1={hits[1]/count*100:.1f}%, Recall={recall[10]/count*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"CWQ FULL TEST RESULTS ({count} samples)")
    print(f"{'='*60}")
    print(f"F1: {f1_sum/count*100:.1f}%")
    print(f"EM: {em_sum/count*100:.1f}%")
    print(f"Hits@1: {hits[1]/count*100:.1f}%")
    print(f"Hits@5: {hits[5]/count*100:.1f}%")
    print(f"Hits@10: {hits[10]/count*100:.1f}%")
    print(f"Candidate Recall: {recall[10]/count*100:.1f}%")
    
    print(f"\nComparison with D-RAG:")
    print(f"  D-RAG:  70.3% Hits@1, 63.8% F1")
    print(f"  Ours:   {hits[1]/count*100:.1f}% Hits@1, {f1_sum/count*100:.1f}% F1")
    
    # Save results
    with open('cwq_full_results.json', 'w') as f:
        json.dump({
            'f1': f1_sum/count*100,
            'em': em_sum/count*100,
            'hits@1': hits[1]/count*100,
            'hits@5': hits[5]/count*100,
            'hits@10': hits[10]/count*100,
            'recall@10': recall[10]/count*100,
            'count': count
        }, f, indent=2)
    print(f"\nResults saved to cwq_full_results.json")

if __name__ == "__main__":
    run_full_evaluation()
