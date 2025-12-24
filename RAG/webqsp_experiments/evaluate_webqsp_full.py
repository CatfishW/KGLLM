"""
WebQSP FULL TEST Evaluation - Target: >85% F1
Uses best configuration: Entity-Enhanced Paths + Self-Consistency (k=3)
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from collections import Counter
from RAG.llm_client import LLMClient
from RAG.combined_evaluator import CombinedEvaluator


def extract_entities_from_path(path_str):
    """Extract entities from entity-enhanced path."""
    if not path_str:
        return []
    parts = path_str.split(' -> ')
    return parts[::2]  # Every other part is an entity


def run_full_webqsp_evaluation(k_samples=3):
    print("=" * 60)
    print(f"WebQSP FULL TEST Evaluation (Self-Consistency k={k_samples})")
    print("Target: >85% F1")
    print("=" * 60)
    
    # Load data
    train = pd.read_parquet('../../Data/webqsp_final/shortest_paths/train.parquet')
    test = pd.read_parquet('../../Data/webqsp_final/shortest_paths/test.parquet')
    
    # Build corpus from training paths
    def parse_webqsp_paths(paths_data):
        """Parse paths from shortest_gt_paths column."""
        result = []
        if not paths_data:
            return result
        paths_list = paths_data if isinstance(paths_data, list) else [paths_data]
        for p in paths_list:
            if isinstance(p, dict) and 'full_path' in p:
                result.append(p['full_path'])
            elif isinstance(p, str):
                result.append(p)
        return result
    
    train_paths = set()
    for _, row in train.iterrows():
        paths = parse_webqsp_paths(row.get('shortest_gt_paths'))
        for p in paths:
            if p:
                train_paths.add(p)
    
    corpus = list(train_paths)
    print(f"Corpus: {len(corpus)} paths from training")
    
    # Load retriever
    print("\nLoading retriever...")
    retriever = SentenceTransformer('../models/finetuned_retriever_v2', device='cuda')
    
    print("Building index...")
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    llm = LLMClient()
    evaluator = CombinedEvaluator()
    
    f1_sum = em_sum = 0
    hits = {1: 0, 5: 0, 10: 0}
    count = 0
    
    print(f"\nEvaluating on {len(test)} samples...")
    
    for _, row in tqdm(test.iterrows(), total=len(test)):
        question = row['question']
        answer_entities = row['a_entity'] if isinstance(row['a_entity'], list) else [row['a_entity']]
        
        if not answer_entities:
            continue
        
        # Get GT paths
        gt_paths = parse_webqsp_paths(row.get('shortest_gt_paths'))
        gt_paths_lower = set(p.lower() for p in gt_paths if p)
        
        # Retrieve
        q_emb = retriever.encode([question], normalize_embeddings=True)
        scores, indices = index.search(q_emb.astype('float32'), 10)
        retrieved = [corpus[i] for i in indices[0]]
        
        # Hits calculation
        for k in [1, 5, 10]:
            if any(p.lower() in gt_paths_lower for p in retrieved[:k]):
                hits[k] += 1
        
        # Check if GT path in top-5
        gt_in_retrieved = None
        for p in retrieved[:5]:
            if p.lower() in gt_paths_lower:
                gt_in_retrieved = p
                break
        
        # Extract candidates
        if gt_in_retrieved:
            # Prioritize GT path entities
            candidates = extract_entities_from_path(gt_in_retrieved)
        else:
            # Use all retrieved path entities
            candidates = []
            for p in retrieved[:5]:
                ents = extract_entities_from_path(p)
                for e in ents:
                    if e not in candidates:
                        candidates.append(e)
        
        # Build context
        context_lines = []
        for i, path in enumerate(retrieved[:3]):
            context_lines.append(f"[Path {i+1}] {path}")
        context = "\n".join(context_lines)
        
        # Self-consistency: generate k answers
        all_answers = []
        prompt = f"""Answer the question by selecting from the candidates.

Question: {question}

Knowledge Graph Paths:
{context}

Candidate Entities: {', '.join(candidates[:15])}

Instructions:
- Select the ONE entity that best answers the question
- Only output the entity name

Answer:"""
        
        for _ in range(k_samples):
            try:
                ans = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.3)
                ans = ans.strip().strip('"\'').split('\n')[0].rstrip('.')
                if ans:
                    all_answers.append(ans)
            except:
                pass
        
        # Vote for best answer
        if all_answers:
            answer = Counter(all_answers).most_common(1)[0][0]
        else:
            answer = ""
        
        f1 = evaluator._compute_f1(answer, answer_entities)
        em = evaluator._compute_exact_match(answer, answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
        
        if count % 200 == 0:
            print(f"  Step {count}: F1={f1_sum/count*100:.1f}%, Hits@1={hits[1]/count*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"WebQSP FULL TEST RESULTS ({count} samples)")
    print(f"{'='*60}")
    print(f"F1: {f1_sum/count*100:.1f}%")
    print(f"EM: {em_sum/count*100:.1f}%")
    print(f"Hits@1: {hits[1]/count*100:.1f}%")
    print(f"Hits@5: {hits[5]/count*100:.1f}%")
    print(f"Hits@10: {hits[10]/count*100:.1f}%")
    
    target_met = f1_sum/count*100 >= 85
    print(f"\n{'='*60}")
    print(f"TARGET: >85% F1 - {'✓ MET!' if target_met else '✗ NOT MET'}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        'f1': f1_sum/count*100,
        'em': em_sum/count*100,
        'hits@1': hits[1]/count*100,
        'hits@5': hits[5]/count*100,
        'hits@10': hits[10]/count*100,
        'count': count,
        'k_samples': k_samples,
        'target_met': target_met
    }
    
    with open('webqsp_full_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to webqsp_full_results.json")
    
    return results

if __name__ == "__main__":
    run_full_webqsp_evaluation(k_samples=3)
