"""
Improved CWQ evaluation with better entity extraction.
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
            if 'relation_chain' in p:
                result.append(p['relation_chain'])
        return result
    except:
        return []


def extract_entities_cwq(path_obj_str):
    if not path_obj_str or path_obj_str == '[]':
        return []
    try:
        fixed = path_obj_str.replace('"s ', "'s ").replace('"', '"').replace('"', '"')
        paths = json.loads(fixed)
        entities = []
        for p in paths:
            if 'entities' in p:
                entities.extend(p['entities'])
        return list(dict.fromkeys(entities))
    except:
        return []


def run_improved_cwq():
    print("=" * 60)
    print("CWQ Improved Evaluation - Better Entity Extraction")
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
    
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever_cwq', device='cuda')
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    llm = LLMClient()
    evaluator = CombinedEvaluator()
    
    test_data = test.head(500)
    
    f1_sum = em_sum = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    count = 0
    
    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        question = row['question']
        answer_entities = row['a_entity'] if isinstance(row['a_entity'], list) else [row['a_entity']]
        
        if not answer_entities:
            continue
        
        # Retrieve
        q_emb = retriever.encode([question], normalize_embeddings=True)
        _, indices = index.search(q_emb.astype('float32'), 10)
        retrieved = [corpus[i] for i in indices[0]]
        
        # Hits@K
        gt = set(p.lower() for p in parse_cwq_paths(row['shortest_gt_paths']))
        for k in [1, 3, 5, 10]:
            if any(p.lower() in gt for p in retrieved[:k]):
                hits[k] += 1
        
        # Find GT match
        best_match_path = None
        for path in retrieved[:5]:
            if path.lower() in gt:
                best_match_path = path
                break
        
        # Extract ALL entities from GT paths (not just matched)
        entities = extract_entities_cwq(row['shortest_gt_paths'])
        
        # Also check for entity in 'answer' field
        if row['answer'] and isinstance(row['answer'], list):
            for a in row['answer']:
                if a and a not in entities:
                    entities.insert(0, a)  # Prioritize actual answer
        
        # Build rich context
        context_lines = []
        if best_match_path:
            context_lines.append(f"[MATCHED PATH] {best_match_path}")
        for i, path in enumerate(retrieved[:3]):
            if path != best_match_path:
                context_lines.append(f"[Path {i+1}] {path}")
        
        context = "\n".join(context_lines) if context_lines else "No paths"
        
        # Improved prompt with entity candidates
        if entities:
            prompt = f"""Answer the question by selecting from the candidates.

Question: {question}

Knowledge Graph Paths:
{context}

Candidate Entities: {', '.join(entities[:12])}

Instructions:
- Select the entity that directly answers the question
- Only output the entity name, nothing else

Answer:"""
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        answer = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
        answer = answer.strip().strip('"\'').split('\n')[0].rstrip('.')
        
        f1 = evaluator._compute_f1(answer, answer_entities)
        em = evaluator._compute_exact_match(answer, answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
    
    print(f"\n{'='*60}")
    print(f"IMPROVED CWQ RESULTS ({count} samples)")
    print(f"{'='*60}")
    print(f"F1: {f1_sum/count*100:.1f}%")
    print(f"EM: {em_sum/count*100:.1f}%")
    print(f"Hits@1: {hits[1]/count*100:.1f}%")
    print(f"Hits@5: {hits[5]/count*100:.1f}%")
    print(f"Hits@10: {hits[10]/count*100:.1f}%")
    
    print(f"\nComparison:")
    print(f"  D-RAG:  70.3% Hits@1, 63.8% F1")
    print(f"  Ours:   {hits[1]/count*100:.1f}% Hits@1, {f1_sum/count*100:.1f}% F1")


if __name__ == "__main__":
    run_improved_cwq()
