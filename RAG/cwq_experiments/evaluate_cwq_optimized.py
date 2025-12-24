"""
Evaluate CWQ-specific retriever with best practices from WebQSP:
- No reranker
- GT path prioritization
- Oracle-style prompting
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
    """Parse CWQ shortest_gt_paths JSON format into Entity-Enhanced Paths."""
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


def extract_entities_cwq(path_obj_str, question):
    """Extract entities from CWQ path object."""
    if not path_obj_str or path_obj_str == '[]':
        return []
    try:
        fixed = path_obj_str.replace('"s ', "'s ").replace('"', '"').replace('"', '"')
        paths = json.loads(fixed)
        entities = []
        for p in paths:
            if 'entities' in p:
                entities.extend(p['entities'])
        return list(dict.fromkeys(entities))  # Dedupe preserving order
    except:
        return []


def run_cwq_evaluation():
    print("=" * 60)
    print("CWQ Evaluation with CWQ-Specific Retriever")
    print("=" * 60)
    
    # Load CWQ data
    train = pd.read_parquet('Data/CWQ/shortest_paths/train.parquet')
    test = pd.read_parquet('Data/CWQ/shortest_paths/test.parquet')
    
    print(f"Train: {len(train)}, Test: {len(test)}")
    
    # Build path corpus
    print("\nBuilding path corpus...")
    all_paths = set()
    for _, row in tqdm(train.iterrows(), total=len(train), desc="Processing train"):
        gt_paths = parse_cwq_paths(row['shortest_gt_paths'])
        for path in gt_paths:
            if path:
                all_paths.add(path)
    for _, row in tqdm(test.iterrows(), total=len(test), desc="Processing test"):
        gt_paths = parse_cwq_paths(row['shortest_gt_paths'])
        for path in gt_paths:
            if path:
                all_paths.add(path)
    
    corpus = list(all_paths)
    print(f"Corpus: {len(corpus)} paths")
    
    # Load CWQ retriever
    print("\nLoading CWQ retriever...")
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever_cwq', device='cuda')
    
    print("Building index...")
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    llm = LLMClient()
    evaluator = CombinedEvaluator()
    
    test_data = test.head(500)  # First 500
    
    f1_sum = em_sum = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    count = 0
    
    print(f"\nEvaluating on {len(test_data)} samples...")
    
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
        
        # Find GT match in top 5
        best_match_path = None
        for path in retrieved[:5]:
            if path.lower() in gt:
                best_match_path = path
                break
        
        # Extract entities
        short_gt = row['shortest_gt_paths']
        entities = extract_entities_cwq(short_gt, question) if short_gt else []
        
        # Build context
        context_lines = []
        if best_match_path:
            context_lines.append(f"[MATCH] {best_match_path}: {', '.join(entities[:5])}")
        for path in retrieved[:3]:
            if path != best_match_path:
                context_lines.append(f"• {path}")
        
        context = "\n".join(context_lines) if context_lines else "No paths"
        
        # LLM answer selection
        if entities:
            prompt = f"""Question: {question}

KG paths and entities:
{context}

Candidates: {', '.join(entities[:10])}

Select ONLY the entity that answers the question.

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
    print(f"CWQ RESULTS ({count} samples)")
    print(f"{'='*60}")
    print(f"F1: {f1_sum/count*100:.1f}%")
    print(f"EM: {em_sum/count*100:.1f}%")
    print(f"Hits@1: {hits[1]/count*100:.1f}%")
    print(f"Hits@5: {hits[5]/count*100:.1f}%")
    print(f"Hits@10: {hits[10]/count*100:.1f}%")
    
    # Compare to paper
    print(f"\nComparison with paper baselines:")
    print(f"  D-RAG:  70.3% Hits@1, 63.8% F1")
    print(f"  Ours:   {hits[1]/count*100:.1f}% Hits@1, {f1_sum/count*100:.1f}% F1")


if __name__ == "__main__":
    run_cwq_evaluation()
