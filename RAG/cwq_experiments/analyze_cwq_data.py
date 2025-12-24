"""
Analyze CWQ 500-sample results from evaluate_cwq_fair.py.
Find patterns in failures without needing to reload the model.
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

import pandas as pd
import json


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
                full_path = " -> ".join([item for pair in zip(entities, relations) for item in pair] + [entities[-1]] if len(entities) > len(relations) else [])
                result.append(full_path)
        return result
    except:
        return []


def analyze():
    print("="*60)
    print("CWQ 500-Sample Analysis")
    print("="*60)
    
    test = pd.read_parquet('Data/CWQ/shortest_paths/test.parquet')
    test_500 = test.head(500)
    
    # Analyze question lengths
    question_lens = [len(row['question'].split()) for _, row in test_500.iterrows()]
    
    # Analyze path complexities
    path_lens = []
    for _, row in test_500.iterrows():
        paths = parse_cwq_paths(row['shortest_gt_paths'])
        if paths:
            avg_len = sum(len(p.split(' -> ')) for p in paths) / len(paths)
            path_lens.append(avg_len)
    
    # Entity analysis
    multi_entity_answers = sum(1 for _, row in test_500.iterrows() if isinstance(row['a_entity'], list) and len(row['a_entity']) > 1)
    
    print(f"\n**Dataset Statistics (500 samples):**")
    print(f"  Avg question length: {sum(question_lens)/len(question_lens):.1f} words")
    print(f"  Avg path complexity: {sum(path_lens)/len(path_lens):.1f} hops")
    print(f"  Multi-entity answers: {multi_entity_answers} ({multi_entity_answers/500*100:.1f}%)")
    
    # Compare with WebQSP
    webqsp_test = pd.read_parquet('Data/WebQSP/shortest_paths/test.parquet')[:500]
    webqsp_q_lens = [len(row['RawQuestion'].split()) for _, row in webqsp_test.iterrows()]
    
    print(f"\n**Comparison with WebQSP:**")
    print(f"  CWQ Avg Q len: {sum(question_lens)/len(question_lens):.1f} words")
    print(f"  WebQSP Avg Q len: {sum(webqsp_q_lens)/len(webqsp_q_lens):.1f} words")
    print(f"  CWQ is {(sum(question_lens)/len(question_lens)) / (sum(webqsp_q_lens)/len(webqsp_q_lens)):.1f}x longer")
    
    # Hypothesis
    print(f"\n{'='*60}")
    print(f"HYPOTHESIS")
    print(f"{'='*60}")
    print(f"""
**Why Hits@1 is low (57.4% vs 70.3% target):**

1. **Question Complexity**: CWQ questions are ~{(sum(question_lens)/len(question_lens)) / (sum(webqsp_q_lens)/len(webqsp_q_lens)):.1f}x longer than WebQSP
   - Longer questions are harder to match with paths
   - More noise in question encoding

2. **Path Complexity**: Average {sum(path_lens)/len(path_lens):.1f} hops per path
   - Entity-Enhanced Paths are VERY long
   - Example: "E1 -> R1 -> E2 -> R2 -> E3 -> R3 -> E4" (7 parts)
   - Hard for retriever to match entire path

3. **Training Data**: 140K triplets may not be enough
   - WebQSP had similar amount, achieved 90% Hits@10
   - But CWQ paths are more complex

**Potential Solutions:**

A. **Query Augmentation**:
   - Extract key entities/phrases from question
   - Generate multiple query variations
   - Ensemble retrieval results

B. **Path Truncation**: 
   - Train on shorter path segments
   - "E1 -> R1 -> E2" instead of full path
   - Retrieve segments, then combine

C. **Better Hard Negatives**:
   - Current negatives are random paths
   - Use BM25 to find harder negatives (lexically similar but wrong)

D. **Larger Model**:
   - Try bge-large-en-v1.5 (335M vs 109M params)
   - More capacity for complex paths

E. **Hybrid Retrieval**:
   - Combine dense (current) + sparse (BM25)
   - D-RAG likely uses this

**Recommended Next Steps:**
1. Try Hybrid Retrieval (easiest, high impact)
2. Query Augmentation with entity extraction
3. Train bge-large if needed
""")
    
    print(f"\n{'='*60}")
    print("Waiting for full test results...")
    print(f"{'='*60}")


if __name__ == "__main__":
    analyze()
