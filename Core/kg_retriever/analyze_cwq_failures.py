
import json
import os
from collections import Counter

def analyze_failures(results_path):
    print(f"Loading results from {results_path}...")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    total = len(results)
    failures = [r for r in results if not r.get('llm_correct')]
    
    print(f"Total samples: {total}")
    print(f"Total failures: {len(failures)} ({len(failures)/total*100:.2f}%)")
    
    categories = Counter()
    
    # Store examples for each category
    examples = {
        'retrieval_failure': [],
        'context_missing': [],
        'llm_reasoning_failure': []
    }
    
    for r in failures:
        # Check retrieval rank (1-indexed)
        # top_k for LLM was 10 in the run command (implied by previous context)
        rank = r.get('rank', 999)
        top_k_paths = r.get('top_k_paths_with_entities', [])
        
        # 1. Retrieval Failure: Ground truth path was not in the top K paths seen by LLM
        if rank > 10:
            categories['retrieval_failure'] += 1
            if len(examples['retrieval_failure']) < 3:
                examples['retrieval_failure'].append(r)
            continue
            
        # 2. Context Missing: Path retrieved, but the specific answer entity wasn't in the display text
        # This checks if ANY of the ground truth answers appear in the context provided to LLM
        gt_answers = r.get('gt_answers', [])
        context_text = " ".join(top_k_paths).lower()
        
        found_in_context = False
        for answer in gt_answers:
            if answer.lower() in context_text:
                found_in_context = True
                break
        
        if not found_in_context:
            categories['context_missing'] += 1
            if len(examples['context_missing']) < 5:
                examples['context_missing'].append(r)
        else:
            # 3. LLM Reasoning Failure: Rank <= 10 AND Answer in Context, but LLM got it wrong
            categories['llm_reasoning_failure'] += 1
            if len(examples['llm_reasoning_failure']) < 5:
                examples['llm_reasoning_failure'].append(r)

    print("\nFailure Categories:")
    print(f"1. Retrieval Failure (Rank > 10): {categories['retrieval_failure']} ({categories['retrieval_failure']/len(failures)*100:.2f}%)")
    print(f"   - Cause: The correct path wasn't in the top 10 passed to the LLM.")
    print(f"2. Context Missing (Rank <= 10 but Answer not visible): {categories['context_missing']} ({categories['context_missing']/len(failures)*100:.2f}%)")
    print(f"   - Cause: Correct path found, but the specific answer entity wasn't in the sampled entity list (limit 10).")
    print(f"3. LLM Reasoning Failure (Rank <= 10 and Answer visible): {categories['llm_reasoning_failure']} ({categories['llm_reasoning_failure']/len(failures)*100:.2f}%)")
    print(f"   - Cause: LLM saw the correct answer but chose something else.")

    print("\n" + "="*80)
    print("ANALYSIS OF LLM REASONING FAILURES")
    print("="*80)
    for i, ex in enumerate(examples['llm_reasoning_failure'], 1):
        print(f"\nExample {i}:")
        print(f"Question: {ex['question']}")
        print(f"Topic: {ex['topic_entities']}")
        print(f"GT Answers: {ex['gt_answers']}")
        print(f"LLM Prediction: {ex.get('llm_answer')}")
        print(f"True Rank: {ex['rank']}")
        # Find the path that contained the answer
        for j, path_str in enumerate(ex['top_k_paths_with_entities']):
            for ans in ex['gt_answers']:
                if ans.lower() in path_str.lower():
                    print(f"Path containing answer (Index {j+1}):")
                    print(f"  {path_str}")
                    break

    print("\n" + "="*80)
    print("ANALYSIS OF CONTEXT MISSING FAILURES")
    print("="*80)
    for i, ex in enumerate(examples['context_missing'], 1):
        print(f"\nExample {i}:")
        print(f"Question: {ex['question']}")
        print(f"GT Answers: {ex['gt_answers']}")
        print(f"True Rank: {ex['rank']}")
        print(f"Top 1 Path (Truncated?):")
        # Show where the answer SHOULD have been
        if ex['rank'] <= len(ex['top_k_paths_with_entities']):
             print(f"  {ex['top_k_paths_with_entities'][ex['rank']-1]}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/data/Yanlai/KGLLM/Core/kg_retriever/outputs_eval_improved_v2/cwq/results.json"
    analyze_failures(path)
