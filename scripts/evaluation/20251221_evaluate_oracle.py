"""
Oracle Test: Use ground truth paths for entity extraction.
This tests the upper bound of our approach.
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.combined_evaluator import CombinedEvaluator

def run_oracle_evaluation():
    print("=" * 60)
    print("ORACLE TEST - Using Ground Truth Paths")
    print("=" * 60)
    
    config = RAGConfig(dataset='webqsp')
    loader = KGDataLoader(config)
    llm = LLMClient()
    evaluator = CombinedEvaluator()

    test_data = loader.test_data[:500]
    
    f1_sum = em_sum = 0
    entity_found = 0
    entity_match = 0
    count = 0
    
    print(f'\nProcessing {len(test_data)} samples with GT paths...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities or not sample.ground_truth_paths:
            continue
            
        # Use ground truth paths
        gt_paths = sample.get_path_strings()
        
        # Extract entities using GT paths
        all_candidates = []
        context_lines = []
        
        for path in gt_paths[:3]:  # Top 3 GT paths
            try:
                ents = sample.extract_entities_for_path(path)
                if ents:
                    seen = set()
                    unique = []
                    for e in ents:
                        if e.lower() not in seen:
                            seen.add(e.lower())
                            unique.append(e)
                    unique = unique[:5]
                    all_candidates.extend(unique)
                    context_lines.append(f"• {path}: {', '.join(unique)}")
                    entity_found += 1
            except:
                pass
        
        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique_candidates.append(c)
        
        # Check if GT answer is in candidates
        gt_lower = set(a.lower() for a in sample.answer_entities)
        candidates_lower = set(c.lower() for c in unique_candidates)
        if gt_lower & candidates_lower:
            entity_match += 1
        
        context = "\n".join(context_lines) if context_lines else "No paths"
        
        # LLM selection
        if unique_candidates:
            prompt = f"""Question: {sample.question}

KG paths and entities:
{context}

Candidates: {', '.join(unique_candidates[:10])}

Select ONLY the entity that answers the question.

Answer:"""
        else:
            prompt = f"Question: {sample.question}\nAnswer directly:"
        
        answer = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
        answer = answer.strip().strip('"\'').split('\n')[0].rstrip('.')
        
        f1 = evaluator._compute_f1(answer, sample.answer_entities)
        em = evaluator._compute_exact_match(answer, sample.answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
        
    print(f'\n{"="*60}')
    print(f'ORACLE RESULTS ({count} samples)')
    print(f'{"="*60}')
    print(f'F1 Score: {100*f1_sum/count:.1f}%')
    print(f'Exact Match: {100*em_sum/count:.1f}%')
    print(f'\nEntity in candidates: {100*entity_match/count:.1f}%')
    print(f'Paths with entities: {entity_found}')
    
    if f1_sum/count >= 0.85:
        print("\n🎉 ORACLE TARGET ACHIEVED: F1 >= 85%!")

if __name__ == "__main__":
    run_oracle_evaluation()
