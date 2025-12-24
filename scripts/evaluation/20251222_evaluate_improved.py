"""
Improved Entity Extraction with Better Filtering and Multi-Hop.
Goal: Push F1 from 76.5% toward 85% by improving extraction accuracy.
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.combined_evaluator import CombinedEvaluator
from sentence_transformers import SentenceTransformer
import faiss
import re


def normalize_entity(entity: str) -> str:
    """Normalize entity for matching."""
    entity = entity.lower().strip()
    # Remove common prefixes
    entity = re.sub(r'^(the|a|an)\s+', '', entity)
    # Remove parenthetical notes
    entity = re.sub(r'\s*\([^)]*\)', '', entity)
    return entity.strip()


def run_improved_evaluation():
    print("=" * 60)
    print("Improved Entity Extraction + Answer Filtering")
    print("=" * 60)
    
    config = RAGConfig(dataset='webqsp')
    loader = KGDataLoader(config)
    llm = LLMClient()
    evaluator = CombinedEvaluator()

    corpus, _ = loader.build_path_corpus(include_test=True)
    print(f"Corpus: {len(corpus)} paths")
    
    retriever = SentenceTransformer('./RAG/models/finetuned_retriever', device='cuda')
    
    embeddings = retriever.encode(corpus, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    test_data = loader.test_data[:500]
    
    f1_sum = em_sum = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    count = 0
    
    print(f'\nProcessing {len(test_data)} samples...')
    
    for sample in tqdm(test_data):
        if not sample.answer_entities:
            continue
        
        # Retrieve top 10 (no reranker)
        q_emb = retriever.encode([sample.question], normalize_embeddings=True)
        _, indices = index.search(q_emb.astype('float32'), 10)
        retrieved = [corpus[i] for i in indices[0]]
        
        # Hits@K
        gt = set(p.lower() for p in sample.get_path_strings()) if sample.ground_truth_paths else set()
        for k in [1, 3, 5, 10]:
            if any(p.lower() in gt for p in retrieved[:k]):
                hits[k] += 1
        
        # Entity extraction from top 5 paths (more paths = better coverage)
        all_candidates = []
        context_lines = []
        
        for path in retrieved[:5]:  # Top 5 for better coverage
            try:
                ents = sample.extract_entities_for_path(path)
                if ents:
                    seen = set()
                    unique = []
                    for e in ents:
                        norm_e = normalize_entity(e)
                        if norm_e and norm_e not in seen:
                            seen.add(norm_e)
                            unique.append(e)
                    unique = unique[:5]
                    all_candidates.extend(unique)
                    context_lines.append(f"• {path}: {', '.join(unique)}")
            except:
                pass
        
        # Deduplicate with normalization
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            norm_c = normalize_entity(c)
            if norm_c and norm_c not in seen:
                seen.add(norm_c)
                unique_candidates.append(c)
        
        context = "\n".join(context_lines) if context_lines else "No paths"
        
        # Improved few-shot prompt with examples
        if unique_candidates:
            prompt = f"""Extract the answer entity for the question.

Examples:
Q: what movies does taylor lautner play in?
Candidates: Twilight, The Twilight Saga, Robert Pattinson, ...
Answer: Twilight, The Twilight Saga

Q: what does jimmy carter do now?
Candidates: 39th President of the United States, Georgia, Nobel Peace Prize, ...  
Answer: 39th President of the United States

Now answer:
Q: {sample.question}
Candidates: {', '.join(unique_candidates[:15])}

Context:
{context}

Answer (entity names only, comma-separated if multiple):"""
        else:
            prompt = f"Q: {sample.question}\nAnswer:"
        
        answer = llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
        answer = answer.strip().strip('"\'').split('\n')[0].rstrip('.')
        
        # Post-process: filter to valid candidates
        if unique_candidates:
            answer_parts = [a.strip() for a in answer.split(',')]
            valid_answers = []
            for part in answer_parts:
                norm_part = normalize_entity(part)
                # Check if answer matches any candidate
                for cand in unique_candidates:
                    if normalize_entity(cand) == norm_part or norm_part in normalize_entity(cand):
                        valid_answers.append(cand)
                        break
                else:
                    valid_answers.append(part)  # Keep original if no match
            answer = ', '.join(valid_answers) if valid_answers else answer
        
        f1 = evaluator._compute_f1(answer, sample.answer_entities)
        em = evaluator._compute_exact_match(answer, sample.answer_entities)
        f1_sum += f1
        em_sum += em
        count += 1
    
    print(f"\n{'='*60}")
    print(f"IMPROVED EXTRACTION ({count} samples)")
    print(f"{'='*60}")
    print(f"F1: {f1_sum/count*100:.1f}%")
    print(f"EM: {em_sum/count*100:.1f}%")
    print(f"Hits@1: {hits[1]/count*100:.1f}%")
    print(f"Hits@5: {hits[5]/count*100:.1f}%")
    print(f"Hits@10: {hits[10]/count*100:.1f}%")
    print(f"Extraction Accuracy: {100*(f1_sum/count)/(hits[10]/count):.1f}% (F1/Hits@10)")
    
    if f1_sum/count >= 0.85:
        print("\n🎉 TARGET ACHIEVED: F1 >= 85%!")
    else:
        gap = 0.85 - f1_sum/count
        print(f"\n📊 Gap to 85%: {gap*100:.1f}%")


if __name__ == "__main__":
    run_improved_evaluation()
