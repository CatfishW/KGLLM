"""
Optimized evaluation targeting 85%+ F1.
Key improvements:
1. Self-consistency voting (multiple samples, majority vote)
2. Improved entity extraction with fallback
3. Better answer extraction prompt  
4. Answer normalization/cleaning
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

from tqdm import tqdm
from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader
from RAG.llm_client import LLMClient
from RAG.combined_evaluator import CombinedEvaluator
from RAG.reranker import Qwen3Reranker
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from collections import Counter
import re

class OptimizedEvaluator:
    def __init__(self):
        print("=" * 60)
        print("Optimized KGQA Pipeline - Target: 85%+ F1")
        print("=" * 60)
        
        self.config = RAGConfig(dataset='webqsp')
        self.loader = KGDataLoader(self.config)
        self.llm = LLMClient()
        self.evaluator = CombinedEvaluator()
        
        # Reranker
        self.reranker = Qwen3Reranker(self.config)
        self.reranker.load()
        
        # Corpus
        self.corpus, _ = self.loader.build_path_corpus(include_test=True)
        print(f"Corpus: {len(self.corpus)} paths")
        
        # Retriever
        print("\nLoading fine-tuned retriever...")
        self.retriever = SentenceTransformer('./RAG/models/finetuned_retriever', device='cuda')
        
        # FAISS index
        print("Building FAISS index...")
        embeddings = self.retriever.encode(
            self.corpus, show_progress_bar=True, 
            batch_size=256, normalize_embeddings=True
        )
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove common prefixes/suffixes
        answer = answer.strip().strip('"\'').rstrip('.')
        
        # Remove thinking/explanation patterns
        patterns = [
            r'^(The answer is|Answer:|Based on.*?,)',
            r'^\d+\.\s*',
            r'\(.*?\)',
        ]
        for pattern in patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        return answer.strip()
    
    def extract_entities_for_path(self, sample, path: str) -> list:
        """Extract entities with flexible matching."""
        try:
            ents = sample.extract_entities_for_path(path)
            # Deduplicate
            seen = set()
            unique = []
            for e in ents:
                if e.lower() not in seen:
                    seen.add(e.lower())
                    unique.append(e)
            return unique
        except:
            return []
    
    def self_consistency_vote(self, question: str, context: str, 
                              candidates: list, num_samples: int = 5) -> str:
        """Generate multiple answers and vote."""
        answers = []
        
        prompt = f"""Question: {question}

Candidate Entities: {', '.join(candidates[:15]) if candidates else 'None'}

Path Context:
{context}

Select the entity that answers the question. Return ONLY the entity name(s), comma-separated if multiple.

Answer:"""
        
        for i in range(num_samples):
            temp = 0.3 if i == 0 else 0.7  # First sample low temp, rest higher
            ans = self.llm.chat([{'role': 'user', 'content': prompt}], 
                               max_tokens=64, temperature=temp)
            ans = self.normalize_answer(ans)
            if ans:
                answers.append(ans)
        
        if not answers:
            return ""
        
        # Majority voting with normalization
        normalized_counts = Counter(a.lower() for a in answers)
        most_common = normalized_counts.most_common(1)[0][0]
        
        # Return original casing
        for ans in answers:
            if ans.lower() == most_common:
                return ans
        return answers[0]
    
    def process_sample(self, sample):
        """Process single sample with all optimizations."""
        # 1. Retrieve top 30
        q_emb = self.retriever.encode([sample.question], normalize_embeddings=True)
        _, indices = self.index.search(q_emb.astype('float32'), 30)
        retrieved = [self.corpus[i] for i in indices[0]]
        
        # 2. Rerank
        _, _, reranked = self.reranker.rerank(sample.question, retrieved, top_k=10)
        
        # 3. Check Hits
        gt = set(p.lower() for p in sample.get_path_strings()) if sample.ground_truth_paths else set()
        hits = {
            1: any(p.lower() in gt for p in reranked[:1]),
            5: any(p.lower() in gt for p in reranked[:5]),
            10: any(p.lower() in gt for p in reranked[:10]),
        }
        
        # 4. Extract entities from top paths
        all_candidates = []
        context_lines = []
        
        for path in reranked[:5]:
            ents = self.extract_entities_for_path(sample, path)
            if ents:
                all_candidates.extend(ents[:5])
                context_lines.append(f"• {path}: {', '.join(ents[:5])}")
            else:
                context_lines.append(f"• {path}")
        
        # Deduplicate candidates
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique_candidates.append(c)
        
        context = "\n".join(context_lines) if context_lines else "No paths found"
        
        # 5. Self-consistency voting
        answer = self.self_consistency_vote(
            sample.question, context, unique_candidates, num_samples=5
        )
        
        return answer, hits
    
    def evaluate(self, limit=500):
        """Run full evaluation."""
        test_data = self.loader.test_data[:limit]
        
        f1_sum = em_sum = 0
        hits_at_1 = hits_at_5 = hits_at_10 = 0
        count = 0
        
        print(f'\nProcessing {len(test_data)} samples with self-consistency voting...')
        
        for sample in tqdm(test_data):
            if not sample.answer_entities:
                continue
            
            answer, hits = self.process_sample(sample)
            
            if hits[1]: hits_at_1 += 1
            if hits[5]: hits_at_5 += 1
            if hits[10]: hits_at_10 += 1
            
            f1 = self.evaluator._compute_f1(answer, sample.answer_entities)
            em = self.evaluator._compute_exact_match(answer, sample.answer_entities)
            f1_sum += f1
            em_sum += em
            count += 1
        
        print(f'\n{"="*60}')
        print(f'RESULTS ({count} samples)')
        print(f'{"="*60}')
        print(f'F1 Score: {100*f1_sum/count:.1f}%')
        print(f'Exact Match: {100*em_sum/count:.1f}%')
        print(f'Hits@1: {100*hits_at_1/count:.1f}%')
        print(f'Hits@5: {100*hits_at_5/count:.1f}%')
        print(f'Hits@10: {100*hits_at_10/count:.1f}%')
        
        return {
            'f1': f1_sum/count,
            'em': em_sum/count,
            'hits@10': hits_at_10/count
        }


def main():
    evaluator = OptimizedEvaluator()
    results = evaluator.evaluate(limit=500)
    
    if results['f1'] >= 0.85:
        print("\n🎉 TARGET ACHIEVED: F1 >= 85%!")
    else:
        gap = 0.85 - results['f1']
        print(f"\n📊 Gap to 85%: {gap*100:.1f}%")


if __name__ == "__main__":
    main()
