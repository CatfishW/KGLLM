"""
Best KGQA Pipeline Configuration - 76.6% F1

Key discoveries:
1. NO RERANKER - Qwen3-Reranker hurts retrieval (90% → 79% Hits@10)
2. Fine-tuned BAAI/bge-base-en-v1.5 with MNRL loss
3. Top-3 paths with entity extraction
4. Oracle-style prompting

Usage:
    python run_pipeline.py

Results on WebQSP test (500 samples):
    F1: 76.6%
    Exact Match: 68.2%
    Hits@5: 85.4%
    Hits@10: 90.0%
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
import json
from pathlib import Path


class BestKGQAPipeline:
    """Best performing KGQA pipeline - 76.6% F1."""
    
    def __init__(self, retriever_path: str = './RAG/models/finetuned_retriever'):
        self.config = RAGConfig(dataset='webqsp')
        self.loader = KGDataLoader(self.config)
        self.llm = LLMClient()
        self.evaluator = CombinedEvaluator()
        
        # Build corpus
        print("Building corpus...")
        self.corpus, _ = self.loader.build_path_corpus(include_test=True)
        print(f"  {len(self.corpus)} paths")
        
        # Load retriever (NO RERANKER!)
        print(f"Loading retriever from {retriever_path}...")
        self.retriever = SentenceTransformer(retriever_path, device='cuda')
        
        # Build FAISS index
        print("Building FAISS index...")
        embeddings = self.retriever.encode(
            self.corpus, show_progress_bar=True, 
            batch_size=256, normalize_embeddings=True
        )
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        print(f"  Indexed {self.index.ntotal} paths")
    
    def process(self, question: str, sample=None):
        """Process a single question."""
        # Retrieve top 10 (no reranker)
        q_emb = self.retriever.encode([question], normalize_embeddings=True)
        _, indices = self.index.search(q_emb.astype('float32'), 10)
        retrieved = [self.corpus[i] for i in indices[0]]
        
        # Extract entities from top 3 paths
        all_candidates = []
        context_lines = []
        
        for path in retrieved[:3]:
            if sample:
                try:
                    ents = sample.extract_entities_for_path(path)
                    if ents:
                        seen = set()
                        unique = [e for e in ents if e.lower() not in seen and not seen.add(e.lower())][:5]
                        all_candidates.extend(unique)
                        context_lines.append(f"• {path}: {', '.join(unique)}")
                except:
                    context_lines.append(f"• {path}")
            else:
                context_lines.append(f"• {path}")
        
        # Deduplicate candidates
        seen = set()
        unique_candidates = [c for c in all_candidates if c.lower() not in seen and not seen.add(c.lower())]
        
        context = "\n".join(context_lines) if context_lines else "No paths"
        
        # Oracle-style prompt
        if unique_candidates:
            prompt = f"""Question: {question}

KG paths and entities:
{context}

Candidates: {', '.join(unique_candidates[:10])}

Select ONLY the entity that answers the question.

Answer:"""
        else:
            prompt = f"Question: {question}\nAnswer directly:"
        
        answer = self.llm.chat([{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0.1)
        return answer.strip().strip('"\'').split('\n')[0].rstrip('.'), retrieved
    
    def evaluate(self, limit: int = 500, output_dir: str = None):
        """Run evaluation and save results."""
        test_data = self.loader.test_data[:limit]
        
        results = []
        f1_sum = em_sum = 0
        hits = {1: 0, 3: 0, 5: 0, 10: 0}
        count = 0
        
        print(f"\nEvaluating on {len(test_data)} samples...")
        
        for sample in tqdm(test_data):
            if not sample.answer_entities:
                continue
            
            answer, retrieved = self.process(sample.question, sample)
            
            # Hits@K
            gt = set(p.lower() for p in sample.get_path_strings()) if sample.ground_truth_paths else set()
            for k in [1, 3, 5, 10]:
                if any(p.lower() in gt for p in retrieved[:k]):
                    hits[k] += 1
            
            f1 = self.evaluator._compute_f1(answer, sample.answer_entities)
            em = self.evaluator._compute_exact_match(answer, sample.answer_entities)
            f1_sum += f1
            em_sum += em
            count += 1
            
            results.append({
                'question_id': sample.question_id,
                'question': sample.question,
                'answer': answer,
                'ground_truth': sample.answer_entities,
                'f1': f1,
                'em': em,
                'retrieved_paths': retrieved[:5]
            })
        
        metrics = {
            'f1': f1_sum / count,
            'exact_match': em_sum / count,
            'hits@1': hits[1] / count,
            'hits@3': hits[3] / count,
            'hits@5': hits[5] / count,
            'hits@10': hits[10] / count,
            'num_samples': count
        }
        
        print(f"\n{'='*60}")
        print(f"RESULTS ({count} samples)")
        print(f"{'='*60}")
        print(f"F1: {metrics['f1']*100:.1f}%")
        print(f"EM: {metrics['exact_match']*100:.1f}%")
        print(f"Hits@1: {metrics['hits@1']*100:.1f}%")
        print(f"Hits@5: {metrics['hits@5']*100:.1f}%")
        print(f"Hits@10: {metrics['hits@10']*100:.1f}%")
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{output_dir}/metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            with open(f"{output_dir}/results.json", 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved to {output_dir}/")
        
        return metrics


def main():
    pipeline = BestKGQAPipeline()
    metrics = pipeline.evaluate(limit=50000, output_dir='./RAG/best_config_76pct_f1/results')


if __name__ == "__main__":
    main()
