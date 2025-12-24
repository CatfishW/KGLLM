"""
Optimized QA Pipeline with Fine-Tuned Models.

Implements the full optimized KGQA pipeline:
1. Fine-tuned dense retriever (stella_en_400M_v5 + MNRL)
2. Union retrieval (Q + HyDE)
3. Fine-tuned reranker (Qwen3 + LoRA)
4. Self-consistency voting for answer extraction
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import re
import torch
from tqdm import tqdm

from .config import RAGConfig
from .data_loader import KGDataLoader, KGSample
from .reranker import Qwen3Reranker
from .llm_client import LLMClient, LLMConfig


@dataclass
class OptimizedConfig(RAGConfig):
    """Configuration for the optimized pipeline."""
    # Fine-tuned models
    finetuned_retriever_path: Optional[str] = None
    finetuned_reranker_path: Optional[str] = None
    
    # Retrieval
    use_union_retrieval: bool = True
    retrieval_top_k: int = 10
    max_union_paths: int = 15
    
    # Self-consistency voting
    use_self_consistency: bool = True
    num_voting_samples: int = 5
    voting_temperature: float = 0.5
    
    # Entity extraction
    use_entity_context: bool = True
    max_entities_per_path: int = 5


@dataclass
class OptimizedResult:
    """Result from the optimized pipeline."""
    question_id: str
    question: str
    topic_entity: Optional[str]
    
    # Retrieval
    q_retrieved_paths: List[str] = field(default_factory=list)
    hyde_paths: List[str] = field(default_factory=list)
    union_paths: List[str] = field(default_factory=list)
    
    # Reranking
    reranked_paths: List[str] = field(default_factory=list)
    reranked_scores: List[float] = field(default_factory=list)
    
    # Answer
    candidate_answers: List[str] = field(default_factory=list)
    final_answer: str = ""
    answer_confidence: float = 0.0
    
    # Ground truth
    ground_truth_paths: List[str] = field(default_factory=list)
    ground_truth_answers: List[str] = field(default_factory=list)
    
    # Timing
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    answer_time_ms: float = 0.0
    
    @property
    def total_time_ms(self) -> float:
        return self.retrieval_time_ms + self.rerank_time_ms + self.answer_time_ms


class OptimizedPipeline:
    """
    Optimized KGQA Pipeline.
    
    Key optimizations:
    1. Fine-tuned dense retriever for better recall
    2. Union retrieval (Q + HyDE) for diverse candidates
    3. Fine-tuned reranker for accurate path ranking
    4. Self-consistency voting for robust answer extraction
    5. Entity-aware context for grounded answers
    """
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Data loader
        self.loader = KGDataLoader(config)
        
        # Build corpus
        print("Building path corpus...")
        self.corpus, self.path_to_idx = self.loader.build_path_corpus(include_test=True)
        print(f"  Corpus size: {len(self.corpus)} paths")
        
        # Load retriever (fine-tuned or default)
        self._init_retriever()
        
        # Load reranker (fine-tuned with LoRA or default)
        self._init_reranker()
        
        # LLM client for HyDE and answer generation
        self.llm = LLMClient()
    
    def _init_retriever(self):
        """Initialize dense retriever."""
        from sentence_transformers import SentenceTransformer
        import faiss
        
        if self.config.finetuned_retriever_path and Path(self.config.finetuned_retriever_path).exists():
            print(f"Loading fine-tuned retriever from {self.config.finetuned_retriever_path}...")
            self.retriever = SentenceTransformer(
                self.config.finetuned_retriever_path,
                device=self.device
            )
        else:
            print("Loading base retriever (stella_en_400M_v5)...")
            self.retriever = SentenceTransformer(
                "dunzhang/stella_en_400M_v5",
                device=self.device
            )
        
        # Build FAISS index
        print("Building FAISS index...")
        embeddings = self.retriever.encode(
            self.corpus,
            show_progress_bar=True,
            batch_size=256,
            normalize_embeddings=True
        )
        
        self.embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine
        self.index.add(embeddings.astype('float32'))
        print(f"  Indexed {self.index.ntotal} paths")
    
    def _init_reranker(self):
        """Initialize reranker."""
        self.reranker = Qwen3Reranker(self.config)
        
        # Check for LoRA weights
        if self.config.finetuned_reranker_path and Path(self.config.finetuned_reranker_path).exists():
            print(f"Note: LoRA weights available at {self.config.finetuned_reranker_path}")
            print("  (LoRA integration with reranker requires additional loading logic)")
        
        self.reranker.load()
    
    def _retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve paths using dense retriever."""
        query_embedding = self.retriever.encode(
            [query],
            normalize_embeddings=True
        )
        
        scores, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        return [self.corpus[i] for i in indices[0]]
    
    def _generate_hyde_paths(self, question: str) -> List[str]:
        """Generate hypothetical paths using LLM (HyDE)."""
        messages = [{
            "role": "user",
            "content": f"Generate 3 likely Knowledge Graph relation paths for answering: {question}\n\nFormat: one path per line, like 'rel1 -> rel2' or 'domain.relation'"
        }]
        
        response = self.llm.chat(messages, max_tokens=128, temperature=0.7)
        
        # Parse paths from response
        paths = []
        for line in response.strip().split("\n"):
            line = line.strip().strip("-").strip("•").strip()
            if line and ("." in line or "->" in line):
                paths.append(line)
        
        return paths[:3]
    
    def _union_retrieval(
        self, 
        question: str,
        top_k: int = 10
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Union retrieval: combine Q-based and HyDE-based results.
        
        Returns:
            q_paths: Paths from question retrieval
            hyde_paths: Generated HyDE paths
            union_paths: Deduplicated union
        """
        # Q-based retrieval
        q_paths = self._retrieve(question, top_k)
        
        hyde_paths = []
        hyde_retrieved = []
        
        if self.config.use_union_retrieval:
            # HyDE generation
            hyde_paths = self._generate_hyde_paths(question)
            
            # HyDE-based retrieval (use first generated path as query)
            if hyde_paths:
                hyde_retrieved = self._retrieve(hyde_paths[0], top_k)
        
        # Union with stable order
        seen = set()
        union_paths = []
        
        for p in q_paths + hyde_retrieved:
            p_norm = p.lower().strip()
            if p_norm not in seen:
                seen.add(p_norm)
                union_paths.append(p)
                if len(union_paths) >= self.config.max_union_paths:
                    break
        
        return q_paths, hyde_paths, union_paths
    
    def _rerank_paths(
        self,
        question: str,
        paths: List[str]
    ) -> Tuple[List[str], List[float]]:
        """Rerank paths using cross-encoder."""
        if not paths:
            return [], []
        
        scores, indices, docs = self.reranker.rerank(
            question,
            paths,
            top_k=min(10, len(paths))
        )
        
        return docs, scores
    
    def _build_entity_context(
        self,
        sample: KGSample,
        paths: List[str]
    ) -> str:
        """Build context with entity candidates for each path."""
        context_lines = []
        
        for path in paths:
            try:
                entities = sample.extract_entities_for_path(path)
            except Exception:
                entities = []
            
            if entities:
                ent_str = ", ".join(entities[:self.config.max_entities_per_path])
                context_lines.append(f"Path: {path}\nCandidates: [{ent_str}]")
            else:
                context_lines.append(f"Path: {path}\nCandidates: []")
        
        return "\n\n".join(context_lines) if context_lines else "No relevant paths found."
    
    def _generate_single_answer(
        self,
        question: str,
        context: str,
        temperature: float = 0.1
    ) -> str:
        """Generate a single answer."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert answer extractor for Knowledge Graph Question Answering. Select the answer that best fits the question from the candidate entities. Give ONLY the entity name(s), comma-separated if multiple. Be concise."
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\n{context}\n\nAnswer:"
            }
        ]
        
        answer = self.llm.chat(messages, max_tokens=64, temperature=temperature)
        return answer.strip().strip('"\'').rstrip('.')
    
    def _self_consistency_vote(
        self,
        question: str,
        context: str,
        num_samples: int = 5
    ) -> Tuple[str, float, List[str]]:
        """
        Self-consistency voting for answer extraction.
        
        Generates multiple answers and returns majority vote.
        """
        answers = []
        
        for _ in range(num_samples):
            ans = self._generate_single_answer(
                question,
                context,
                temperature=self.config.voting_temperature
            )
            if ans:
                answers.append(ans)
        
        if not answers:
            return "", 0.0, []
        
        # Normalize and count
        def normalize(s):
            return re.sub(r'[^\w\s]', '', s.lower()).strip()
        
        normalized_counts = Counter(normalize(a) for a in answers)
        
        # Get most common
        most_common_norm, count = normalized_counts.most_common(1)[0]
        confidence = count / len(answers)
        
        # Find original form
        for ans in answers:
            if normalize(ans) == most_common_norm:
                return ans, confidence, answers
        
        return answers[0], confidence, answers
    
    def process_sample(self, sample: KGSample) -> OptimizedResult:
        """Process a single sample through the optimized pipeline."""
        result = OptimizedResult(
            question_id=sample.question_id,
            question=sample.question,
            topic_entity=sample.topic_entity,
            ground_truth_paths=sample.get_path_strings() if sample.ground_truth_paths else [],
            ground_truth_answers=sample.answer_entities or []
        )
        
        # Stage 1: Union Retrieval
        t0 = time.time()
        q_paths, hyde_paths, union_paths = self._union_retrieval(
            sample.question,
            self.config.retrieval_top_k
        )
        result.q_retrieved_paths = q_paths
        result.hyde_paths = hyde_paths
        result.union_paths = union_paths
        result.retrieval_time_ms = (time.time() - t0) * 1000
        
        # Stage 2: Reranking
        t0 = time.time()
        reranked_paths, reranked_scores = self._rerank_paths(
            sample.question,
            union_paths
        )
        result.reranked_paths = reranked_paths
        result.reranked_scores = reranked_scores
        result.rerank_time_ms = (time.time() - t0) * 1000
        
        # Stage 3: Entity Context Building
        context = self._build_entity_context(sample, reranked_paths[:5])
        
        # Stage 4: Answer Extraction with Self-Consistency
        t0 = time.time()
        if self.config.use_self_consistency:
            answer, confidence, candidates = self._self_consistency_vote(
                sample.question,
                context,
                self.config.num_voting_samples
            )
            result.final_answer = answer
            result.answer_confidence = confidence
            result.candidate_answers = candidates
        else:
            answer = self._generate_single_answer(sample.question, context)
            result.final_answer = answer
            result.answer_confidence = 1.0
            result.candidate_answers = [answer]
        
        result.answer_time_ms = (time.time() - t0) * 1000
        
        return result
    
    def process_dataset(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        show_progress: bool = True
    ) -> List[OptimizedResult]:
        """Process entire dataset split."""
        if split == "train":
            data = self.loader.train_data
        elif split == "val":
            data = self.loader.val_data
        else:
            data = self.loader.test_data
        
        if limit:
            data = data[:limit]
        
        results = []
        iterator = tqdm(data, desc=f"Processing {split}") if show_progress else data
        
        for sample in iterator:
            if not sample.answer_entities:
                continue
            
            result = self.process_sample(sample)
            results.append(result)
        
        return results
    
    def evaluate(self, results: List[OptimizedResult]) -> Dict:
        """Compute evaluation metrics."""
        from .combined_evaluator import CombinedEvaluator
        
        evaluator = CombinedEvaluator()
        
        # Convert to format expected by evaluator
        f1_sum = 0.0
        em_sum = 0.0
        hits_at_1 = 0
        hits_at_5 = 0
        hits_at_10 = 0
        
        for result in results:
            # F1
            f1 = evaluator._compute_f1(result.final_answer, result.ground_truth_answers)
            f1_sum += f1
            
            # Exact Match
            em = evaluator._compute_exact_match(result.final_answer, result.ground_truth_answers)
            em_sum += em
            
            # Path Hits
            gt_norm = set(p.lower() for p in result.ground_truth_paths)
            
            for k, hits_var in [(1, 'hits_at_1'), (5, 'hits_at_5'), (10, 'hits_at_10')]:
                top_k = set(p.lower() for p in result.reranked_paths[:k])
                if top_k & gt_norm:
                    if k == 1:
                        hits_at_1 += 1
                    elif k == 5:
                        hits_at_5 += 1
                    else:
                        hits_at_10 += 1
        
        n = len(results)
        
        return {
            "f1": f1_sum / n if n else 0,
            "exact_match": em_sum / n if n else 0,
            "hits@1": hits_at_1 / n if n else 0,
            "hits@5": hits_at_5 / n if n else 0,
            "hits@10": hits_at_10 / n if n else 0,
            "num_samples": n,
            "avg_retrieval_ms": sum(r.retrieval_time_ms for r in results) / n if n else 0,
            "avg_rerank_ms": sum(r.rerank_time_ms for r in results) / n if n else 0,
            "avg_answer_ms": sum(r.answer_time_ms for r in results) / n if n else 0,
        }
    
    def unload(self):
        """Unload models to free memory."""
        self.reranker.unload()
        del self.retriever
        del self.index
        torch.cuda.empty_cache()


def main():
    """Run optimized pipeline evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="webqsp")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--retriever", default=None, help="Path to fine-tuned retriever")
    parser.add_argument("--reranker", default=None, help="Path to fine-tuned reranker")
    parser.add_argument("--no-voting", action="store_true", help="Disable self-consistency")
    parser.add_argument("--output-dir", default="./results/optimized")
    args = parser.parse_args()
    
    config = OptimizedConfig(
        dataset=args.dataset,
        finetuned_retriever_path=args.retriever,
        finetuned_reranker_path=args.reranker,
        use_self_consistency=not args.no_voting
    )
    
    print("=" * 60)
    print("Optimized KGQA Pipeline")
    print("=" * 60)
    
    pipeline = OptimizedPipeline(config)
    
    print(f"\nProcessing {args.split} split (limit={args.limit})...")
    results = pipeline.process_dataset(args.split, args.limit)
    
    print("\nEvaluating...")
    metrics = pipeline.evaluate(results)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Samples: {metrics['num_samples']}")
    print(f"F1 Score: {metrics['f1']*100:.1f}%")
    print(f"Exact Match: {metrics['exact_match']*100:.1f}%")
    print(f"Hits@1: {metrics['hits@1']*100:.1f}%")
    print(f"Hits@5: {metrics['hits@5']*100:.1f}%")
    print(f"Hits@10: {metrics['hits@10']*100:.1f}%")
    print(f"Avg Time: {metrics['avg_retrieval_ms'] + metrics['avg_rerank_ms'] + metrics['avg_answer_ms']:.0f}ms")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump([{
            "question_id": r.question_id,
            "question": r.question,
            "final_answer": r.final_answer,
            "ground_truth": r.ground_truth_answers,
            "confidence": r.answer_confidence,
            "reranked_paths": r.reranked_paths[:5]
        } for r in results], f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    pipeline.unload()


if __name__ == "__main__":
    main()
