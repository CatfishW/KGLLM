"""
Combined Evaluator for the KGQA system.

Extended evaluation metrics for the combined diffusion + RAG pipeline.
Includes path retrieval metrics and answer accuracy metrics.
"""
import json
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from .combined_pipeline import CombinedResult


@dataclass
class CombinedMetrics:
    """Evaluation metrics for the combined pipeline."""
    
    # Path retrieval metrics
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    hits_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    
    # Source contribution
    retrieved_contribution: float = 0.0  # % paths from retrieval in final top-k
    generated_contribution: float = 0.0  # % paths from diffusion in final top-k
    
    # Answer metrics (if using LLM answer generation)
    answer_exact_match: float = 0.0
    answer_lenient_match: float = 0.0  # Allows partial entity matches
    answer_f1: float = 0.0  
    answer_entity_overlap: float = 0.0
    
    # Timing
    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_rerank_time_ms: float = 0.0
    avg_llm_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    
    num_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_metrics": {
                "recall@k": self.recall_at_k,
                "hits@k": self.hits_at_k,
                "mrr": self.mrr,
            },
            "source_contribution": {
                "retrieved_pct": self.retrieved_contribution,
                "generated_pct": self.generated_contribution,
            },
            "answer_metrics": {
                "exact_match": self.answer_exact_match,
                "lenient_match": self.answer_lenient_match,
                "f1": self.answer_f1,
                "entity_overlap": self.answer_entity_overlap,
            },
            "timing": {
                "avg_retrieval_ms": self.avg_retrieval_time_ms,
                "avg_generation_ms": self.avg_generation_time_ms,
                "avg_rerank_ms": self.avg_rerank_time_ms,
                "avg_llm_ms": self.avg_llm_time_ms,
                "avg_total_ms": self.avg_total_time_ms,
            },
            "num_samples": self.num_samples,
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "Combined KGQA System Evaluation Results",
            "=" * 60,
            f"Number of samples: {self.num_samples}",
            "",
            "PATH RETRIEVAL METRICS",
            "-" * 30,
            "Recall@K:",
        ]
        for k, v in sorted(self.recall_at_k.items()):
            lines.append(f"  @{k}: {v:.4f}")
        
        lines.extend([
            "",
            "Hits@K:",
        ])
        for k, v in sorted(self.hits_at_k.items()):
            lines.append(f"  @{k}: {v:.4f}")
        
        lines.extend([
            "",
            f"MRR: {self.mrr:.4f}",
            "",
            "SOURCE CONTRIBUTION",
            "-" * 30,
            f"Retrieved paths: {self.retrieved_contribution:.1%}",
            f"Generated paths: {self.generated_contribution:.1%}",
            "",
            "ANSWER METRICS",
            "-" * 30,
            f"Exact Match: {self.answer_exact_match:.4f}",
            f"F1 Score: {self.answer_f1:.4f}",
            f"Entity Overlap: {self.answer_entity_overlap:.4f}",
            "",
            "TIMING",
            "-" * 30,
            f"Avg Retrieval: {self.avg_retrieval_time_ms:.2f} ms",
            f"Avg Generation: {self.avg_generation_time_ms:.2f} ms",
            f"Avg Reranking: {self.avg_rerank_time_ms:.2f} ms",
            f"Avg LLM: {self.avg_llm_time_ms:.2f} ms",
            f"Avg Total: {self.avg_total_time_ms:.2f} ms",
            f"Throughput: {1000 / self.avg_total_time_ms:.2f} q/s" if self.avg_total_time_ms > 0 else "Throughput: N/A",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class CombinedEvaluator:
    """
    Evaluator for the combined KGQA pipeline.
    
    Metrics:
    - Path Retrieval: Recall@K, Hits@K, MRR
    - Source Contribution: % from retrieval vs generation
    - Answer Quality: Exact Match, F1, Entity Overlap
    - Timing Analysis
    """
    
    def __init__(self, eval_ks: List[int] = None):
        self.eval_ks = eval_ks or [1, 3, 5, 10, 20, 50, 100]
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path string for comparison."""
        return " -> ".join([
            r.strip().lower()
            for r in path.replace("->", " -> ").split("->")
        ])
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer string for comparison (SQuAD-style).
        
        Following standard QA evaluation:
        1. Lower case
        2. Remove punctuation
        3. Remove articles (a, an, the)
        4. Remove extra whitespace
        """
        import re
        import string
        
        # Lower case
        answer = answer.lower()
        
        # Remove punctuation
        answer = ''.join(ch for ch in answer if ch not in string.punctuation)
        
        # Remove articles
        articles = ['a', 'an', 'the']
        words = answer.split()
        words = [w for w in words if w not in articles]
        answer = ' '.join(words)
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        return answer
    
    def _get_answer_tokens(self, answer: str) -> Set[str]:
        """Get normalized tokens from answer string."""
        normalized = self._normalize_answer(answer)
        if not normalized:
            return set()
        return set(normalized.split())
    
    def _compute_recall_at_k(
        self,
        retrieved: List[str],
        ground_truth: List[str],
        k: int
    ) -> float:
        """Compute recall at k for paths."""
        if not ground_truth:
            return 0.0
        
        retrieved_k = set(self._normalize_path(p) for p in retrieved[:k])
        gt_normalized = set(self._normalize_path(p) for p in ground_truth)
        
        found = len(retrieved_k & gt_normalized)
        return found / len(gt_normalized)
    
    def _compute_hits_at_k(
        self,
        retrieved: List[str],
        ground_truth: List[str],
        k: int
    ) -> int:
        """Compute hits at k (binary: 1 if any gt in top-k)."""
        if not ground_truth:
            return 0
        
        retrieved_k = set(self._normalize_path(p) for p in retrieved[:k])
        gt_normalized = set(self._normalize_path(p) for p in ground_truth)
        
        return 1 if (retrieved_k & gt_normalized) else 0
    
    def _compute_reciprocal_rank(
        self,
        retrieved: List[str],
        ground_truth: List[str]
    ) -> float:
        """Compute reciprocal rank of first correct result."""
        if not ground_truth:
            return 0.0
        
        gt_normalized = set(self._normalize_path(p) for p in ground_truth)
        
        for rank, path in enumerate(retrieved, 1):
            if self._normalize_path(path) in gt_normalized:
                return 1.0 / rank
        
        return 0.0
    
    def _compute_source_contribution(
        self,
        result: CombinedResult
    ) -> Tuple[float, float]:
        """Compute % of top paths from retrieval vs generation."""
        if not result.reranked_paths:
            return 0.0, 0.0
        
        retrieved_normalized = set(self._normalize_path(p) for p in result.retrieved_paths)
        generated_normalized = set(self._normalize_path(p) for p in result.generated_paths)
        
        from_retrieval = 0
        from_generation = 0
        
        for path in result.reranked_paths:
            normalized = self._normalize_path(path)
            if normalized in retrieved_normalized:
                from_retrieval += 1
            if normalized in generated_normalized:
                from_generation += 1
        
        total = len(result.reranked_paths)
        return from_retrieval / total, from_generation / total
    
    def _compute_f1(self, prediction: str, ground_truth: List[str]) -> float:
        """Compute F1 score between prediction and ground truth strings."""
        if not ground_truth:
            return 0.0
        
        pred_tokens = set(self._normalize_answer(prediction).split())
        
        best_f1 = 0.0
        for gt in ground_truth:
            gt_tokens = set(self._normalize_answer(gt).split())
            
            if not pred_tokens or not gt_tokens:
                continue
            
            common = pred_tokens & gt_tokens
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(gt_tokens) if gt_tokens else 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)
        
        return best_f1
    
    def _compute_exact_match(self, prediction: str, ground_truth: List[str]) -> float:
        """Check if prediction matches any ground truth exactly (strict)."""
        pred_normalized = self._normalize_answer(prediction)
        
        for gt in ground_truth:
            if pred_normalized == self._normalize_answer(gt):
                return 1.0
        
        return 0.0
    
    def _compute_lenient_match(self, prediction: str, ground_truth: List[str]) -> float:
        """
        Check for lenient match where:
        - GT is contained in prediction, OR
        - Prediction is contained in GT
        
        Handles cases like 'Mobile' (GT) vs 'Mobile, Alabama' (pred)
        or 'United States Representative' (GT) vs 'Representative' (pred)
        """
        pred_normalized = self._normalize_answer(prediction)
        if not pred_normalized:
            return 0.0
        
        for gt in ground_truth:
            gt_normalized = self._normalize_answer(gt)
            if not gt_normalized:
                continue
            
            # Check containment in both directions
            if gt_normalized in pred_normalized or pred_normalized in gt_normalized:
                return 1.0
            
            # Check if main words match (for multi-word entities)
            pred_words = set(pred_normalized.split())
            gt_words = set(gt_normalized.split())
            if pred_words and gt_words:
                # If one is subset of other or significant overlap
                overlap = len(pred_words & gt_words)
                min_len = min(len(pred_words), len(gt_words))
                if overlap >= min_len * 0.7:  # 70% overlap
                    return 1.0
        
        return 0.0
    
    def _compute_entity_overlap(
        self, 
        prediction: str, 
        ground_truth_entities: List[str]
    ) -> float:
        """Compute overlap between predicted answer and ground truth entities."""
        if not ground_truth_entities:
            return 0.0
        
        pred_lower = prediction.lower()
        
        matches = 0
        for entity in ground_truth_entities:
            if entity.lower() in pred_lower:
                matches += 1
        
        return matches / len(ground_truth_entities)
    
    def evaluate(self, results: List[CombinedResult]) -> CombinedMetrics:
        """
        Evaluate a list of combined results.
        
        Args:
            results: List of CombinedResult from the pipeline
            
        Returns:
            CombinedMetrics object with all computed metrics
        """
        if not results:
            return CombinedMetrics()
        
        # Accumulators
        recall_sums = defaultdict(float)
        hits_sums = defaultdict(int)
        mrr_sum = 0.0
        
        retrieved_contrib_sum = 0.0
        generated_contrib_sum = 0.0
        
        em_sum = 0.0
        lenient_sum = 0.0
        f1_sum = 0.0
        entity_overlap_sum = 0.0
        
        total_ret_time = 0.0
        total_gen_time = 0.0
        total_rerank_time = 0.0
        total_llm_time = 0.0
        
        num_path_valid = 0
        num_answer_valid = 0
        
        for result in results:
            # Path metrics
            if result.ground_truth_paths:
                num_path_valid += 1
                
                for k in self.eval_ks:
                    recall_sums[k] += self._compute_recall_at_k(
                        result.reranked_paths, result.ground_truth_paths, k
                    )
                    hits_sums[k] += self._compute_hits_at_k(
                        result.reranked_paths, result.ground_truth_paths, k
                    )
                
                mrr_sum += self._compute_reciprocal_rank(
                    result.reranked_paths, result.ground_truth_paths
                )
                
                ret_contrib, gen_contrib = self._compute_source_contribution(result)
                retrieved_contrib_sum += ret_contrib
                generated_contrib_sum += gen_contrib
            
            # Answer metrics
            if result.generated_answer and result.ground_truth_answers:
                num_answer_valid += 1
                em_sum += self._compute_exact_match(
                    result.generated_answer, result.ground_truth_answers
                )
                lenient_sum += self._compute_lenient_match(
                    result.generated_answer, result.ground_truth_answers
                )
                f1_sum += self._compute_f1(
                    result.generated_answer, result.ground_truth_answers
                )
                entity_overlap_sum += self._compute_entity_overlap(
                    result.generated_answer, result.ground_truth_answers
                )
            
            # Timing
            total_ret_time += result.retrieval_time_ms
            total_gen_time += result.generation_time_ms
            total_rerank_time += result.rerank_time_ms
            total_llm_time += result.llm_time_ms
        
        num_total = len(results)
        
        metrics = CombinedMetrics(
            recall_at_k={k: v / num_path_valid if num_path_valid else 0 
                        for k, v in recall_sums.items()},
            hits_at_k={k: v / num_path_valid if num_path_valid else 0 
                      for k, v in hits_sums.items()},
            mrr=mrr_sum / num_path_valid if num_path_valid else 0,
            
            retrieved_contribution=retrieved_contrib_sum / num_path_valid if num_path_valid else 0,
            generated_contribution=generated_contrib_sum / num_path_valid if num_path_valid else 0,
            
            answer_exact_match=em_sum / num_answer_valid if num_answer_valid else 0,
            answer_lenient_match=lenient_sum / num_answer_valid if num_answer_valid else 0,
            answer_f1=f1_sum / num_answer_valid if num_answer_valid else 0,
            answer_entity_overlap=entity_overlap_sum / num_answer_valid if num_answer_valid else 0,
            
            avg_retrieval_time_ms=total_ret_time / num_total if num_total else 0,
            avg_generation_time_ms=total_gen_time / num_total if num_total else 0,
            avg_rerank_time_ms=total_rerank_time / num_total if num_total else 0,
            avg_llm_time_ms=total_llm_time / num_total if num_total else 0,
            avg_total_time_ms=(total_ret_time + total_gen_time + total_rerank_time + total_llm_time) / num_total if num_total else 0,
            
            num_samples=num_total,
        )
        
        return metrics
    
    def generate_report(
        self,
        results: List[CombinedResult],
        metrics: CombinedMetrics,
        output_path: Optional[Path] = None,
        include_examples: int = 5,
    ) -> str:
        """Generate a comprehensive evaluation report."""
        lines = [
            "# Combined KGQA System Evaluation Report",
            "",
            f"**Generated at**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary Metrics",
            "",
            "### Path Retrieval",
            "| K | Recall | Hits |",
            "|---|--------|------|",
        ]
        
        for k in sorted(metrics.recall_at_k.keys()):
            recall = metrics.recall_at_k.get(k, 0)
            hits = metrics.hits_at_k.get(k, 0)
            lines.append(f"| {k} | {recall:.4f} | {hits:.4f} |")
        
        lines.extend([
            "",
            f"**MRR**: {metrics.mrr:.4f}",
            "",
            "### Source Contribution",
            f"- Retrieved paths in top-k: {metrics.retrieved_contribution:.1%}",
            f"- Generated paths in top-k: {metrics.generated_contribution:.1%}",
            "",
            "### Answer Quality",
            f"- Exact Match: {metrics.answer_exact_match:.4f}",
            f"- F1 Score: {metrics.answer_f1:.4f}",
            f"- Entity Overlap: {metrics.answer_entity_overlap:.4f}",
            "",
            "### Timing",
            f"- Retrieval: {metrics.avg_retrieval_time_ms:.2f} ms",
            f"- Generation: {metrics.avg_generation_time_ms:.2f} ms",
            f"- Reranking: {metrics.avg_rerank_time_ms:.2f} ms",
            f"- LLM: {metrics.avg_llm_time_ms:.2f} ms",
            f"- **Total**: {metrics.avg_total_time_ms:.2f} ms",
            "",
        ])
        
        # Example results
        if include_examples > 0 and results:
            lines.extend([
                "## Example Results",
                "",
            ])
            
            for i, result in enumerate(results[:include_examples]):
                lines.extend([
                    f"### Example {i+1}",
                    f"**Question**: {result.question}",
                    "",
                    f"**Topic Entity**: {result.topic_entity}",
                    "",
                    "**Top Retrieved Paths**:",
                    "```",
                    *[f"  {j+1}. {p} (score: {s:.4f})" 
                      for j, (p, s) in enumerate(zip(result.reranked_paths[:5], 
                                                     result.reranked_scores[:5]))],
                    "```",
                    "",
                    f"**Generated Answer**: {result.generated_answer}",
                    "",
                    f"**Ground Truth Answers**: {', '.join(result.ground_truth_answers[:5])}",
                    "",
                ])
        
        report = "\n".join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Report saved to: {output_path}")
        
        return report
    
    def save_results(
        self,
        results: List[CombinedResult],
        metrics: CombinedMetrics,
        output_dir: Path,
    ):
        """Save detailed results to JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        # Save detailed results
        results_data = []
        for r in results:
            results_data.append({
                "question_id": r.question_id,
                "question": r.question,
                "topic_entity": r.topic_entity,
                "retrieved_paths": r.retrieved_paths,
                "generated_paths": r.generated_paths,
                "reranked_paths": r.reranked_paths,
                "reranked_scores": r.reranked_scores,
                "ground_truth_paths": r.ground_truth_paths,
                "ground_truth_answers": r.ground_truth_answers,
                "generated_answer": r.generated_answer,
                "retrieval_time_ms": r.retrieval_time_ms,
                "generation_time_ms": r.generation_time_ms,
                "rerank_time_ms": r.rerank_time_ms,
                "llm_time_ms": r.llm_time_ms,
            })
        
        with open(output_dir / "detailed_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to: {output_dir}")
