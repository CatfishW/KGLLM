"""
Evaluation metrics for KGQA RAG system.
Includes Recall@K, Hit@K, MRR, and detailed analysis.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
import time
from pathlib import Path

from .pipeline import RetrievalResult


@dataclass
class EvaluationMetrics:
    """Evaluation metrics at different k values."""
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    hits_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_rerank_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    num_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall@k": self.recall_at_k,
            "hits@k": self.hits_at_k,
            "mrr": self.mrr,
            "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
            "avg_rerank_time_ms": self.avg_rerank_time_ms,
            "avg_total_time_ms": self.avg_total_time_ms,
            "num_samples": self.num_samples,
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "KGQA RAG Evaluation Results",
            "=" * 50,
            f"Number of samples: {self.num_samples}",
            "",
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
            "Timing:",
            f"  Avg Retrieval: {self.avg_retrieval_time_ms:.2f} ms",
            f"  Avg Reranking: {self.avg_rerank_time_ms:.2f} ms",
            f"  Avg Total: {self.avg_total_time_ms:.2f} ms",
            "=" * 50,
        ])
        
        return "\n".join(lines)


class RAGEvaluator:
    """
    Evaluator for the KGQA RAG system.
    
    Metrics:
    - Recall@K: Fraction of ground truth paths found in top-K
    - Hits@K: Whether at least one ground truth path is in top-K
    - MRR: Mean Reciprocal Rank of first correct path
    """
    
    def __init__(self, eval_ks: List[int] = None):
        self.eval_ks = eval_ks or [1, 3, 5, 10, 20, 50, 100]
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path string for comparison."""
        # Remove extra spaces, lowercase
        return " -> ".join([
            r.strip().lower() 
            for r in path.replace("->", " -> ").split("->")
        ])
    
    def _compute_recall_at_k(
        self,
        retrieved: List[str],
        ground_truth: List[str],
        k: int
    ) -> float:
        """Compute recall at k."""
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
        """Compute hits at k (binary: 1 if any gt in top-k, 0 otherwise)."""
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
    
    def evaluate(self, results: List[RetrievalResult]) -> EvaluationMetrics:
        """
        Evaluate a list of retrieval results.
        
        Args:
            results: List of RetrievalResult from the pipeline
        
        Returns:
            EvaluationMetrics object with all computed metrics
        """
        if not results:
            return EvaluationMetrics()
        
        # Accumulators
        recall_sums = defaultdict(float)
        hits_sums = defaultdict(int)
        mrr_sum = 0.0
        total_ret_time = 0.0
        total_rerank_time = 0.0
        
        num_valid = 0
        
        for result in results:
            if not result.ground_truth_paths:
                continue
            
            num_valid += 1
            retrieved = result.reranked_paths
            gt = result.ground_truth_paths
            
            # Compute metrics at each k
            for k in self.eval_ks:
                recall_sums[k] += self._compute_recall_at_k(retrieved, gt, k)
                hits_sums[k] += self._compute_hits_at_k(retrieved, gt, k)
            
            # MRR
            mrr_sum += self._compute_reciprocal_rank(retrieved, gt)
            
            # Timing
            total_ret_time += result.retrieval_time_ms
            total_rerank_time += result.rerank_time_ms
        
        if num_valid == 0:
            return EvaluationMetrics()
        
        # Average metrics
        metrics = EvaluationMetrics(
            recall_at_k={k: v / num_valid for k, v in recall_sums.items()},
            hits_at_k={k: v / num_valid for k, v in hits_sums.items()},
            mrr=mrr_sum / num_valid,
            avg_retrieval_time_ms=total_ret_time / num_valid,
            avg_rerank_time_ms=total_rerank_time / num_valid,
            avg_total_time_ms=(total_ret_time + total_rerank_time) / num_valid,
            num_samples=num_valid,
        )
        
        return metrics
    
    def generate_report(
        self,
        results: List[RetrievalResult],
        metrics: EvaluationMetrics,
        output_path: Optional[Path] = None,
        include_examples: int = 5,
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: List of RetrievalResult
            metrics: Computed metrics
            output_path: Path to save report (optional)
            include_examples: Number of example results to include
        
        Returns:
            Report as string
        """
        lines = [
            "# KGQA RAG System Evaluation Report",
            "",
            f"**Generated at**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary Metrics",
            "",
            "### Recall@K",
            "| K | Recall |",
            "|---|--------|",
        ]
        
        for k in sorted(metrics.recall_at_k.keys()):
            lines.append(f"| {k} | {metrics.recall_at_k[k]:.4f} |")
        
        lines.extend([
            "",
            "### Hits@K",
            "| K | Hits |",
            "|---|------|",
        ])
        
        for k in sorted(metrics.hits_at_k.keys()):
            lines.append(f"| {k} | {metrics.hits_at_k[k]:.4f} |")
        
        lines.extend([
            "",
            f"### MRR: {metrics.mrr:.4f}",
            "",
            "## Timing Analysis",
            "",
            f"- **Average Retrieval Time**: {metrics.avg_retrieval_time_ms:.2f} ms",
            f"- **Average Reranking Time**: {metrics.avg_rerank_time_ms:.2f} ms",
            f"- **Average Total Time**: {metrics.avg_total_time_ms:.2f} ms",
            f"- **Throughput**: {1000 / metrics.avg_total_time_ms:.2f} queries/sec" if metrics.avg_total_time_ms > 0 else "- **Throughput**: N/A",
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
                    f"**Ground Truth Paths**: ",
                    "```",
                    *[f"  - {p}" for p in result.ground_truth_paths[:5]],
                    "```",
                    "",
                    f"**Top Retrieved Paths**:",
                    "```",
                    *[f"  {j+1}. {p} (score: {s:.4f})" 
                      for j, (p, s) in enumerate(zip(result.reranked_paths[:5], result.reranked_scores[:5]))],
                    "```",
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
        results: List[RetrievalResult],
        metrics: EvaluationMetrics,
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
                "ground_truth_paths": r.ground_truth_paths,
                "reranked_paths": r.reranked_paths,
                "reranked_scores": r.reranked_scores,
                "retrieval_time_ms": r.retrieval_time_ms,
                "rerank_time_ms": r.rerank_time_ms,
            })
        
        with open(output_dir / "detailed_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to: {output_dir}")
