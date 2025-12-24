# WebQSP Optimization Experiments

This folder contains experiments for optimizing WebQSP to achieve >85% F1.

## Current Best: 80.0% F1 (Self-Consistency k=3)

**Target: >85% F1**

**Configuration:**
- Model: `../models/finetuned_retriever_v2`
- Method: Entity-Enhanced Paths + Self-Consistency
- k=3 voting samples

## Files

- `evaluate_webqsp_full.py` - Full test evaluation with self-consistency
- `webqsp_full_results.json` - Results file (when complete)

## Experiments to Try

1. **Increase k** - k=5 or k=7 self-consistency
2. **Hybrid Retrieval** - BM25 + Dense like CWQ
3. **Larger Model** - bge-large-en-v1.5
4. **Better Prompting** - Few-shot examples
5. **Query Augmentation** - Entity extraction, HyDE

## Results History

| Version | F1 | Hits@1 | Hits@10 | Notes |
|---------|-----|--------|---------|-------|
| Baseline | 71.2% | - | - | Union RAG |
| v1 Fine-tuned | 76.5% | 67.8% | 90% | No reranker |
| v2 Oracle-style | 76.6% | - | - | Better prompting |
| v3 Self-consistency | 80.0% | - | - | k=3 voting |
| FULL TEST | TBD | TBD | TBD | Running... |
