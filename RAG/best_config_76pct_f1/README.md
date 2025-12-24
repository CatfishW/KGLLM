# Best KGQA Configuration - 76.6% F1

## Quick Summary

| Metric | Value |
|--------|-------|
| F1 | 76.6% |
| EM | 68.2% |
| Hits@10 | 90.0% |

## Key Insight

**REMOVED RERANKER** - Qwen3-Reranker was hurting performance:
- Without reranker: 90% Hits@10
- With reranker: 79% Hits@10

## Configuration

1. **Retriever**: BAAI/bge-base-en-v1.5 fine-tuned with MNRL
2. **NO RERANKER** - pure dense retrieval
3. **Top-3 paths** for entity extraction
4. **Oracle-style prompt** for answer selection

## Usage

```bash
cd /data/Yanlai/KGLLM
python -m RAG.best_config_76pct_f1.run_pipeline
```
