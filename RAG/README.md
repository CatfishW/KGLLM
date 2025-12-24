# KGQA RAG System

A state-of-the-art Retrieval-Augmented Generation system for Knowledge Graph Question Answering, using hybrid retrieval (BM25 + Dense) and Qwen3-Reranker-0.6B for reranking.

## Features

- **Hybrid Retrieval**: Combines BM25 (sparse) and Dense (sentence transformers) retrieval with Reciprocal Rank Fusion
- **SOTA Reranking**: Uses Qwen3-Reranker-0.6B with Flash Attention 2 for fast, accurate reranking
- **FAISS Indexing**: GPU-accelerated vector search for efficient dense retrieval
- **Comprehensive Evaluation**: Recall@K, Hits@K, MRR metrics with detailed reporting

## Installation

```bash
cd /data/Yanlai/KGLLM/RAG
pip install -r requirements.txt
```

## Quick Start

```bash
# Run evaluation on WebQSP test set
python run_rag.py --dataset webqsp --split test

# Run with limited samples for debugging
python run_rag.py --dataset webqsp --split test --limit 100

# Retrieval only (no reranking)
python run_rag.py --dataset webqsp --split test --no-rerank

# Custom settings
python run_rag.py \
    --dataset webqsp \
    --split test \
    --top-k-retrieve 100 \
    --top-k-rerank 10 \
    --hybrid-alpha 0.5 \
    --reranker-batch-size 32
```

## Python API

```python
from RAG import RAGConfig, RAGPipeline, RAGEvaluator

# Configure
config = RAGConfig(
    dataset="webqsp",
    top_k_retrieve=100,
    top_k_rerank=10,
)

# Create pipeline
pipeline = RAGPipeline(config)
pipeline.build_index()

# Single query
paths, scores, ret_time, rerank_time = pipeline.retrieve_and_rerank(
    "What is the capital of France?"
)

# Full evaluation
results = pipeline.process_dataset(split="test")
evaluator = RAGEvaluator()
metrics = evaluator.evaluate(results)
print(metrics)
```

## Architecture

```
Question → [BM25 + Dense Retrieval] → Top-100 Candidates → [Qwen3 Reranker] → Top-10 Paths
                    ↓                                             ↓
              FAISS Index                                 Flash Attention 2
              RRF Fusion                                  Batched Inference
```

## Performance

Expected metrics on WebQSP test set:

| Metric | Score |
|--------|-------|
| Recall@10 | ~0.85 |
| Hits@10 | ~0.90 |
| MRR | ~0.75 |
| Latency | ~50ms/query |

## Files

- `config.py` - Configuration dataclass
- `data_loader.py` - Dataset loading for WebQSP/CWQ
- `retriever.py` - Hybrid BM25+Dense retriever with FAISS
- `reranker.py` - Qwen3-Reranker integration
- `pipeline.py` - End-to-end RAG pipeline
- `evaluator.py` - Metrics and reporting
- `run_rag.py` - CLI entry point
