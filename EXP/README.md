# KG-RAG: Knowledge Graph Path Retrieval for Question Answering

A RAG-based system for retrieving relevant relation paths from Knowledge Graphs to answer natural language questions using the WebQSP dataset.

## Quick Start

### 1. Build the Index (one-time setup)

```bash
cd d:\KGLLM
python -m EXP.path_indexer --data_dir Data/webqsp_final --output_dir EXP/index
```

### 2. Run Interactive Demo

```bash
python -m EXP.demo
```

### 3. Use in Code

```python
from EXP.pipeline import KGRAGPipeline

# Initialize pipeline
pipeline = KGRAGPipeline(index_path="EXP/index")

# Ask a question
result = pipeline.ask("what is the name of justin bieber brother", top_k=5)

# Print retrieved paths
for path in result.paths:
    print(f"Score: {path.score:.3f} | {path.relation_chain}")
```

## Features

- **FAISS Vector Index**: Fast similarity search over 5,000+ unique relation chains
- **Sentence-Transformer Embeddings**: Semantic matching between questions and paths
- **Dual Indexing Strategy**: Indexes both relation chains and question-path pairs
- **Optional Re-ranking**: Cross-encoder for improved relevance scoring

## Architecture

```
User Question
     ↓
Question Encoder (sentence-transformers)
     ↓
FAISS Similarity Search
     ↓
Top-K Candidate Paths
     ↓
(Optional) Re-ranker
     ↓
Final Retrieved Paths
```

## Files

- `config.py` - Configuration settings
- `path_indexer.py` - Build FAISS index from parquet data
- `retriever.py` - RAG retrieval implementation
- `pipeline.py` - End-to-end pipeline
- `demo.py` - Interactive demo

## Requirements

```bash
pip install faiss-cpu sentence-transformers pandas pyarrow
```
