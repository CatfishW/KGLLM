# GSR vs. Current Codebase: Comprehensive Comparison

## Executive Summary

This document compares the **GSR (Generative Subgraph Retrieval)** approach from the EMNLP 2024 paper with our current **KG Path Diffusion** implementation. The two approaches differ fundamentally in their objectives, architectures, and data processing pipelines.

---

## 1. Core Philosophy & Objective

### GSR (Generative Subgraph Retrieval)
- **Goal**: Generate **subgraph IDs** that identify relevant subgraphs in the knowledge graph
- **Approach**: Two-stage pipeline
  1. **Retriever (GSR)**: Question → Subgraph ID
  2. **Reader**: Question + Retrieved Subgraph → Answer
- **Output**: Subgraph identifier (not the actual path)

### Our Codebase (KG Path Diffusion)
- **Goal**: Generate **full reasoning paths** (sequences of entities and relations)
- **Approach**: Single-stage end-to-end generation
- **Output**: Complete path: `[entity1, relation1, entity2, relation2, ..., entityN]`

---

## 2. Model Architecture

### GSR Architecture
```
Question → T5 Encoder-Decoder → Subgraph ID
```
- **Base Model**: T5 (smaller language models like T5-small/base)
- **Tokenization**: Augmented T5 tokenizer (adds special tokens for entities/relations)
- **Output Format**: Subgraph ID string (e.g., "path_12345")
- **Training**: Sequence-to-sequence generation task

### Our Architecture
```
Question + KG Subgraph → [Question Encoder + Graph Encoder] → Path Diffusion/Flow Matching → Full Path
```
- **Question Encoder**: Pretrained transformer (e.g., `all-MiniLM-L6-v2`)
- **Graph Encoder**: Hybrid (RGCN + Transformer) or pure RGCN
- **Path Generator**: Discrete diffusion or flow matching
- **Output Format**: Sequence of entity and relation indices

---

## 3. Data Processing & Input Format

### GSR Data Processing

#### Preprocessing Steps:
1. **Index Data Construction** (`preprocess/get_example_examples.py`)
   - Builds index of subgraphs from Freebase SPARQL
   - Each subgraph gets a unique ID

2. **Pseudo-Question Generation** (`preprocess/pseudo_question_generation.py`)
   - Generates synthetic questions for training
   - Creates question-subgraph ID pairs

3. **GSR Training Data** (`preprocess/prepare_gsr_data.py`)
   - Format: `(question, subgraph_id)`
   - Training data sources: WebQSP, CWQ, pseudo-questions

4. **Reader Data** (`preprocess/prepare_reader_data.py`)
   - Format: `(question, retrieved_subgraph, answer)`
   - Generated after GSR inference

#### Input Format:
```python
# GSR Training Sample
{
    "question": "What is the capital of France?",
    "subgraph_id": "subgraph_12345"  # Target output
}

# Model Input (T5 format)
Input: "Question: What is the capital of France?"
Output: "subgraph_12345"
```

### Our Data Processing

#### Preprocessing Steps:
1. **Graph Extraction** (from ROG format)
   - Extracts local KG subgraph for each question
   - Format: List of triples `[(subj, rel, obj), ...]`

2. **Path Extraction** (`extract_reasoning_paths.py`)
   - Extracts reasoning paths from answer entities
   - Format: `{"entities": [e1, e2, ...], "relations": [r1, r2, ...]}`

3. **Data Combination** (`prepare_combined_data.py`)
   - Combines graph structure with labeled paths
   - Creates training samples with both input graph and target paths

#### Input Format:
```python
# Our Training Sample
{
    "id": "WebQTrn-0",
    "question": "What is the name of justin bieber brother?",
    "q_entity": ["m.0xxx"],
    "a_entity": ["m.0yyy"],
    "graph": [
        ["m.0xxx", "people.person.sibling_s", "m.0yyy"],
        ["m.0yyy", "common.topic.alias", "Jaxon Bieber"],
        ...
    ],
    "paths": [
        {
            "entities": ["m.0xxx", "m.0yyy"],
            "relations": ["people.person.sibling_s"]
        }
    ]
}
```

#### Model Input (Our Format):
```python
# Encoded Input
{
    "question_input_ids": [101, 2054, 2003, ...],  # Tokenized question
    "question_attention_mask": [1, 1, 1, ...],
    "graph_batch": {
        "edge_index": [[0, 1], [1, 0]],  # PyG format
        "edge_type": [rel_idx1, rel_idx2],
        "node_ids": [entity_idx1, entity_idx2]
    },
    "target_paths": {
        "path_entities": [BOS, e1_idx, e2_idx, EOS],
        "path_relations": [r1_idx, r2_idx]
    }
}
```

---

## 4. Training Strategy

### GSR Training

#### Three Training Strategies:
1. **Retrieval Data Only**
   ```bash
   python gsr/pretrain.py --pretrain_data webqsp cwq
   ```

2. **Joint Training**
   ```bash
   python gsr/pretrain.py --pretrain_data pq webqsp cwq
   ```

3. **Pretrain + Finetune**
   ```bash
   # Step 1: Pretrain on pseudo-questions
   python gsr/pretrain.py --pretrain_data pq
   
   # Step 2: Finetune on real data
   python gsr/finetune.py --finetune_data webqsp cwq
   ```

#### Training Data Sources:
- **Pseudo-questions (pq)**: Synthetic question-subgraph pairs
- **WebQSP**: Real question-answer pairs
- **CWQ**: Complex WebQuestions dataset

### Our Training

#### Single-Stage Training:
```bash
python train.py --config configs/diffusion.yaml
```

#### Training Features:
- **Multi-path training**: Learns to generate ALL diverse paths per question
- **Path diversity**: Selects diverse paths covering different answer entities
- **Relation-only mode**: Can generate only relations (entities frozen)
- **Entity+relation mode**: Generates both entities and relations

#### Training Data:
- Single dataset with graph + paths
- No separate pretraining stage
- End-to-end optimization

---

## 5. Model Input Comparison

### GSR Model Input

```python
# T5 Input Format
input_text = f"Question: {question}"
target_text = f"{subgraph_id}"

# Tokenization
input_ids = tokenizer.encode(input_text)
target_ids = tokenizer.encode(target_text)

# Model sees:
# Encoder: Question tokens
# Decoder: Subgraph ID tokens
```

**Key Characteristics:**
- Pure text-to-text generation
- No explicit graph structure in input
- Subgraph information encoded in ID
- Uses augmented tokenizer with entity/relation tokens

### Our Model Input

```python
# Multi-modal Input
{
    # Text: Question
    "question_input_ids": [batch, seq_len],
    "question_attention_mask": [batch, seq_len],
    
    # Graph: KG Subgraph
    "graph_batch": {
        "node_ids": [total_nodes],  # Global entity indices
        "edge_index": [2, total_edges],  # Graph connectivity
        "edge_type": [total_edges],  # Relation indices
        "batch": [total_nodes]  # Graph batching
    },
    
    # Optional: Node text features
    "node_input_ids": [total_nodes, node_seq_len],
    "node_attention_mask": [total_nodes, node_seq_len]
}
```

**Key Characteristics:**
- Multi-modal: Text (question) + Graph (KG subgraph)
- Explicit graph structure via PyG Batch
- Node embeddings from graph encoder
- Can use entity embeddings or tokenized entity names

---

## 6. Output Format Comparison

### GSR Output

```python
# Generated Output
subgraph_id = "subgraph_12345"

# Then retrieve actual subgraph:
subgraph = index[subgraph_id]  # Contains triples, entities, etc.
```

**Characteristics:**
- Indirect: Generates ID, then retrieves subgraph
- Subgraph must be pre-indexed
- Reader model uses retrieved subgraph

### Our Output

```python
# Generated Output
{
    "entities": [BOS, e1_idx, e2_idx, ..., eN_idx, EOS],
    "relations": [r1_idx, r2_idx, ..., rN-1_idx]
}

# Direct path representation
path = [
    (entity1, relation1, entity2),
    (entity2, relation2, entity3),
    ...
]
```

**Characteristics:**
- Direct: Generates actual path sequence
- No indexing required
- Path can be directly executed on KG

---

## 7. Key Differences Summary

| Aspect | GSR | Our Codebase |
|--------|-----|--------------|
| **Output Type** | Subgraph ID | Full path sequence |
| **Architecture** | T5 encoder-decoder | Question encoder + Graph encoder + Diffusion |
| **Input** | Question text only | Question + KG subgraph |
| **Training** | Two-stage (retriever + reader) | Single-stage end-to-end |
| **Graph Handling** | Implicit (via subgraph ID) | Explicit (PyG graph structure) |
| **Path Representation** | Indirect (ID → retrieve) | Direct (sequence generation) |
| **Model Size** | Smaller (T5-small/base) | Larger (custom architecture) |
| **Training Data** | Pseudo-questions + real data | Real data with extracted paths |
| **Inference** | Generate ID → Retrieve → Reader | Direct path generation |

---

## 8. Advantages & Trade-offs

### GSR Advantages:
1. **Simplicity**: Text-to-text generation, easier to implement
2. **Efficiency**: Smaller models (T5-small/base)
3. **Scalability**: Subgraph indexing allows efficient retrieval
4. **Modularity**: Separates retrieval from answer generation

### GSR Limitations:
1. **Indirect**: Requires subgraph indexing step
2. **Limited Expressiveness**: Can only retrieve pre-indexed subgraphs
3. **Two-stage**: Requires training both retriever and reader

### Our Approach Advantages:
1. **Direct**: Generates actual reasoning paths
2. **Flexible**: Can generate any valid path in the graph
3. **End-to-end**: Single model, joint optimization
4. **Explicit Graph**: Leverages graph structure directly

### Our Approach Limitations:
1. **Complexity**: More complex architecture
2. **Memory**: Requires full graph encoding
3. **Training**: More challenging optimization

---

## 9. Data Format Examples

### GSR Training Data Example:
```json
{
    "question": "What is the capital of France?",
    "subgraph_id": "subgraph_12345"
}
```

### Our Training Data Example:
```json
{
    "id": "WebQTrn-0",
    "question": "What is the name of justin bieber brother?",
    "q_entity": ["m.0xxx"],
    "a_entity": ["m.0yyy"],
    "graph": [
        ["m.0xxx", "people.person.sibling_s", "m.0yyy"],
        ["m.0yyy", "common.topic.alias", "Jaxon Bieber"]
    ],
    "paths": [
        {
            "entities": ["m.0xxx", "m.0yyy"],
            "relations": ["people.person.sibling_s"]
        }
    ]
}
```

---

## 10. Recommendations for Integration

If we want to incorporate GSR ideas into our codebase:

1. **Subgraph ID Generation**: Add a mode to generate subgraph IDs instead of full paths ✅ **DONE**
2. **Pseudo-Question Generation**: Implement synthetic question generation for data augmentation
3. **Two-Stage Training**: Consider pretraining on pseudo-questions, then finetuning on real data
4. **T5 Integration**: Add T5 as an alternative path generator (text-to-text) ✅ **DONE**
5. **Subgraph Indexing**: Build subgraph index for efficient retrieval ✅ **DONE**

### Integration Status

✅ **COMPLETED**: GSR has been fully integrated into the repository!

- **Location**: `Core/gsr/` directory
- **Components**: Subgraph index, T5 generator, reader model, training/inference scripts
- **Documentation**: See `Core/gsr/README.md` and `GSR_Integration_Summary.md`
- **Quick Start**: See `Core/gsr/QUICKSTART.md`

You can now use GSR alongside the existing path diffusion model!

---

## 11. Conclusion

**GSR** and **our approach** solve the same problem (multi-hop KGQA) but with fundamentally different strategies:

- **GSR**: Indirect retrieval via subgraph IDs (simpler, more modular)
- **Our approach**: Direct path generation (more flexible, end-to-end)

Both have merits. GSR's strength is simplicity and efficiency, while our approach offers more direct control over path generation and explicit graph structure utilization.

The choice depends on:
- **Use case**: Do you need pre-indexed subgraphs or flexible path generation?
- **Resources**: Can you afford larger models or need smaller ones?
- **Complexity**: Prefer modularity or end-to-end optimization?

