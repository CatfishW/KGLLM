# CWQ Optimization - Best Configuration Tracker

## Current Best: 55.9% F1 (Entity-Enhanced Paths v2)

**Model:** `RAG/models/finetuned_retriever_cwq_v2_fix`

**Full Test Results (3531 samples):**
- **F1: 55.9%** (target: 63.8%, **gap: -7.9%**)
- **EM: 44.8%**
- **Hits@1: 55.4%** (target: 70.3%, **gap: -14.9%**)
- **Hits@5: 71.1%**
- **Hits@10: 74.4%**
- **Candidate Recall: 63.0%**

**500-Sample Results:**
- F1: 60.3% (overfitted on small sample)
- Hits@1: 57.4%

**Target (D-RAG):**
- F1: 63.8%
- Hits@1: 70.3%

## Training Details

**Data:**
- 140,316 triplets (Entity-Enhanced Paths)
- 58,752 unique paths in training
- 66,631 unique paths in full corpus

**Model:**
- Base: BAAI/bge-base-en-v1.5
- Training: 5 epochs, 40 minutes
- Final accuracy: 99.98% triplet accuracy
- Batch size: 16 (OOMed at 32)

**Path Format:**
```
Entity1 -> Relation1 -> Entity2 -> Relation2 -> Entity3
```

## Identified Bottlenecks

### 1. **Retrieval (Primary)** ⚠️
- Hits@1: 55.4% (vs 70.3% target) - **14.9% gap**
- Hits@10: 74.4% (reasonable)
- **Root Cause:** Dense retriever alone can't match long Entity-Enhanced Paths well
- **Impact:** Limits F1 ceiling to ~56%

### 2. **Candidate Recall** ⚠️
- 63.0% - Answer is missing from retrieved candidates in 37% of cases
- Even perfect extraction would only achieve ~63% F1

### 3. **Answer Extraction** ✓
- Given good candidates, extraction works reasonably (45% EM)
- Not the bottleneck currently

## Experiments

### v1: Relation Chains (Failed)
- Training: Relation-only paths
- Results: 61.0% F1, 43.8% Hits@1
- Problem: No lexical overlap

### v2: Entity-Enhanced Paths (Current)
- Training: Full paths with entities
- Results: 55.9% F1, 55.4% Hits@1
- Problem: Paths too long, dense retrieval insufficient

### v3: Hybrid Retrieval (RUNNING)
- Method: BM25 + Dense (alpha=0.5)
- Hypothesis: +5-10% Hits@1 from lexical matching
- Status: Running...

## Next Experiments

1. **Hybrid Retrieval** (RUNNING) - Expect 60-65% Hits@1, 58-62% F1
2. **Query Augmentation** - Extract key entities, generate variations
3. **Larger Model** - bge-large-en-v1.5 (335M vs 109M params)
4. **Self-Consistency** - Like WebQSP (achieved 80% F1)
5. **Hard Negative Mining** - Retrain with BM25-based hard negatives

## Files Location

All scripts moved to: `RAG/cwq_experiments/`
- `generate_cwq_training.py`
- `evaluate_cwq_*.py` (fair, full, oracle, hybrid)
- `config.py` - Experiment tracker
- `README.md` - Documentation

