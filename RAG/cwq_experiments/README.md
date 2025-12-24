# CWQ Optimization Experiments

This folder contains all experiments for optimizing the CWQ benchmark to beat D-RAG's baseline (63.8% F1, 70.3% Hits@1).

## Current Best: 60.3% F1 (Entity-Enhanced Paths)

**Gap to D-RAG:**
- F1: -3.5% (60.3% vs 63.8%)
- Hits@1: -12.9% (57.4% vs 70.3%)

**Bottleneck:** Retrieval (low Hits@1)

## Files

### Training & Data Generation
- `generate_cwq_training.py` - Generate training triplets from CWQ data (Entity-Enhanced Paths)
- Model: `../models/finetuned_retriever_cwq_v2_fix/` (99.98% triplet accuracy)

### Evaluation Scripts
- `evaluate_cwq_fair.py` - Fair evaluation (500 samples, no oracle)
- `evaluate_cwq_full.py` - Full test set (3531 samples) **[RUNNING]**
- `evaluate_cwq_oracle_entity.py` - Oracle with GT entities (extraction ceiling) **[RUNNING]**
- `evaluate_cwq_hybrid.py` - Hybrid BM25+Dense retrieval **[NEXT]**

### Analysis
- `analyze_cwq_data.py` - Dataset statistics
- `diagnose_cwq_retrieval.py` - Failure case analysis
- `CWQ_best_config.md` - Progress tracker

## Results

### v1: Relation Chains (Failed)
- Training: Relation-only paths ("R1 -> R2 -> R3")
- Results: 61.0% F1, 43.8% Hits@1
- Problem: No lexical overlap with questions

### v2: Entity-Enhanced Paths (Current)
- Training: Full paths ("E1 -> R1 -> E2 -> R2 -> E3")
- Results: 60.3% F1, 57.4% Hits@1 (500 samples)
- Problem: Paths too long, Hits@1 still low

### v3: Hybrid Retrieval (Testing)
- Method: BM25 + Dense (alpha=0.5)
- Hypothesis: Combine lexical + semantic matching
- Status: Preparing to run

## Next Steps

1. **Hybrid Retrieval** - Combine BM25 + Dense (expected +5-10% Hits@1)
2. **Query Augmentation** - Extract entities, generate variations
3. **Larger Model** - Try bge-large-en-v1.5 if needed
4. **Self-Consistency** - Multiple passes like WebQSP (80% F1)

## D-RAG Comparison

**D-RAG (Paper):**
- Hits@1: 70.3%
- F1: 63.8%
- Likely uses: Hybrid retrieval, larger model, query augmentation

**Ours (Current):**
- Hits@1: 57.4%
- F1: 60.3%
- Method: Dense retriever only (bge-base)

**Our Target:**
- Hits@1: >70%
- F1: >64%
