"""
Config for CWQ experiments - tracks best results and parameters.
"""

EXPERIMENTS = {
    "v1_relation_chains": {
        "description": "Relation-only paths (R1 -> R2 -> R3)",
        "model": "finetuned_retriever_cwq_v1",
        "results": {
            "f1": 61.0,
            "hits_at_1": 43.8,
            "hits_at_10": 70.4,
            "samples": 500
        },
        "status": "failed - no lexical overlap"
    },
    
    "v2_entity_enhanced": {
        "description": "Entity-Enhanced Paths (E1 -> R1 -> E2 -> R2 -> E3)",
        "model": "../models/finetuned_retriever_cwq_v2_fix",
        "training": {
            "triplets": 140316,
            "unique_paths": 58752,
            "epochs": 5,
            "batch_size": 16,
            "accuracy": 99.98
        },
        "results_500": {
            "f1": 60.3,
            "em": 45.0,
            "hits_at_1": 57.4,
            "hits_at_10": 73.6,
            "recall": 58.0
        },
        "results_full": {
            "f1": 55.9,
            "em": 44.8,
            "hits_at_1": 55.4,
            "hits_at_5": 71.1,
            "hits_at_10": 74.4,
            "recall": 63.0,
            "samples": 3531
        },
        "status": "current - retrieval bottleneck (Hits@1=55%)"
    },
    
    "v3_hybrid_retrieval": {
        "description": "BM25 + Dense (alpha=0.5)",
        "model": "../models/finetuned_retriever_cwq_v2_fix",
        "method": "hybrid",
        "results": "running...",
        "status": "testing"
    }
}

TARGETS = {
    "d_rag": {
        "f1": 63.8,
        "hits_at_1": 70.3,
        "source": "D-RAG paper baseline"
    }
}

CURRENT_BEST = "v2_entity_enhanced"

GAPS = {
    "f1": TARGETS["d_rag"]["f1"] - EXPERIMENTS["v2_entity_enhanced"]["results_full"]["f1"],  # 7.9%
    "hits_at_1": TARGETS["d_rag"]["hits_at_1"] - EXPERIMENTS["v2_entity_enhanced"]["results_full"]["hits_at_1"]  # 14.9%
}

NEXT_STEPS = [
    "v3_hybrid_retrieval - Combine BM25 + Dense (expected +5-10% Hits@1)",
    "v4_query_augmentation - Extract entities, generate query variations",
    "v5_larger_model - Try bge-large-en-v1.5 (335M params)",
    "v6_self_consistency - Multiple passes + voting (like WebQSP 80% F1)"
]
