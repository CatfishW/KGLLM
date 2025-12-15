
import sys
import os
import json
from pathlib import Path

# Add core to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import KGPathDataset, EntityRelationVocab

def build_combined_vocab():
    print("Building combined vocabulary from all datasets...")
    
    # Define all paths
    data_paths = [
        # WebQSP
        "/data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/train.parquet",
        "/data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/val.parquet",
        "/data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/test.parquet",
        # CWQ
        "/data/Yanlai/KGLLM/Data/CWQ/shortest_paths/train.parquet",
        "/data/Yanlai/KGLLM/Data/CWQ/shortest_paths/val.parquet",
        "/data/Yanlai/KGLLM/Data/CWQ/shortest_paths/test.parquet"
    ]
    
    # Initialize Vocab
    vocab = EntityRelationVocab(max_entities=20000000000, max_relations=50000)
    
    total_samples = 0
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"Warning: Path not found: {path}")
            continue
            
        print(f"Processing {path}...")
        # We use KGPathDataset to load logic, but specifically asking it to build vocab (add to our vocab)
        dataset = KGPathDataset(
            path,
            vocab=vocab,
            build_vocab=True,
            training=False, # We just want to iterate
            tokenizer_name="sentence-transformers/all-MiniLM-L6-v2" # Dummy
        )
        total_samples += len(dataset)
        
    print(f"\nProcessing complete.")
    print(f"Total samples scanned: {total_samples}")
    print(f"Vocabulary Size: {vocab.num_entities} entities, {vocab.num_relations} relations")
    
    output_path = "/data/Yanlai/KGLLM/Data/vocab_combined.json"
    print(f"Saving vocabulary to {output_path}...")
    vocab.save(output_path)
    print("Done.")

if __name__ == "__main__":
    build_combined_vocab()
