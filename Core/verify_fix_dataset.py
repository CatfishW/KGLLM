import torch
from data.dataset import KGPathDataset
import tempfile
import json
import os
import shutil

def verify_fix():
    # Create a dummy dataset file with an empty graph
    data = [
        {
            "id": "test_1",
            "question": "test question",
            "graph": [],  # Empty graph
            "paths": [],
            "q_entities": [],
            "a_entities": []
        }
    ]
    
    tmp_dir = tempfile.mkdtemp()
    data_path = os.path.join(tmp_dir, "test_data.jsonl")
    
    try:
        with open(data_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        print("Created dummy dataset with empty graph.")
        
        # Initialize dataset with tokenize_nodes=True
        print("Initializing dataset...")
        dataset = KGPathDataset(
            data_path=data_path,
            tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
            tokenize_nodes=True,
            build_vocab=True
        )
        
        print("Accessing item 0...")
        item = dataset[0]
        
        print("Successfully accessed item 0!")
        print("node_input_ids shape:", item['node_input_ids'].shape)
        
        assert item['node_input_ids'].shape == (1, 32)
        print("Verification passed: node_input_ids has correct shape (1, 32)")
        
        print(f"num_nodes: {item['num_nodes']}")
        if item['num_nodes'] != item['node_input_ids'].shape[0]:
            print("MISMATCH DETECTED: num_nodes != node_input_ids.shape[0]")
        else:
            print("Consistency check passed.")
        
    finally:
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    verify_fix()
