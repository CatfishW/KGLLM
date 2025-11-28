import torch
from data.dataset import KGPathDataset, collate_kg_batch
from torch_geometric.data import Batch
import sys

def debug_dataset():
    # Load dataset (use val data as it was the one crashing)
    data_path = "../Data/webqsp_final/val.parquet"
    vocab_path = "../Data/webqsp_final/vocab.json"
    
    print(f"Loading dataset from {data_path}...")
    try:
        dataset = KGPathDataset(
            data_path=data_path,
            vocab=None, # Load vocab from file if needed, but for shape check it might not matter if we don't use it
            # Actually we need vocab for encoding
            tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
            max_graph_nodes=180,
            build_vocab=True # Build from data if vocab file not found/used
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Dataset size: {len(dataset)}")
    
    # Check each item for consistency
    print("Checking dataset items for node_ids vs num_nodes consistency...")
    mismatch_count = 0
    for i in range(len(dataset)):
        item = dataset[i]
        node_ids = item['node_ids']
        num_nodes = item['num_nodes']
        
        if len(node_ids) != num_nodes.item():
            print(f"Mismatch at index {i}: node_ids len {len(node_ids)} != num_nodes {num_nodes.item()}")
            mismatch_count += 1
            if mismatch_count > 10:
                break
    
    if mismatch_count == 0:
        print("No mismatches found in individual items.")
    
    # Test collation
    print("Testing collation...")
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_kg_batch,
        shuffle=False
    )
    
    for i, batch in enumerate(dataloader):
        graph_batch = batch['graph_batch']
        
        # Check batch vs x size
        # x comes from node_ids
        x_len = graph_batch.node_ids.size(0)
        batch_len = graph_batch.batch.size(0)
        
        if x_len != batch_len:
            print(f"Batch {i} mismatch: node_ids len {x_len} != batch len {batch_len}")
            
            # Debug why
            # Reconstruct batch manually
            node_counts = [item['num_nodes'].item() for item in dataset[i*batch_size : (i+1)*batch_size]]
            print(f"  Node counts from items: {node_counts}")
            print(f"  Sum of node counts: {sum(node_counts)}")
            print(f"  Batch vector max: {graph_batch.batch.max()}")
            break
        
        # if i > 5:
        #     break
            
    print("Done.")

if __name__ == "__main__":
    debug_dataset()
