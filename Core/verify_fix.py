
import torch
import sys
import os

# Add Core to path
sys.path.insert(0, os.path.abspath("Z:/20251125_KGLLM/Core"))

from kg_path_diffusion import KGPathDiffusionLightning
from data.dataset import collate_kg_batch

def verify():
    print("Verifying fix...")
    
    # 1. Initialize model with new defaults
    model = KGPathDiffusionLightning(
        num_entities=1000,
        num_relations=100,
        hidden_dim=128,  # New default
        num_diffusion_steps=10,
        max_path_length=10
    )
    
    # 2. Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if trainable_params > 65000000:
        print("WARNING: Parameter count is still high!")
    else:
        print("Parameter count is within target range (~60M).")

    # 3. Dummy data forward pass
    batch_size = 4
    dummy_batch = []
    for i in range(batch_size):
        dummy_batch.append({
            'id': str(i),
            'question_input_ids': torch.randint(0, 100, (32,)),
            'question_attention_mask': torch.ones(32),
            'edge_index': torch.randint(0, 10, (2, 20)),
            'edge_type': torch.randint(0, 5, (20,)),
            'node_ids': torch.randint(0, 1000, (10,)),
            'num_nodes': torch.tensor(10),
            'q_entity_indices': torch.tensor([0]),
            'a_entity_indices': torch.tensor([1]),
            'path_entities': torch.randint(0, 1000, (10,)),
            'path_relations': torch.randint(0, 100, (9,)),
            'path_local_entities': torch.randint(0, 10, (10,)),
            'path_length': torch.tensor(10),
            'all_path_entities': torch.randint(0, 1000, (5, 10)),
            'all_path_relations': torch.randint(0, 100, (5, 9)),
            'all_path_local_entities': torch.randint(0, 10, (5, 10)),
            'all_path_lengths': torch.full((5,), 10),
            'num_paths': torch.tensor(5)
        })
    
    batch = collate_kg_batch(dummy_batch)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        # Recursive move to device for batch
        def move_to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cuda()
            elif isinstance(obj, dict):
                return {k: move_to_device(v) for k, v in obj.items()}
            elif hasattr(obj, 'to'): # PyG Batch
                return obj.to('cuda')
            return obj
        batch = move_to_device(batch)
        print("Running on GPU")
    else:
        print("Running on CPU")

    # Forward pass
    model.train()
    try:
        outputs = model(batch)
        loss = outputs['loss']
        print(f"Loss: {loss.item()}")
        
        if torch.isnan(loss):
            print("FAILED: Loss is NaN!")
        else:
            print("SUCCESS: Loss is not NaN.")
            
        # Backward pass
        loss.backward()
        print("Backward pass successful.")
        
    except Exception as e:
        print(f"FAILED: Exception during forward/backward: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
