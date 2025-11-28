import sys
import os

# Add Core directory to sys.path to allow imports from modules
sys.path.insert(0, os.path.abspath("Core"))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.dataset import KGPathDataModule
from kg_path_diffusion import KGPathDiffusionLightning

# Set precision to highest (FP32) to avoid NaN
torch.set_float32_matmul_precision('highest')

def main():
    print("Setting up data module...")
    
    data_module = KGPathDataModule(
        train_path="Data/webqsp_final/train.parquet",
        val_path="Data/webqsp_final/val.jsonl",
        vocab_path="Data/webqsp_final/vocab.json",
        batch_size=4,
        num_workers=0,
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        max_question_length=64,
        max_path_length=10,
        max_graph_nodes=200,
        max_entities=50000,
        max_relations=5000,
        tokenize_nodes=False
    )
    
    data_module.setup('fit')
    
    print("Creating model...")
    model = KGPathDiffusionLightning(
        num_entities=data_module.vocab.num_entities,
        num_relations=data_module.vocab.num_relations,
        hidden_dim=128,
        num_graph_layers=3,
        num_diffusion_layers=6,
        num_heads=8,
        num_diffusion_steps=1000,
        max_path_length=10,
        dropout=0.1,
        use_multipath_training=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model moved to {device}")
    
    train_loader = data_module.train_dataloader()
    
    print("Starting training loop to find NaN...")
    opt = model.configure_optimizers()
    if isinstance(opt, dict):
        optimizer = opt['optimizer']
    elif isinstance(opt, tuple):
        optimizer = opt[0][0]
    else:
        optimizer = opt
        
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif hasattr(v, 'to'):
                batch[k] = v.to(device)
                
        try:
            # Forward pass
            outputs = model(batch)
            loss = outputs['loss']
            
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}!")
                print(f"Outputs: {outputs}")
                break
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Check gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name} at batch {batch_idx}!")
                    has_nan_grad = True
            
            if has_nan_grad:
                break
                
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item()}")
                
            if batch_idx > 20:
                print("No NaN found in first 20 batches.")
                break
                
        except RuntimeError as e:
            print(f"RuntimeError at batch {batch_idx}: {e}")
            break

if __name__ == "__main__":
    main()
