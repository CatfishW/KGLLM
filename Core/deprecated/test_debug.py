"""
Simple debug script to test data loading and model components.
"""

import sys
import os

# Test imports one by one
print("Testing imports...")

try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"[FAIL] PyTorch: {e}")
    sys.exit(1)

try:
    import torch_geometric
    print(f"[OK] PyTorch Geometric {torch_geometric.__version__}")
except Exception as e:
    print(f"[FAIL] PyTorch Geometric: {e}")

try:
    import pytorch_lightning as pl
    print(f"[OK] PyTorch Lightning {pl.__version__}")
except Exception as e:
    print(f"[FAIL] PyTorch Lightning: {e}")

try:
    from transformers import AutoTokenizer, AutoModel
    print("[OK] Transformers")
except Exception as e:
    print(f"[FAIL] Transformers: {e}")

try:
    import pandas as pd
    print(f"[OK] Pandas {pd.__version__}")
except Exception as e:
    print(f"[FAIL] Pandas: {e}")

print("\n" + "="*60)
print("Testing data loading...")
print("="*60)

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data.dataset import KGPathDataset, EntityRelationVocab, collate_kg_batch
    print("[OK] Dataset module imported")
except Exception as e:
    print(f"[FAIL] Dataset module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test loading data
data_path = "../Data/webqsp_combined/train.jsonl"
print(f"\nLoading data from {data_path}...")

try:
    dataset = KGPathDataset(
        data_path,
        vocab=None,
        max_question_length=64,
        max_path_length=10,
        max_graph_nodes=100,  # Limit for faster testing
        build_vocab=True
    )
    print(f"[OK] Dataset loaded: {len(dataset)} samples")
    print(f"  Vocabulary: {dataset.vocab.num_entities} entities, {dataset.vocab.num_relations} relations")
except Exception as e:
    print(f"[FAIL] Dataset loading: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test getting a sample
print("\nTesting sample retrieval...")
try:
    sample = dataset[0]
    print(f"[OK] Sample retrieved")
    print(f"  ID: {sample['id']}")
    print(f"  Question shape: {sample['question_input_ids'].shape}")
    print(f"  Graph edges: {sample['edge_index'].shape}")
    print(f"  Num nodes: {sample['num_nodes']}")
    print(f"  Path entities: {sample['path_entities'].shape}")
    print(f"  Path relations: {sample['path_relations'].shape}")
except Exception as e:
    print(f"[FAIL] Sample retrieval: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test batching
print("\nTesting batch collation...")
try:
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_kg_batch,
        num_workers=0
    )
    
    batch = next(iter(loader))
    print(f"[OK] Batch created")
    print(f"  Batch size: {len(batch['ids'])}")
    print(f"  Question IDs shape: {batch['question_input_ids'].shape}")
    print(f"  Graph batch nodes: {batch['graph_batch'].num_nodes}")
    print(f"  Path entities shape: {batch['path_entities'].shape}")
except Exception as e:
    print(f"[FAIL] Batch collation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("Testing model creation...")
print("="*60)

try:
    from modules.graph_encoder import HybridGraphEncoder
    from modules.diffusion import PathDiffusionTransformer, DiscreteDiffusion
    
    num_entities = dataset.vocab.num_entities
    num_relations = dataset.vocab.num_relations
    hidden_dim = 128
    
    print(f"\nCreating graph encoder...")
    graph_encoder = HybridGraphEncoder(
        num_entities=num_entities,
        num_relations=num_relations,
        entity_dim=hidden_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_rgcn_layers=1,
        num_transformer_layers=1,
        num_heads=4
    )
    print(f"[OK] Graph encoder created")
    print(f"  Parameters: {sum(p.numel() for p in graph_encoder.parameters()):,}")
    
    print(f"\nCreating diffusion model...")
    denoiser = PathDiffusionTransformer(
        num_entities=num_entities,
        num_relations=num_relations,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        question_dim=hidden_dim,
        graph_dim=hidden_dim,
        max_path_length=10
    )
    print(f"[OK] Denoiser created")
    print(f"  Parameters: {sum(p.numel() for p in denoiser.parameters()):,}")
    
    diffusion = DiscreteDiffusion(
        num_entities=num_entities,
        num_relations=num_relations,
        num_timesteps=100
    )
    print(f"[OK] Diffusion process created")
    
except Exception as e:
    print(f"[FAIL] Model creation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("Testing forward pass...")
print("="*60)

try:
    from kg_path_diffusion import KGPathDiffusionModel
    
    model = KGPathDiffusionModel(
        num_entities=num_entities,
        num_relations=num_relations,
        hidden_dim=hidden_dim,
        graph_encoder_type="hybrid",
        num_graph_layers=2,
        num_diffusion_layers=2,
        num_heads=4,
        num_diffusion_steps=100,
        max_path_length=10
    )
    print(f"[OK] Full model created")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Move to CPU for testing
    model.eval()
    
    # Create dummy path mask
    path_mask = batch['path_entities'] != 0  # Non-padding positions
    
    with torch.no_grad():
        outputs = model(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            graph_batch=batch['graph_batch'],
            target_entities=batch['path_entities'],
            target_relations=batch['path_relations'],
            path_mask=path_mask
        )
    
    print(f"[OK] Forward pass successful!")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Entity loss: {outputs['entity_loss'].item():.4f}")
    print(f"  Relation loss: {outputs['relation_loss'].item():.4f}")
    
except Exception as e:
    print(f"[FAIL] Forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("[OK] ALL TESTS PASSED!")
print("="*60)
print("\nYou can now run full training with:")
print("python train.py --train_data ../Data/webqsp_combined/train.jsonl --val_data ../Data/webqsp_combined/val.jsonl --batch_size 8 --hidden_dim 256")

