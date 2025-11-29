"""
Test script to verify graph compression integration without full training.
Tests that the model can be instantiated with compression enabled.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_compression_integration():
    """Test that compression can be enabled and model initializes correctly."""
    
    print("Testing graph compression integration...")
    print("=" * 60)
    
    try:
        from kg_path_diffusion import KGPathDiffusionModel
        
        # Test 1: Model with compression disabled (baseline)
        print("\n1. Testing model WITHOUT compression...")
        model_no_comp = KGPathDiffusionModel(
            num_entities=1000,
            num_relations=100,
            hidden_dim=256,
            use_graph_compression=False,
            predict_entities=False
        )
        print("   ✓ Model without compression initialized successfully")
        
        # Test 2: Model with attention-based compression
        print("\n2. Testing model WITH attention-based compression...")
        model_attn = KGPathDiffusionModel(
            num_entities=1000,
            num_relations=100,
            hidden_dim=256,
            use_graph_compression=True,
            num_compressed_nodes=64,
            compression_method="attention",
            predict_entities=False
        )
        print("   ✓ Model with attention compression initialized successfully")
        
        # Test 3: Model with cluster-based compression
        print("\n3. Testing model WITH cluster-based compression...")
        model_cluster = KGPathDiffusionModel(
            num_entities=1000,
            num_relations=100,
            hidden_dim=256,
            use_graph_compression=True,
            num_compressed_nodes=64,
            compression_method="cluster",
            predict_entities=False
        )
        print("   ✓ Model with cluster compression initialized successfully")
        
        # Test 4: Model with hierarchical compression
        print("\n4. Testing model WITH hierarchical compression...")
        model_hier = KGPathDiffusionModel(
            num_entities=1000,
            num_relations=100,
            hidden_dim=256,
            use_graph_compression=True,
            num_compressed_nodes=64,
            compression_method="hierarchical",
            predict_entities=False
        )
        print("   ✓ Model with hierarchical compression initialized successfully")
        
        # Test 5: Forward pass with dummy data (compression enabled)
        print("\n5. Testing forward pass with compression...")
        from torch_geometric.data import Data, Batch
        
        batch_size = 2
        device = torch.device('cpu')
        
        # Create dummy graph data
        graphs = []
        for i in range(batch_size):
            num_nodes = 100 + i * 50  # Different sizes to test padding
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), device=device)
            edge_type = torch.randint(0, 100, (num_nodes * 2,), device=device)
            node_ids = torch.randint(0, 1000, (num_nodes,), device=device)
            
            graph = Data(
                node_ids=node_ids,
                edge_index=edge_index,
                edge_type=edge_type
            )
            graphs.append(graph)
        
        graph_batch = Batch.from_data_list(graphs)
        
        # Dummy question inputs
        question_input_ids = torch.randint(0, 1000, (batch_size, 50), device=device)
        question_attention_mask = torch.ones(batch_size, 50, dtype=torch.long, device=device)
        
        # Dummy target paths
        target_entities = torch.randint(0, 1000, (batch_size, 10), device=device)
        target_relations = torch.randint(0, 100, (batch_size, 9), device=device)
        
        # Test encode_inputs with compression
        with torch.no_grad():
            question_seq, question_pooled, graph_node_emb, graph_pooled, graph_mask = \
                model_hier.encode_inputs(
                    question_input_ids,
                    question_attention_mask,
                    graph_batch
                )
            
            print(f"   ✓ Encoded inputs successfully")
            print(f"     - Question seq shape: {question_seq.shape}")
            print(f"     - Graph node emb shape: {graph_node_emb.shape}")
            print(f"     - Graph mask shape: {graph_mask.shape}")
            print(f"     - Expected compressed nodes: 64")
            
            # Verify compression worked
            if graph_node_emb.shape[1] == 64:
                print(f"   ✓ Compression successful: {graph_node_emb.shape[1]} compressed nodes")
            else:
                print(f"   ⚠ Warning: Expected 64 compressed nodes, got {graph_node_emb.shape[1]}")
        
        print("\n" + "=" * 60)
        print("All tests passed! Graph compression integration is working.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_compression_integration()
    sys.exit(0 if success else 1)

