"""
Test script to verify new model implementations work correctly.
Tests both autoregressive and gnn_decoder models.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_autoregressive_model():
    """Test autoregressive model creation and forward pass."""
    print("Testing Autoregressive Model...")
    
    try:
        from kg_path_autoregressive import KGPathAutoregressiveLightning
        
        # Create model
        model = KGPathAutoregressiveLightning(
            num_entities=1000,
            num_relations=500,
            hidden_dim=128,
            num_layers=2,  # Small for testing
            num_heads=4,
            max_path_length=10,
            dropout=0.1,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=1000
        )
        
        print("  ✓ Model created successfully")
        
        # Test forward pass
        batch_size = 2
        seq_len = 5
        q_len = 10
        
        batch = {
            'question_input_ids': torch.randint(0, 1000, (batch_size, q_len)),
            'question_attention_mask': torch.ones(batch_size, q_len, dtype=torch.long),
            'path_relations': torch.randint(0, 500, (batch_size, seq_len)),
            'path_lengths': torch.tensor([seq_len, seq_len])
        }
        
        output = model.forward_single(batch)
        
        assert 'loss' in output
        assert 'relation_loss' in output
        assert output['loss'].requires_grad
        print("  ✓ Forward pass works")
        
        # Test multipath forward
        batch_multipath = {
            'question_input_ids': torch.randint(0, 1000, (batch_size, q_len)),
            'question_attention_mask': torch.ones(batch_size, q_len, dtype=torch.long),
            'all_path_relations': torch.randint(0, 500, (batch_size, 3, seq_len)),
            'all_path_lengths': torch.tensor([[seq_len, seq_len, seq_len], [seq_len, seq_len, seq_len]]),
            'num_paths': torch.tensor([3, 3])
        }
        
        output_multipath = model.forward_multipath(batch_multipath)
        
        assert 'loss' in output_multipath
        assert 'num_paths_avg' in output_multipath
        print("  ✓ Multipath forward pass works")
        
        print("  ✓ Autoregressive model test PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Autoregressive model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gnn_decoder_model():
    """Test GNN decoder model creation and forward pass."""
    print("Testing GNN Decoder Model...")
    
    try:
        from kg_path_gnn_decoder import KGPathGNNDecoderLightning
        
        # Create model
        model = KGPathGNNDecoderLightning(
            num_entities=1000,
            num_relations=500,
            hidden_dim=128,
            gnn_type='gat',
            gnn_layers=2,  # Small for testing
            gnn_heads=4,
            decoder_layers=2,
            decoder_heads=4,
            max_path_length=10,
            dropout=0.1,
            use_graph_structure=True,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=1000
        )
        
        print("  ✓ Model created successfully")
        
        # Test forward pass
        batch_size = 2
        seq_len = 5
        q_len = 10
        
        batch = {
            'question_input_ids': torch.randint(0, 1000, (batch_size, q_len)),
            'question_attention_mask': torch.ones(batch_size, q_len, dtype=torch.long),
            'path_relations': torch.randint(0, 500, (batch_size, seq_len)),
            'path_lengths': torch.tensor([seq_len, seq_len])
        }
        
        output = model.forward_single(batch)
        
        assert 'loss' in output
        assert 'relation_loss' in output
        assert output['loss'].requires_grad
        print("  ✓ Forward pass works")
        
        # Test multipath forward
        batch_multipath = {
            'question_input_ids': torch.randint(0, 1000, (batch_size, q_len)),
            'question_attention_mask': torch.ones(batch_size, q_len, dtype=torch.long),
            'all_path_relations': torch.randint(0, 500, (batch_size, 3, seq_len)),
            'all_path_lengths': torch.tensor([[seq_len, seq_len, seq_len], [seq_len, seq_len, seq_len]]),
            'num_paths': torch.tensor([3, 3])
        }
        
        output_multipath = model.forward_multipath(batch_multipath)
        
        assert 'loss' in output_multipath
        assert 'num_paths_avg' in output_multipath
        print("  ✓ Multipath forward pass works")
        
        print("  ✓ GNN Decoder model test PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ✗ GNN Decoder model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_factory():
    """Test model factory."""
    print("Testing Model Factory...")
    
    try:
        from model_factory import create_model
        
        config = {
            'hidden_dim': 128,
            'question_encoder': 'sentence-transformers/all-MiniLM-L6-v2',
            'freeze_question_encoder': False,
            'num_layers': 2,
            'num_heads': 4,
            'max_path_length': 10,
            'dropout': 0.1,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'max_steps': 1000,
            'augment_paths': True
        }
        
        # Test autoregressive
        model_ar = create_model('autoregressive', 1000, 500, config)
        assert model_ar is not None
        print("  ✓ Autoregressive model created via factory")
        
        # Test gnn_decoder
        config['gnn_type'] = 'gat'
        config['gnn_layers'] = 2
        config['gnn_heads'] = 4
        config['decoder_layers'] = 2
        config['decoder_heads'] = 4
        config['use_graph_structure'] = True
        
        model_gnn = create_model('gnn_decoder', 1000, 500, config)
        assert model_gnn is not None
        print("  ✓ GNN Decoder model created via factory")
        
        # Test diffusion (should still work)
        config['num_diffusion_layers'] = 2
        config['num_diffusion_steps'] = 100
        config['use_entity_embeddings'] = True
        config['predict_entities'] = False
        
        model_diff = create_model('diffusion', 1000, 500, config)
        assert model_diff is not None
        print("  ✓ Diffusion model created via factory")
        
        print("  ✓ Model factory test PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Model factory test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Testing New Model Implementations")
    print("="*60)
    print()
    
    results = []
    
    results.append(("Autoregressive Model", test_autoregressive_model()))
    results.append(("GNN Decoder Model", test_gnn_decoder_model()))
    results.append(("Model Factory", test_model_factory()))
    
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print("\n✗ Some tests FAILED!")
        return 1


if __name__ == '__main__':
    sys.exit(main())

