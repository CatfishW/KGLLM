"""
Test script to verify reorganization works correctly.
Tests imports, model creation, and basic functionality.
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add Core directory to path
core_dir = Path(__file__).parent
sys.path.insert(0, str(core_dir))

def test_imports():
    """Test that all imports work correctly."""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test model imports
    try:
        from models.base import QuestionEncoder
        print("[OK] models.base.QuestionEncoder")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] models.base.QuestionEncoder: {e}")
        tests_failed += 1
    
    try:
        from models.diffusion import KGPathDiffusionModel, KGPathDiffusionLightning
        print("[OK] models.diffusion")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] models.diffusion: {e}")
        tests_failed += 1
    
    try:
        from models.autoregressive import KGPathAutoregressiveModel, KGPathAutoregressiveLightning
        print("[OK] models.autoregressive")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] models.autoregressive: {e}")
        tests_failed += 1
    
    try:
        from models.gnn_decoder import KGPathGNNDecoderModel, KGPathGNNDecoderLightning
        print("[OK] models.gnn_decoder")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] models.gnn_decoder: {e}")
        tests_failed += 1
    
    try:
        from models.factory import create_model
        print("[OK] models.factory")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] models.factory: {e}")
        tests_failed += 1
    
    try:
        from models import (
            QuestionEncoder,
            KGPathDiffusionLightning,
            KGPathAutoregressiveLightning,
            KGPathGNNDecoderLightning,
            create_model
        )
        print("[OK] models package __init__")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] models package __init__: {e}")
        tests_failed += 1
    
    # Test training imports
    try:
        from training.callbacks.path_examples_logger import PathExamplesLogger
        print("[OK] training.callbacks.path_examples_logger")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] training.callbacks.path_examples_logger: {e}")
        tests_failed += 1
    
    try:
        from training import PathExamplesLogger
        print("[OK] training package __init__")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] training package __init__: {e}")
        tests_failed += 1
    
    # Test data imports
    try:
        from data.dataset import KGPathDataModule, KGPathDataset, EntityRelationVocab
        print("[OK] data.dataset")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] data.dataset: {e}")
        tests_failed += 1
    
    print(f"\nImports: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_model_creation():
    """Test that models can be created."""
    print("\n" + "=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        from models.factory import create_model
        
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
        
        # Test diffusion model
        try:
            config_diff = config.copy()
            config_diff['num_diffusion_layers'] = 2
            config_diff['num_diffusion_steps'] = 100
            config_diff['use_entity_embeddings'] = True
            config_diff['predict_entities'] = False
            
            model = create_model('diffusion', 1000, 500, config_diff)
            assert model is not None
            print("[OK] Diffusion model creation")
            tests_passed += 1
        except Exception as e:
            print(f"[FAIL] Diffusion model creation: {e}")
            tests_failed += 1
        
        # Test autoregressive model
        try:
            model = create_model('autoregressive', 1000, 500, config)
            assert model is not None
            print("[OK] Autoregressive model creation")
            tests_passed += 1
        except Exception as e:
            print(f"[FAIL] Autoregressive model creation: {e}")
            tests_failed += 1
        
        # Test GNN decoder model
        try:
            config_gnn = config.copy()
            config_gnn['gnn_type'] = 'gat'
            config_gnn['gnn_layers'] = 2
            config_gnn['gnn_heads'] = 4
            config_gnn['decoder_layers'] = 2
            config_gnn['decoder_heads'] = 4
            config_gnn['use_graph_structure'] = True
            
            model = create_model('gnn_decoder', 1000, 500, config_gnn)
            assert model is not None
            print("[OK] GNN decoder model creation")
            tests_passed += 1
        except Exception as e:
            print(f"[FAIL] GNN decoder model creation: {e}")
            tests_failed += 1
        
        # Test invalid model type
        try:
            model = create_model('invalid_type', 1000, 500, config)
            print("[FAIL] Invalid model type should raise error")
            tests_failed += 1
        except ValueError:
            print("[OK] Invalid model type correctly raises ValueError")
            tests_passed += 1
        except Exception as e:
            print(f"[FAIL] Invalid model type raised wrong exception: {e}")
            tests_failed += 1
            
    except Exception as e:
        print(f"[FAIL] Model creation test setup failed: {e}")
        tests_failed += 1
    
    print(f"\nModel Creation: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_model_forward():
    """Test that models can perform forward pass."""
    print("\n" + "=" * 60)
    print("Testing Model Forward Pass")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        import torch
        from models.factory import create_model
        
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
        
        batch_size = 2
        seq_len = 5
        q_len = 10
        
        batch = {
            'question_input_ids': torch.randint(0, 1000, (batch_size, q_len)),
            'question_attention_mask': torch.ones(batch_size, q_len, dtype=torch.long),
            'path_relations': torch.randint(0, 500, (batch_size, seq_len)),
            'path_lengths': torch.tensor([seq_len, seq_len])
        }
        
        # Test autoregressive model forward
        try:
            model = create_model('autoregressive', 1000, 500, config)
            output = model.forward_single(batch)
            assert 'loss' in output
            assert 'relation_loss' in output
            print("[OK] Autoregressive forward pass")
            tests_passed += 1
        except Exception as e:
            print(f"[FAIL] Autoregressive forward pass: {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
        
        # Test GNN decoder model forward
        try:
            config_gnn = config.copy()
            config_gnn['gnn_type'] = 'gat'
            config_gnn['gnn_layers'] = 2
            config_gnn['gnn_heads'] = 4
            config_gnn['decoder_layers'] = 2
            config_gnn['decoder_heads'] = 4
            config_gnn['use_graph_structure'] = True
            
            model = create_model('gnn_decoder', 1000, 500, config_gnn)
            output = model.forward_single(batch)
            assert 'loss' in output
            assert 'relation_loss' in output
            print("[OK] GNN decoder forward pass")
            tests_passed += 1
        except Exception as e:
            print(f"[FAIL] GNN decoder forward pass: {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
        
    except Exception as e:
        print(f"[FAIL] Forward pass test setup failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    print(f"\nForward Pass: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_train_script_imports():
    """Test that train.py can be imported and parsed."""
    print("\n" + "=" * 60)
    print("Testing Train Script")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        # Test that train.py can be imported
        import train
        print("[OK] train.py imports successfully")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] train.py import failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    try:
        # Test that parse_args function exists and works with minimal args
        from train import parse_args
        print("[OK] parse_args function exists")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] parse_args function: {e}")
        tests_failed += 1
    
    print(f"\nTrain Script: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_callback():
    """Test that callback can be instantiated."""
    print("\n" + "=" * 60)
    print("Testing Callback")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        from training.callbacks.path_examples_logger import PathExamplesLogger
        
        callback = PathExamplesLogger(
            num_examples=5,
            log_to_file=False,
            log_to_tensorboard=False
        )
        assert callback is not None
        print("[OK] PathExamplesLogger instantiation")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] PathExamplesLogger instantiation: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    print(f"\nCallback: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("REORGANIZATION TEST SUITE")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Model Forward Pass", test_model_forward()))
    results.append(("Train Script", test_train_script_imports()))
    results.append(("Callback", test_callback()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n[SUCCESS] All tests PASSED! Reorganization is successful.")
        return 0
    else:
        print("\n[WARNING] Some tests FAILED. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

