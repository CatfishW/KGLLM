"""
Simple import test that doesn't require PyTorch to be loaded.
Tests syntax and import structure only.
"""

import sys
import ast
from pathlib import Path

def test_syntax(file_path):
    """Test that a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def test_import_structure():
    """Test import structure without actually importing."""
    print("=" * 60)
    print("Testing Import Structure (Syntax Only)")
    print("=" * 60)
    
    core_dir = Path(__file__).parent
    tests_passed = 0
    tests_failed = 0
    
    # Test key files
    files_to_test = [
        'models/__init__.py',
        'models/base.py',
        'models/diffusion.py',
        'models/autoregressive.py',
        'models/gnn_decoder.py',
        'models/factory.py',
        'training/__init__.py',
        'training/callbacks/__init__.py',
        'training/callbacks/path_examples_logger.py',
        'train.py',
    ]
    
    for file_path in files_to_test:
        full_path = core_dir / file_path
        if full_path.exists():
            passed, error = test_syntax(full_path)
            if passed:
                print(f"[OK] {file_path}")
                tests_passed += 1
            else:
                print(f"[FAIL] {file_path}: {error}")
                tests_failed += 1
        else:
            print(f"[SKIP] {file_path} (not found)")
    
    print(f"\nSyntax Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0

def test_import_paths():
    """Test that import statements are correct."""
    print("\n" + "=" * 60)
    print("Testing Import Paths")
    print("=" * 60)
    
    core_dir = Path(__file__).parent
    tests_passed = 0
    tests_failed = 0
    
    # Check that models use relative imports
    try:
        with open(core_dir / 'models/diffusion.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from .base import' in content:
                print("[OK] models/diffusion.py uses relative import for base")
                tests_passed += 1
            else:
                print("[FAIL] models/diffusion.py should use 'from .base import'")
                tests_failed += 1
    except Exception as e:
        print(f"[FAIL] Could not check models/diffusion.py: {e}")
        tests_failed += 1
    
    # Check that autoregressive uses relative import
    try:
        with open(core_dir / 'models/autoregressive.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from .base import' in content:
                print("[OK] models/autoregressive.py uses relative import for base")
                tests_passed += 1
            else:
                print("[FAIL] models/autoregressive.py should use 'from .base import'")
                tests_failed += 1
    except Exception as e:
        print(f"[FAIL] Could not check models/autoregressive.py: {e}")
        tests_failed += 1
    
    # Check that gnn_decoder uses relative import
    try:
        with open(core_dir / 'models/gnn_decoder.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from .base import' in content:
                print("[OK] models/gnn_decoder.py uses relative import for base")
                tests_passed += 1
            else:
                print("[FAIL] models/gnn_decoder.py should use 'from .base import'")
                tests_failed += 1
    except Exception as e:
        print(f"[FAIL] Could not check models/gnn_decoder.py: {e}")
        tests_failed += 1
    
    # Check that factory uses relative imports
    try:
        with open(core_dir / 'models/factory.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from .diffusion import' in content and 'from .autoregressive import' in content:
                print("[OK] models/factory.py uses relative imports")
                tests_passed += 1
            else:
                print("[FAIL] models/factory.py should use relative imports")
                tests_failed += 1
    except Exception as e:
        print(f"[FAIL] Could not check models/factory.py: {e}")
        tests_failed += 1
    
    # Check that train.py uses new imports
    try:
        with open(core_dir / 'train.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from models.factory import' in content:
                print("[OK] train.py uses new model imports")
                tests_passed += 1
            else:
                print("[FAIL] train.py should use 'from models.factory import'")
                tests_failed += 1
    except Exception as e:
        print(f"[FAIL] Could not check train.py: {e}")
        tests_failed += 1
    
    print(f"\nImport Path Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("REORGANIZATION STRUCTURE TEST")
    print("=" * 60)
    print()
    
    results = []
    results.append(("Syntax", test_import_structure()))
    results.append(("Import Paths", test_import_paths()))
    
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
        print("\n[SUCCESS] All structure tests PASSED!")
        print("Note: Full functionality tests require PyTorch to be properly installed.")
        return 0
    else:
        print("\n[WARNING] Some structure tests FAILED.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

