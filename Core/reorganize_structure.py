"""
Script to reorganize the project structure.

This script:
1. Moves files to appropriate directories
2. Updates imports in moved files
3. Creates proper __init__.py files
"""

import os
import shutil
from pathlib import Path
import re

# Define new structure
STRUCTURE = {
    'models/': [
        'kg_path_diffusion.py -> diffusion.py',
        'kg_path_autoregressive.py -> autoregressive.py',
        'kg_path_gnn_decoder.py -> gnn_decoder.py',
        'model_factory.py -> factory.py',
    ],
    'training/': [
        'train.py',
    ],
    'training/callbacks/': [
        'callbacks/path_examples_logger.py -> path_examples_logger.py',
    ],
    'utils/': [
        'inference.py',
        'evaluate_with_metrics.py',
        'prepare_combined_data.py',
        'test_new_models.py',
        'test_tokenizer_empty.py',
        'verify_fix_dataset.py',
        'verify_fix.py',
    ],
    'scripts/': [
        'run_train.sh',
        'run_train.bat',
        'run_inference.sh',
        'run_inference.bat',
        'run_evaluate.bat',
        'run_quick_test.sh',
        'run_quick_test.bat',
        'run_tensorboard.bat',
    ],
}

# Import replacements
IMPORT_REPLACEMENTS = {
    'from kg_path_diffusion import': 'from models.diffusion import',
    'from kg_path_autoregressive import': 'from models.autoregressive import',
    'from kg_path_gnn_decoder import': 'from models.gnn_decoder import',
    'from model_factory import': 'from models.factory import',
    'from callbacks.path_examples_logger import': 'from training.callbacks.path_examples_logger import',
    'from data.dataset import': 'from data.dataset import',  # Keep as is
    'from modules.diffusion import': 'from modules.diffusion import',  # Keep as is
}

def update_imports(file_path: Path, replacements: dict):
    """Update imports in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        for old_import, new_import in replacements.items():
            content = content.replace(old_import, new_import)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Updated imports in {file_path}")
            return True
        return False
    except Exception as e:
        print(f"  Error updating {file_path}: {e}")
        return False

def main():
    """Main reorganization function."""
    base_dir = Path(__file__).parent
    
    print("Reorganizing project structure...")
    print("=" * 60)
    
    # Create directories
    for dir_path in STRUCTURE.keys():
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Move files
    for target_dir, files in STRUCTURE.items():
        target_path = base_dir / target_dir
        for file_spec in files:
            if ' -> ' in file_spec:
                source_file, target_file = file_spec.split(' -> ')
            else:
                source_file = target_file = file_spec
            
            source_path = base_dir / source_file
            target_path_file = target_path / target_file
            
            if source_path.exists():
                # Create target directory if needed
                target_path_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_path, target_path_file)
                print(f"Copied: {source_file} -> {target_dir}{target_file}")
                
                # Update imports
                update_imports(target_path_file, IMPORT_REPLACEMENTS)
            else:
                print(f"Warning: Source file not found: {source_file}")
    
    print("\n" + "=" * 60)
    print("Reorganization complete!")
    print("\nNext steps:")
    print("1. Update models/*.py to import QuestionEncoder from models.base")
    print("2. Create __init__.py files in new directories")
    print("3. Test imports")

if __name__ == '__main__':
    main()

