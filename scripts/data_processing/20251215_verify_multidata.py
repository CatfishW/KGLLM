
import sys
import os
sys.path.insert(0, "/data/Yanlai/KGLLM/Core")
from data.dataset import KGPathDataModule, EntityRelationVocab

def test_multidata():
    train_paths = [
        "/data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/train.parquet",
        "/data/Yanlai/KGLLM/Data/CWQ/shortest_paths/train.parquet"
    ]
    
    # Minimal init
    dm = KGPathDataModule(
        train_path=train_paths,
        batch_size=2,
        max_entities=1000,
        max_relations=100
    )
    
    print("Setting up data module...")
    dm.setup('fit')
    
    print(f"Train dataset length: {len(dm.train_dataset)}")
    
    # Check first and last sample to see if they look valid
    print("First sample ID:", dm.train_dataset.samples[0].get('id'))
    print("Last sample ID:", dm.train_dataset.samples[-1].get('id'))

if __name__ == "__main__":
    test_multidata()
