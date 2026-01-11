"""
Question Classifier Dataset for KGQA - 3-Class Version

Three classes:
- one_hop: Questions with single relation path (1 hop)
- multi_hop: Questions with 2+ relation paths
- numeric: Questions with numeric answers (detected from answer entities)

Uses both CWQ and WebQSP with their augmented versions.
Labels derived from both relation paths AND answer entities for accuracy.
"""

import os
import re
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import pytorch_lightning as pl
except ImportError:
    class _LightningDataModule:
        def __init__(self):
            pass
    class _PL:
        LightningDataModule = _LightningDataModule
    pl = _PL()


# 4-class label mapping
LABEL_MAP = {
    'one_hop': 0,
    'multi_hop': 1,
    'one_hop_numeric': 2,
    'multi_hop_numeric': 3,
}

LABEL_NAMES = ['one_hop', 'multi_hop', 'one_hop_numeric', 'multi_hop_numeric']
NUM_CLASSES = 4


def is_numeric_answer(row: pd.Series) -> bool:
    """
    Check if the answer is numeric using answer entities.
    
    This is more reliable than keyword matching in questions.
    Checks both 'answer' and 'a_entity' fields.
    Now more permissive to catch '50 states' etc.
    """
    # Check a_entity (answer entities)
    if 'a_entity' in row:
        a_ent = row['a_entity']
        
        # Handle list of entities
        if isinstance(a_ent, (list, np.ndarray)) and len(a_ent) > 0:
            val = str(a_ent[0]).strip()
            # Starts with digit (e.g., '123', '50 states', '2003-04')
            if re.match(r'^[\d\.,\$€£¥]+', val):
                return True
                
        # Handle JSON string
        elif isinstance(a_ent, str):
            try:
                parsed = json.loads(a_ent.replace("'", '"'))
                if isinstance(parsed, list) and len(parsed) > 0:
                    val = str(parsed[0]).strip()
                    if re.match(r'^[\d\.,\$€£¥]+', val):
                        return True
            except:
                pass
    
    # Check answer field as fallback
    if 'answer' in row:
        ans = row['answer']
        
        # Normalize to list
        if isinstance(ans, str):
            try:
                ans = json.loads(ans.replace("'", '"'))
            except:
                ans = [ans]
        
        if isinstance(ans, (list, np.ndarray)) and len(ans) > 0:
            val = str(ans[0]).strip()
            if re.match(r'^[\d\.,\$€£¥]+', val):
                return True
    
    return False


def get_path_hops_from_shortest_paths(shortest_gt_paths_str: str) -> Optional[int]:
    """Extract hop count from shortest_gt_paths JSON string."""
    try:
        if not shortest_gt_paths_str or shortest_gt_paths_str == '[]':
            return None
        
        # Handle different JSON formats
        paths_str = shortest_gt_paths_str.replace("'", '"')
        
        # Try to parse
        try:
            paths = json.loads(paths_str)
        except:
            return None
        
        if not paths or len(paths) == 0:
            return None
        
        path = paths[0]
        
        # Count relations in the path
        if isinstance(path, dict):
            if 'relations' in path:
                return len(path['relations'])
            elif 'full_path' in path:
                # Count --> in the path 
                hop_count = path['full_path'].count('-->')
                # Handle edge case: if path has no '-->' but has entities, it's 0-hop (direct)
                # But we should have at least 1 hop for valid paths
                return max(1, hop_count) if hop_count >= 0 else None
        
        return None
    except Exception as e:
        return None


def get_path_hops_from_gt_paths(gt_paths: List) -> Optional[int]:
    """Extract hop count from gt_paths list (fallback)."""
    if len(gt_paths) == 0:
        return None
    
    first_path = gt_paths[0]
    path_len = len(first_path) if hasattr(first_path, '__len__') else 0
    
    if path_len == 0:
        return None
    
    return path_len


def extract_label(row: pd.Series, max_hops: int = 5) -> Optional[int]:
    """
    Extract 4-class label from a data row using BOTH:
    1. Relation paths (for hop count)
    2. Answer entities (for numeric detection)
    
    Logic:
            | Numeric           | Non-Numeric
    --------|-------------------|-------------
    1 Hop   | one_hop_numeric   | one_hop
    >1 Hop  | multi_hop_numeric | multi_hop
    
    Returns:
        Label index or None if invalid/outlier
    """
    # 1. Determine Hop Count
    hops = None
    if 'shortest_gt_paths' in row and row['shortest_gt_paths']:
        hops = get_path_hops_from_shortest_paths(str(row['shortest_gt_paths']))
    
    # Fall back to gt_paths
    if hops is None and 'gt_paths' in row:
        gt_paths = row['gt_paths']
        # Handle NaN or non-list values
        if isinstance(gt_paths, (list, np.ndarray)) and len(gt_paths) > 0:
            hops = get_path_hops_from_gt_paths(gt_paths)
    
    # Filter invalid/outlier samples
    if hops is None or hops == 0 or hops > max_hops:
        return None
        
    # 2. Determine Metric/Numeric
    is_numeric = is_numeric_answer(row)
    
    # 3. Assign Label
    if hops == 1:
        return LABEL_MAP['one_hop_numeric'] if is_numeric else LABEL_MAP['one_hop']
    else:
        return LABEL_MAP['multi_hop_numeric'] if is_numeric else LABEL_MAP['multi_hop']


class QuestionClassifierDataset(Dataset):
    """Dataset for 3-class question type classification with entity-aware input."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        max_hops: int = 5,
        use_entities: bool = True,  # Whether to include entity info in input
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_hops = max_hops
        self.use_entities = use_entities
        
        # Extract labels and filter invalid samples
        self.samples = []
        self.entities = []  # Store entity info
        self.labels = []
        
        for idx, row in data.iterrows():
            # Use LLM label if available
            label = None
            if 'llm_label' in row and row['llm_label']:
                llm_lbl = row['llm_label']
                if llm_lbl in LABEL_MAP:
                    label = LABEL_MAP[llm_lbl]
            
            # Fallback to heuristic
            if label is None:
                label = extract_label(row, max_hops)
                
            if label is not None:
                self.samples.append(row['question'])
                
                # Extract entity info (q_entity)
                q_entity = ""
                if 'q_entity' in row:
                    q_ent = row['q_entity']
                    if isinstance(q_ent, list) and len(q_ent) > 0:
                        q_entity = q_ent[0] if isinstance(q_ent[0], str) else str(q_ent[0])
                    elif isinstance(q_ent, str):
                        # Try parsing as JSON list
                        try:
                            parsed = json.loads(q_ent.replace("'", '"'))
                            if isinstance(parsed, list) and len(parsed) > 0:
                                q_entity = parsed[0]
                        except:
                            q_entity = q_ent
                
                self.entities.append(q_entity)
                self.labels.append(label)
        
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        # Calculate class weights for balanced training (CRITICAL for imbalanced numeric class)
        label_counts = torch.bincount(self.labels, minlength=NUM_CLASSES)
        total = len(self.labels)
        self.class_weights = total / (NUM_CLASSES * label_counts.float() + 1e-6)
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        question = self.samples[idx]
        entity = self.entities[idx]
        label = self.labels[idx]
        
        # Create input text with entity if available
        if self.use_entities and entity:
            # Format: "[question] [SEP] [entity]"
            input_text = question
            text_pair = entity
        else:
            input_text = question
            text_pair = None
        
        encoding = self.tokenizer(
            input_text,
            text_pair,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for weighted sampling (crucial for numeric class)."""
        return self.class_weights[self.labels]


def load_all_datasets(
    preprocessed_dir: str = '/data/Yanlai/KGLLM/Data/preprocessed_paths',
    shortest_paths_cwq: str = '/data/Yanlai/KGLLM/Data/CWQ/shortest_paths',
    shortest_paths_webqsp: str = '/data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths',
    include_augmented: bool = True,
    prefer_shortest_paths: bool = True,
    use_llm_corrected: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all datasets and combine them.
    
    Uses both CWQ and WebQSP with their augmented versions.
    Prefers shortest_paths data if available (has cleaner path info),
    falls back to preprocessed_paths.
    
    If use_llm_corrected is True, attempts to load train_llm_corrected.parquet.
    """
    splits = {'train': [], 'val': [], 'test': []}
    
    # Check for LLM corrected training data
    if use_llm_corrected:
        llm_file = Path('train_llm_corrected.parquet')
        if llm_file.exists():
            print(f"Loading LLM corrected training data from {llm_file}")
            train_df = pd.read_parquet(llm_file)
            
            # Load validations/tests normally
            # (We need to load them separately as we're overriding train)
            # Re-use logic but skip train splitting
            pass 
        else:
            print(f"Warning: use_llm_corrected=True but {llm_file} not found. Falling back to standard loading.")
            use_llm_corrected = False

    # Load WebQSP shortest_paths (small enough to load quickly)
    webqsp_sp = Path(shortest_paths_webqsp)
    if prefer_shortest_paths and webqsp_sp.exists():
        for split in splits.keys():
            # If using LLM corrected, skip train loading
            if use_llm_corrected and split == 'train':
                continue
                
            sp_file = webqsp_sp / f'{split}.parquet'
            if sp_file.exists():
                try:
                    df = pd.read_parquet(sp_file)
                    df['source'] = 'webqsp_sp'
                    splits[split].append(df)
                    print(f"Loaded {sp_file.name}: {len(df)} samples (shortest_paths)")
                except Exception as e:
                    print(f"Error loading {sp_file}: {e}")
    
    # Load preprocessed paths (includes augmented versions)
    pp_dir = Path(preprocessed_dir)
    datasets = ['cwq', 'webqsp']
    
    for dataset in datasets:
        for split in splits.keys():
            # Skip train if using LLM corrected
            if use_llm_corrected and split == 'train':
                continue
                
            # Skip webqsp if we already have shortest_paths
            if dataset == 'webqsp' and prefer_shortest_paths and webqsp_sp.exists():
                continue
            
            # Original files
            orig_file = pp_dir / f'{dataset}_{split}.parquet'
            if orig_file.exists():
                df = pd.read_parquet(orig_file)
                df['source'] = dataset
                splits[split].append(df)
                print(f"Loaded {orig_file.name}: {len(df)} samples")
            
            # Augmented files
            if include_augmented:
                aug_file = pp_dir / f'{dataset}_{split}_augmented.parquet'
                if aug_file.exists():
                    df = pd.read_parquet(aug_file)
                    df['source'] = f'{dataset}_aug'
                    splits[split].append(df)
                    print(f"Loaded {aug_file.name}: {len(df)} samples")
    
    # Combine
    val_df = pd.concat(splits['val'], ignore_index=True) if splits['val'] else pd.DataFrame()
    test_df = pd.concat(splits['test'], ignore_index=True) if splits['test'] else pd.DataFrame()
    
    if not use_llm_corrected:
        train_df = pd.concat(splits['train'], ignore_index=True) if splits['train'] else pd.DataFrame()
    
    print(f"\nCombined: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


class QuestionClassifierDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for 3-class question classification."""
    
    def __init__(
        self,
        data_dir: str = '/data/Yanlai/KGLLM/Data/preprocessed_paths',
        tokenizer_name: str = 'huawei-noah/TinyBERT_General_6L_768D',
        max_length: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
        max_hops: int = 5,
        include_augmented: bool = True,
        use_weighted_sampling: bool = True,
        use_entities: bool = True,  # Entity-aware input
        use_llm_corrected: bool = True,  # Use LLM corrected labels
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_hops = max_hops
        self.include_augmented = include_augmented
        self.use_weighted_sampling = use_weighted_sampling
        self.use_entities = use_entities
        self.use_llm_corrected = use_llm_corrected
        
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None
        
    def setup(self, stage: Optional[str] = None):
        """Load data and create datasets."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        # Load all datasets (CWQ + WebQSP + augmented)
        train_df, val_df, test_df = load_all_datasets(
            preprocessed_dir=self.data_dir,
            include_augmented=self.include_augmented,
            use_llm_corrected=self.use_llm_corrected,
        )
        
        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = QuestionClassifierDataset(
                train_df, self.tokenizer, self.max_length, self.max_hops,
                use_entities=self.use_entities
            )
            self.val_dataset = QuestionClassifierDataset(
                val_df, self.tokenizer, self.max_length, self.max_hops,
                use_entities=self.use_entities
            )
            self.class_weights = self.train_dataset.class_weights
            
            # Print class distribution
            print("\nTraining set class distribution:")
            for i, name in enumerate(LABEL_NAMES):
                count = (self.train_dataset.labels == i).sum().item()
                pct = 100 * count / len(self.train_dataset)
                print(f"  {name}: {count} ({pct:.1f}%)")
            
            print(f"\nClass weights for balanced sampling:")
            for i, name in enumerate(LABEL_NAMES):
                print(f"  {name}: {self.class_weights[i]:.3f}")
        
        if stage == 'test' or stage is None:
            self.test_dataset = QuestionClassifierDataset(
                test_df, self.tokenizer, self.max_length, self.max_hops,
                use_entities=self.use_entities
            )
    
    def train_dataloader(self) -> DataLoader:
        if self.use_weighted_sampling:
            # CRITICAL: Weighted sampling ensures numeric class is seen frequently
            sampler = WeightedRandomSampler(
                weights=self.train_dataset.get_sample_weights(),
                num_samples=len(self.train_dataset),
                replacement=True
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == '__main__':
    # Test the dataset
    print("Testing QuestionClassifierDataset (3-class with answer entity labels)...")
    
    # Load and check data
    train_df, val_df, test_df = load_all_datasets()
    
    # Test label extraction
    print("\nLabel extraction test:")
    sample = train_df.iloc[0]
    label = extract_label(sample)
    print(f"Q: {sample['question'][:80]}...")
    print(f"Label: {LABEL_NAMES[label] if label is not None else 'None'}")
    
    # Count distribution
    label_counts = {0: 0, 1: 0, 2: 0}
    for idx, row in train_df.iterrows():
        lbl = extract_label(row)
        if lbl is not None:
            label_counts[lbl] += 1
    
    total = sum(label_counts.values())
    print(f"\nTrain distribution:")
    print(f"  one_hop: {label_counts[0]} ({100*label_counts[0]/total:.1f}%)")
    print(f"  multi_hop: {label_counts[1]} ({100*label_counts[1]/total:.1f}%)")
    print(f"  numeric: {label_counts[2]} ({100*label_counts[2]/total:.1f}%)")
    print("Dataset test passed!")
