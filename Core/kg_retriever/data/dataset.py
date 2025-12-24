"""
Dataset for KG-Conditioned Diffusion Retriever.

Efficiently loads and pre-processes KG triples for each question.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import random
import ast
from pathlib import Path
from transformers import AutoTokenizer
import pytorch_lightning as pl


class KGRetrieverDataset(Dataset):
    """
    Dataset for KG-conditioned path retrieval.
    
    Each sample contains:
    - Question text
    - FULL KG triples (efficient handling with dynamic padding)
    - Ground truth relation path
    """
    
    def __init__(
        self,
        data_path: Union[str, List[str]],
        vocab_path: Optional[str] = None,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_question_length: int = 64,
        max_triples: int = 0,  # 0 = use full KG, >0 = limit to max_triples
        max_path_length: int = 8,
        num_entity_buckets: int = 50000,  # Larger for full KG
        training: bool = True,
    ):
        super().__init__()
        self.max_question_length = max_question_length
        self.max_triples = max_triples  # 0 means no limit (full KG)
        self.max_path_length = max_path_length
        self.num_entity_buckets = num_entity_buckets
        self.training = training
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load vocabulary
        if vocab_path:
            self.relation_to_idx = self._load_vocab(vocab_path)
        else:
            self.relation_to_idx = {}
        
        # Special token indices
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.MASK_IDX = 2
        self.RELATION_OFFSET = 3  # Real relations start at index 3
        
        # Load data
        self.data = self._load_data(data_path)
        
        # Build relation vocab if not provided
        if not self.relation_to_idx:
            self._build_relation_vocab()
        
        # Pre-compute max KG size for padding
        self._max_kg_size = max(len(s['graph']) for s in self.data) if self.data else 10000
        if self.max_triples > 0:
            self._max_kg_size = min(self._max_kg_size, self.max_triples)
        
        print(f"Loaded {len(self.data)} samples, {len(self.relation_to_idx)} relations, max KG size: {self._max_kg_size}")
    
    def _load_vocab(self, vocab_path: str) -> Dict[str, int]:
        """Load relation vocabulary from JSON file."""
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        # Extract relation vocabulary
        if 'relation_to_idx' in vocab_data:
            return vocab_data['relation_to_idx']
        elif 'relations' in vocab_data:
            # Convert list to dict
            return {r: i + self.RELATION_OFFSET for i, r in enumerate(vocab_data['relations'])}
        else:
            return {}
    
    def _load_data(self, data_path: Union[str, List[str]]) -> List[Dict]:
        """Load data from parquet file(s) with pickle caching."""
        import pickle
        import hashlib
        
        if isinstance(data_path, str):
            data_path = [data_path]
        
        # Generate cache key from file paths
        cache_key = hashlib.md5("_".join(sorted(data_path)).encode()).hexdigest()[:12]
        cache_dir = Path(data_path[0]).parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"parsed_data_{cache_key}.pkl"
        
        # Try loading from cache
        if cache_file.exists():
            print(f"Loading from cache: {cache_file}", flush=True)
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"Loaded {len(cached_data)} samples from cache", flush=True)
                return cached_data
            except Exception as e:
                print(f"Cache load failed: {e}, rebuilding...", flush=True)
        
        # Parse data from parquet files
        all_data = []
        for path in data_path:
            print(f"Processing: {path}", flush=True)
            if path.endswith('.parquet'):
                df = pd.read_parquet(path)
            else:
                df = pd.read_json(path, lines=True)
            
            print(f"Loaded df with {len(df)} rows", flush=True)
            
            # Use to_dict('records') - much faster than iterrows()
            rows_list = df.to_dict('records')
            rows_added = 0
            for row in rows_list:
                sample = self._parse_row(row)
                if sample is not None:
                    all_data.append(sample)
                    rows_added += 1
            print(f"Added {rows_added} samples from {path}", flush=True)
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(all_data, f)
            print(f"Saved cache to: {cache_file}", flush=True)
        except Exception as e:
            print(f"Cache save failed: {e}", flush=True)
        
        return all_data
    
    def _parse_row(self, row) -> Optional[Dict]:
        """Parse a single data row."""
        try:
            # Parse graph
            graph_data = row['graph']
            if isinstance(graph_data, str):
                try:
                    graph = json.loads(graph_data)
                except json.JSONDecodeError:
                    try:
                        graph = ast.literal_eval(graph_data)
                    except:
                        # print(f"Failed to parse graph for {row.get('id', 'unknown')}")
                        return None
            else:
                graph = graph_data
            
            if not graph or len(graph) == 0:
                return None
            
            # Parse ground truth paths
            # Try multiple column names
            gt_paths_data = None
            for col in ['shortest_gt_paths', 'ground_truth_paths', 'paths']:
                if col in row and row[col] is not None:
                    gt_paths_data = row[col]
                    break
            
            if gt_paths_data is None:
                return None

            if isinstance(gt_paths_data, str):
                try:
                    gt_paths = json.loads(gt_paths_data)
                except json.JSONDecodeError:
                    try:
                        gt_paths = ast.literal_eval(gt_paths_data)
                    except:
                        return None
            else:
                gt_paths = gt_paths_data
            
            if not gt_paths:
                return None
            
            # Extract q_entity (optional, for backward compat)
            q_entity_data = row.get('q_entity', '[]')
            if isinstance(q_entity_data, str):
                try:
                    q_entities = json.loads(q_entity_data)
                except:
                    q_entities = [q_entity_data] if q_entity_data else []
            else:
                q_entities = q_entity_data if q_entity_data else []
            
            if isinstance(q_entities, str):
                q_entities = [q_entities]
            
            return {
                'id': row['id'],
                'question': row['question'],
                'graph': graph,
                'gt_paths': gt_paths,
                'q_entities': q_entities,
            }
        except Exception as e:
            # Print error for first few failures to debug
            if random.random() < 0.001: 
                print(f"Error parsing row {row.get('id', 'unknown')}: {e}")
            return None
    
    def _build_relation_vocab(self):
        """Build relation vocabulary from dataset."""
        relations = set()
        for sample in self.data:
            # From graph
            for triple in sample['graph']:
                if len(triple) >= 3:
                    relations.add(triple[1])
            # From paths
            for path in sample['gt_paths']:
                if 'relations' in path:
                    relations.update(path['relations'])
        
        self.relation_to_idx = {
            r: i + self.RELATION_OFFSET 
            for i, r in enumerate(sorted(relations))
        }
    
    def get_relation_idx(self, relation: str) -> int:
        """Get index for a relation."""
        return self.relation_to_idx.get(relation, self.UNK_IDX)
    
    def hash_entity(self, entity: str) -> int:
        """Hash entity string to bucket index (1-indexed, 0 is PAD)."""
        return (hash(entity) % self.num_entity_buckets) + 1
    
    @property
    def num_relations(self) -> int:
        """Total number of relations including special tokens."""
        return len(self.relation_to_idx) + self.RELATION_OFFSET
    
    def _encode_triples(
        self,
        triples: List[List[str]],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode ALL triples into tensor format with dynamic padding.
        
        Returns:
            kg_relation_ids: [actual_num_triples]
            kg_head_hash_ids: [actual_num_triples]
            kg_tail_hash_ids: [actual_num_triples]
            kg_triple_mask: [actual_num_triples]
        """
        # Limit if max_triples is set
        if self.max_triples > 0:
            triples = triples[:self.max_triples]
        
        num_triples = len(triples)
        
        # Pre-allocate tensors
        relation_ids = torch.zeros(num_triples, dtype=torch.long)
        head_hash_ids = torch.zeros(num_triples, dtype=torch.long)
        tail_hash_ids = torch.zeros(num_triples, dtype=torch.long)
        triple_mask = torch.ones(num_triples, dtype=torch.float)
        
        for i, triple in enumerate(triples):
            head, rel, tail = triple[0], triple[1], triple[2]
            relation_ids[i] = self.get_relation_idx(rel)
            head_hash_ids[i] = self.hash_entity(head)
            tail_hash_ids[i] = self.hash_entity(tail)
        
        return {
            'kg_relation_ids': relation_ids,
            'kg_head_hash_ids': head_hash_ids,
            'kg_tail_hash_ids': tail_hash_ids,
            'kg_triple_mask': triple_mask,
            'num_triples': num_triples,
        }

    
    def _encode_path(self, path: Dict) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encode a single reasoning path.
        
        Returns:
            target_relations: [max_path_length]
            path_mask: [max_path_length]
            path_length: int
        """
        relations = path.get('relations', [])
        path_length = min(len(relations), self.max_path_length)
        
        target_relations = torch.full((self.max_path_length,), self.PAD_IDX, dtype=torch.long)
        path_mask = torch.zeros(self.max_path_length, dtype=torch.float)
        
        for i, rel in enumerate(relations[:path_length]):
            target_relations[i] = self.get_relation_idx(rel)
            path_mask[i] = 1.0
        
        return target_relations, path_mask, path_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        # Tokenize question
        question_encoding = self.tokenizer(
            sample['question'],
            max_length=self.max_question_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Encode FULL KG triples (no sampling)
        triple_encoding = self._encode_triples(sample['graph'])
        
        # Select and encode ground truth path
        if self.training and len(sample['gt_paths']) > 1:
            # Random path during training
            gt_path = random.choice(sample['gt_paths'])
        else:
            gt_path = sample['gt_paths'][0]
        
        target_relations, path_mask, path_length = self._encode_path(gt_path)
        
        return {
            'id': sample['id'],
            'question_input_ids': question_encoding['input_ids'].squeeze(0),
            'question_attention_mask': question_encoding['attention_mask'].squeeze(0),
            **triple_encoding,
            'target_relations': target_relations,
            'path_mask': path_mask,
            'path_length': torch.tensor(path_length, dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length KG batching.
    Pads KG triples to the max size in the batch.
    """
    # Find max KG size in this batch
    max_triples = max(b['num_triples'] for b in batch)
    batch_size = len(batch)
    
    # Pre-allocate padded tensors
    kg_relation_ids = torch.zeros(batch_size, max_triples, dtype=torch.long)
    kg_head_hash_ids = torch.zeros(batch_size, max_triples, dtype=torch.long)
    kg_tail_hash_ids = torch.zeros(batch_size, max_triples, dtype=torch.long)
    kg_triple_mask = torch.zeros(batch_size, max_triples, dtype=torch.float)
    
    # Stack fixed-size tensors
    question_input_ids = torch.stack([b['question_input_ids'] for b in batch])
    question_attention_mask = torch.stack([b['question_attention_mask'] for b in batch])
    target_relations = torch.stack([b['target_relations'] for b in batch])
    path_mask = torch.stack([b['path_mask'] for b in batch])
    path_length = torch.stack([b['path_length'] for b in batch])
    
    # Copy variable-length KG data with padding
    for i, b in enumerate(batch):
        num = b['num_triples']
        kg_relation_ids[i, :num] = b['kg_relation_ids']
        kg_head_hash_ids[i, :num] = b['kg_head_hash_ids']
        kg_tail_hash_ids[i, :num] = b['kg_tail_hash_ids']
        kg_triple_mask[i, :num] = b['kg_triple_mask']
    
    return {
        'id': [b['id'] for b in batch],
        'question_input_ids': question_input_ids,
        'question_attention_mask': question_attention_mask,
        'kg_relation_ids': kg_relation_ids,
        'kg_head_hash_ids': kg_head_hash_ids,
        'kg_tail_hash_ids': kg_tail_hash_ids,
        'kg_triple_mask': kg_triple_mask,
        'target_relations': target_relations,
        'path_mask': path_mask,
        'path_length': path_length,
    }




class KGRetrieverDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for KG Retriever."""
    
    def __init__(
        self,
        train_path: Union[str, List[str]],
        val_path: Optional[Union[str, List[str]]] = None,
        test_path: Optional[Union[str, List[str]]] = None,
        vocab_path: Optional[str] = None,
        batch_size: int = 8,  # Smaller for full KG
        num_workers: int = 4,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_question_length: int = 64,
        max_triples: int = 0,  # 0 = full KG
        max_path_length: int = 8,
        num_entity_buckets: int = 50000,  # Larger for full KG
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.vocab_path = vocab_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.max_question_length = max_question_length
        self.max_triples = max_triples
        self.max_path_length = max_path_length
        self.num_entity_buckets = num_entity_buckets
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = KGRetrieverDataset(
                data_path=self.train_path,
                vocab_path=self.vocab_path,
                tokenizer_name=self.tokenizer_name,
                max_question_length=self.max_question_length,
                max_triples=self.max_triples,
                max_path_length=self.max_path_length,
                num_entity_buckets=self.num_entity_buckets,
                training=True,
            )
            
            # Save vocab for inference
            self.relation_to_idx = self.train_dataset.relation_to_idx
            self.num_relations = self.train_dataset.num_relations
            
            if self.val_path:
                self.val_dataset = KGRetrieverDataset(
                    data_path=self.val_path,
                    vocab_path=self.vocab_path,
                    tokenizer_name=self.tokenizer_name,
                    max_question_length=self.max_question_length,
                    max_triples=self.max_triples,
                    max_path_length=self.max_path_length,
                    num_entity_buckets=self.num_entity_buckets,
                    training=False,
                )
                # Share vocab
                self.val_dataset.relation_to_idx = self.train_dataset.relation_to_idx
        
        if stage == 'test' or stage is None:
            if self.test_path:
                self.test_dataset = KGRetrieverDataset(
                    data_path=self.test_path,
                    vocab_path=self.vocab_path,
                    tokenizer_name=self.tokenizer_name,
                    max_question_length=self.max_question_length,
                    max_triples=self.max_triples,
                    max_path_length=self.max_path_length,
                    num_entity_buckets=self.num_entity_buckets,
                    training=False,
                )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


if __name__ == '__main__':
    # Test the dataset
    dataset = KGRetrieverDataset(
        data_path='../Data/webqsp_final/shortest_paths/train.parquet',
        max_triples=500,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Num relations: {dataset.num_relations}")
    
    sample = dataset[0]
    print("\nSample keys:", sample.keys())
    print(f"Question shape: {sample['question_input_ids'].shape}")
    print(f"KG relation shape: {sample['kg_relation_ids'].shape}")
    print(f"Target relations shape: {sample['target_relations'].shape}")
    print(f"Path length: {sample['path_length']}")
