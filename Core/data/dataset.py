"""
Dataset classes for KG Path Generation from webqsp_rog format.
Optimized for fast parallel data loading.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import random
from pathlib import Path
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer
import pytorch_lightning as pl


@dataclass
class PathGenerationSample:
    """Single sample for path generation task."""
    id: str
    question: str
    answer: str
    q_entities: List[str]
    a_entities: List[str]
    graph_triples: List[Tuple[str, str, str]]  # (subj, rel, obj)
    target_paths: List[List[Tuple[str, str, str]]]  # List of paths, each path is list of triples


class EntityRelationVocab:
    """Vocabulary for entities and relations with efficient lookup."""
    
    def __init__(self, max_entities: int = 50000, max_relations: int = 5000):
        self.entity2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<MASK>": 4}
        self.idx2entity: Dict[int, str] = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>", 4: "<MASK>"}
        self.relation2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2}
        self.idx2relation: Dict[int, str] = {0: "<PAD>", 1: "<UNK>", 2: "<MASK>"}
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.entity_counts: Dict[str, int] = {}
        self.relation_counts: Dict[str, int] = {}
        
    def add_entity(self, entity: str) -> int:
        # Track count
        self.entity_counts[entity] = self.entity_counts.get(entity, 0) + 1
        
        if entity not in self.entity2idx:
            if len(self.entity2idx) < self.max_entities:
                idx = len(self.entity2idx)
                self.entity2idx[entity] = idx
                self.idx2entity[idx] = entity
            else:
                return self.entity2idx["<UNK>"]
        return self.entity2idx[entity]
    
    def add_relation(self, relation: str) -> int:
        # Track count
        self.relation_counts[relation] = self.relation_counts.get(relation, 0) + 1
        
        if relation not in self.relation2idx:
            if len(self.relation2idx) < self.max_relations:
                idx = len(self.relation2idx)
                self.relation2idx[relation] = idx
                self.idx2relation[idx] = relation
            else:
                return self.relation2idx["<UNK>"]
        return self.relation2idx[relation]
    
    def get_entity_idx(self, entity: str) -> int:
        return self.entity2idx.get(entity, self.entity2idx["<UNK>"])
    
    def get_relation_idx(self, relation: str) -> int:
        return self.relation2idx.get(relation, self.relation2idx["<UNK>"])
    
    @property
    def num_entities(self) -> int:
        return len(self.entity2idx)
    
    @property
    def num_relations(self) -> int:
        return len(self.relation2idx)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'entity2idx': self.entity2idx,
                'relation2idx': self.relation2idx
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'EntityRelationVocab':
        vocab = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        vocab.entity2idx = data['entity2idx']
        vocab.idx2entity = {int(v): k for k, v in data['entity2idx'].items()}
        vocab.relation2idx = data['relation2idx']
        vocab.idx2relation = {int(v): k for k, v in data['relation2idx'].items()}
        return vocab


class KGPathDataset(Dataset):
    """
    Dataset for Knowledge Graph Path Generation.
    
    Input: Question + KG subgraph
    Output: Full reasoning path (sequence of entities and relations)
    
    During training, randomly samples from ALL available paths for each sample.
    During validation/inference, uses the first path for consistent evaluation.
    """
    
    def __init__(
        self,
        data_path: str,
        vocab: Optional[EntityRelationVocab] = None,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_question_length: int = 64,
        max_path_length: int = 10,
        max_graph_nodes: int = 200,
        max_entities: int = 50000,
        max_relations: int = 5000,
        build_vocab: bool = False,
        training: bool = True  # If True, randomly sample paths; if False, use first path
    ):
        self.data_path = Path(data_path)
        self.max_question_length = max_question_length
        self.max_path_length = max_path_length
        self.max_graph_nodes = max_graph_nodes
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.training = training  # Whether to randomly sample paths
        
        # Load tokenizer for questions
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load data
        self.samples = self._load_data()
        
        # Build or use existing vocabulary
        if vocab is None:
            self.vocab = EntityRelationVocab(max_entities=max_entities, max_relations=max_relations)
            if build_vocab:
                self._build_vocab()
        else:
            self.vocab = vocab
    
    def _load_data(self) -> List[Dict]:
        """Load data from parquet or jsonl file."""
        if self.data_path.suffix == '.parquet':
            df = pd.read_parquet(self.data_path)
            samples = df.to_dict('records')
            # Handle JSON-encoded fields in parquet
            for sample in samples:
                for key in ['graph', 'paths', 'answer', 'q_entity', 'a_entity']:
                    if key in sample and isinstance(sample[key], str):
                        try:
                            sample[key] = json.loads(sample[key])
                        except:
                            pass
            return samples
        elif self.data_path.suffix in ['.jsonl', '.json']:
            samples = []
            with open(self.data_path, 'r', encoding='utf-8') as f:
                if self.data_path.suffix == '.jsonl':
                    for line in f:
                        samples.append(json.loads(line))
                else:
                    samples = json.load(f)
            return samples
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
    
    def _build_vocab(self):
        """Build vocabulary from dataset."""
        for sample in self.samples:
            # Add entities and relations from graph
            graph = sample.get('graph', [])
            for triple in graph:
                if len(triple) == 3:
                    subj, rel, obj = triple[0], triple[1], triple[2]
                    self.vocab.add_entity(str(subj))
                    self.vocab.add_relation(str(rel))
                    self.vocab.add_entity(str(obj))
            
            # Add entities from paths if available
            paths = sample.get('paths', [])
            for path in paths:
                if isinstance(path, dict):
                    for entity in path.get('entities', []):
                        self.vocab.add_entity(str(entity))
                    for rel in path.get('relations', []):
                        self.vocab.add_relation(str(rel))
    
    def _parse_graph(self, graph_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        Parse graph triples into PyG format.
        
        Returns:
            edge_index: [2, num_edges] tensor
            edge_type: [num_edges] tensor
            node_ids: [num_nodes] tensor
            entity_to_local_idx: mapping from entity to local node index
        """
        entity_to_local_idx = {}
        edges = []
        edge_types = []
        
        for triple in graph_data:
            if len(triple) != 3:
                continue
            
            subj, rel, obj = str(triple[0]), str(triple[1]), str(triple[2])
            
            # Get or create local node indices
            if subj not in entity_to_local_idx:
                entity_to_local_idx[subj] = len(entity_to_local_idx)
            if obj not in entity_to_local_idx:
                entity_to_local_idx[obj] = len(entity_to_local_idx)
            
            if len(entity_to_local_idx) > self.max_graph_nodes:
                break
            
            src_idx = entity_to_local_idx[subj]
            dst_idx = entity_to_local_idx[obj]
            rel_idx = self.vocab.get_relation_idx(rel)
            
            edges.append([src_idx, dst_idx])
            edge_types.append(rel_idx)
        
        if len(edges) == 0:
            # Return empty graph with at least one node
            return (
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
                {}
            )
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        # Create node_ids tensor (global vocab indices)
        node_ids = torch.zeros(len(entity_to_local_idx), dtype=torch.long)
        for entity, local_idx in entity_to_local_idx.items():
            node_ids[local_idx] = self.vocab.get_entity_idx(entity)
        
        return edge_index, edge_type, node_ids, entity_to_local_idx
    
    def _encode_path(self, path: Dict, entity_to_local_idx: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Encode a single path as sequence of (entity_idx, relation_idx) pairs.
        
        Returns dict with:
            - path_entities: [path_length] entity indices
            - path_relations: [path_length-1] relation indices
            - path_local_entities: [path_length] local node indices in the graph
            - path_length: scalar
        """
        entities = path.get('entities', [])
        relations = path.get('relations', [])
        
        # Truncate to max length
        entities = entities[:self.max_path_length]
        relations = relations[:self.max_path_length - 1]
        
        # Encode
        path_entities = [self.vocab.get_entity_idx(str(e)) for e in entities]
        path_relations = [self.vocab.get_relation_idx(str(r)) for r in relations]
        path_local = [entity_to_local_idx.get(str(e), 0) for e in entities]
        
        # Add BOS/EOS tokens
        bos_idx = self.vocab.entity2idx["<BOS>"]
        eos_idx = self.vocab.entity2idx["<EOS>"]
        path_entities = [bos_idx] + path_entities + [eos_idx]
        path_local = [0] + path_local + [0]  # Dummy local indices for special tokens
        
        return {
            'path_entities': torch.tensor(path_entities, dtype=torch.long),
            'path_relations': torch.tensor(path_relations, dtype=torch.long),
            'path_local_entities': torch.tensor(path_local, dtype=torch.long),
            'path_length': torch.tensor(len(path_entities), dtype=torch.long)
        }
    
    def _select_diverse_paths(self, paths: List[Dict], max_paths: int = 20) -> List[Dict]:
        """
        Select diverse paths that cover different answer entities.
        
        Strategy:
        1. Group paths by their target entity (last entity in path)
        2. Select shortest path for each unique target entity
        3. Return up to max_paths diverse paths
        """
        if not paths:
            return []
        
        # Group by target entity
        paths_by_target = {}
        for path in paths:
            entities = path.get('entities', [])
            if entities:
                target = entities[-1]
                if target not in paths_by_target:
                    paths_by_target[target] = []
                paths_by_target[target].append(path)
        
        # Select shortest path for each target
        diverse_paths = []
        for target, target_paths in paths_by_target.items():
            # Sort by path length (shorter first)
            target_paths.sort(key=lambda p: len(p.get('relations', [])))
            diverse_paths.append(target_paths[0])
        
        # Sort by path length and limit
        diverse_paths.sort(key=lambda p: len(p.get('relations', [])))
        return diverse_paths[:max_paths]
    
    def _encode_all_paths(self, paths: List[Dict], entity_to_local_idx: Dict[str, int], max_paths: int = 20) -> Dict[str, torch.Tensor]:
        """
        Encode ALL paths (up to max_paths) as padded tensors.
        
        Returns dict with:
            - all_path_entities: [num_paths, max_path_length] entity indices
            - all_path_relations: [num_paths, max_path_length-1] relation indices
            - all_path_local_entities: [num_paths, max_path_length] local node indices
            - all_path_lengths: [num_paths] actual length of each path
            - num_paths: scalar - number of valid paths
        """
        bos_idx = self.vocab.entity2idx["<BOS>"]
        eos_idx = self.vocab.entity2idx["<EOS>"]
        pad_idx = self.vocab.entity2idx["<PAD>"]
        rel_pad_idx = self.vocab.relation2idx["<PAD>"]
        
        # Select diverse paths covering different answer entities
        diverse_paths = self._select_diverse_paths(paths, max_paths=max_paths)
        num_paths = len(diverse_paths)
        
        if num_paths == 0:
            # Return empty tensors
            return {
                'all_path_entities': torch.zeros(1, self.max_path_length + 2, dtype=torch.long),
                'all_path_relations': torch.zeros(1, self.max_path_length, dtype=torch.long),
                'all_path_local_entities': torch.zeros(1, self.max_path_length + 2, dtype=torch.long),
                'all_path_lengths': torch.tensor([2], dtype=torch.long),
                'num_paths': torch.tensor(0, dtype=torch.long)
            }
        
        # Initialize padded tensors
        max_len = self.max_path_length + 2  # +2 for BOS and EOS
        all_entities = torch.full((num_paths, max_len), pad_idx, dtype=torch.long)
        all_relations = torch.full((num_paths, max_len - 1), rel_pad_idx, dtype=torch.long)
        all_local = torch.zeros((num_paths, max_len), dtype=torch.long)
        all_lengths = torch.zeros(num_paths, dtype=torch.long)
        
        for i, path in enumerate(diverse_paths):
            entities = path.get('entities', [])[:self.max_path_length]
            relations = path.get('relations', [])[:self.max_path_length - 1]
            
            # Encode entities
            path_entities = [self.vocab.get_entity_idx(str(e)) for e in entities]
            path_relations = [self.vocab.get_relation_idx(str(r)) for r in relations]
            path_local = [entity_to_local_idx.get(str(e), 0) for e in entities]
            
            # Add BOS/EOS
            path_entities = [bos_idx] + path_entities + [eos_idx]
            path_local = [0] + path_local + [0]
            
            # Fill tensors
            path_len = len(path_entities)
            all_entities[i, :path_len] = torch.tensor(path_entities, dtype=torch.long)
            all_relations[i, :len(path_relations)] = torch.tensor(path_relations, dtype=torch.long)
            all_local[i, :path_len] = torch.tensor(path_local, dtype=torch.long)
            all_lengths[i] = path_len
        
        return {
            'all_path_entities': all_entities,
            'all_path_relations': all_relations,
            'all_path_local_entities': all_local,
            'all_path_lengths': all_lengths,
            'num_paths': torch.tensor(num_paths, dtype=torch.long)
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Encode question
        question = sample.get('question', '')
        question_encoding = self.tokenizer(
            question,
            max_length=self.max_question_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Parse graph
        graph_data = sample.get('graph', [])
        edge_index, edge_type, node_ids, entity_to_local_idx = self._parse_graph(graph_data)
        
        # Get question and answer entity indices in local graph
        q_entities = sample.get('q_entity', sample.get('q_entities', []))
        a_entities = sample.get('a_entity', sample.get('a_entities', []))
        
        if isinstance(q_entities, str):
            q_entities = [q_entities]
        if isinstance(a_entities, str):
            a_entities = [a_entities]
        
        q_local_indices = [entity_to_local_idx.get(str(e), 0) for e in q_entities]
        a_local_indices = [entity_to_local_idx.get(str(e), 0) for e in a_entities]
        
        # Encode ALL target paths (diverse paths covering different answer entities)
        paths = sample.get('paths', [])
        if paths and isinstance(paths[0], dict):
            # Encode all diverse paths
            all_paths = self._encode_all_paths(paths, entity_to_local_idx, max_paths=20)
            
            # Also encode first path for backward compatibility (single path output)
            first_path = self._encode_path(paths[0], entity_to_local_idx)
        else:
            # Empty paths
            all_paths = {
                'all_path_entities': torch.zeros(1, self.max_path_length + 2, dtype=torch.long),
                'all_path_relations': torch.zeros(1, self.max_path_length, dtype=torch.long),
                'all_path_local_entities': torch.zeros(1, self.max_path_length + 2, dtype=torch.long),
                'all_path_lengths': torch.tensor([2], dtype=torch.long),
                'num_paths': torch.tensor(0, dtype=torch.long)
            }
            first_path = {
                'path_entities': torch.tensor([self.vocab.entity2idx["<BOS>"], self.vocab.entity2idx["<EOS>"]], dtype=torch.long),
                'path_relations': torch.tensor([], dtype=torch.long),
                'path_local_entities': torch.tensor([0, 0], dtype=torch.long),
                'path_length': torch.tensor(2, dtype=torch.long)
            }
        
        return {
            'id': sample.get('id', str(idx)),
            'question_input_ids': question_encoding['input_ids'].squeeze(0),
            'question_attention_mask': question_encoding['attention_mask'].squeeze(0),
            'edge_index': edge_index,
            'edge_type': edge_type,
            'node_ids': node_ids,
            'num_nodes': torch.tensor(len(entity_to_local_idx) if entity_to_local_idx else 1, dtype=torch.long),
            'q_entity_indices': torch.tensor(q_local_indices, dtype=torch.long),
            'a_entity_indices': torch.tensor(a_local_indices, dtype=torch.long),
            # Single path (for backward compatibility)
            **first_path,
            # All paths
            **all_paths
        }


def collate_kg_batch(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for batching KG data with variable graph sizes.
    Uses PyG's Batch for efficient graph batching.
    
    Now supports multi-path output:
    - path_entities, path_relations, etc.: Single path (first path, for backward compatibility)
    - all_path_entities, all_path_relations, etc.: All diverse paths
    """
    # Separate fixed-size tensors and variable-size graph data
    ids = [item['id'] for item in batch]
    
    # Stack fixed-size tensors
    question_input_ids = torch.stack([item['question_input_ids'] for item in batch])
    question_attention_mask = torch.stack([item['question_attention_mask'] for item in batch])
    
    # Create PyG Data objects and batch them
    graph_data_list = []
    for item in batch:
        data = Data(
            edge_index=item['edge_index'],
            edge_type=item['edge_type'],
            node_ids=item['node_ids'],
            num_nodes=item['num_nodes'].item()
        )
        graph_data_list.append(data)
    
    graph_batch = Batch.from_data_list(graph_data_list)
    
    # ===== Single path (backward compatibility) =====
    max_path_len = max(item['path_entities'].size(0) for item in batch)
    
    path_entities = []
    path_relations = []
    path_local_entities = []
    path_lengths = []
    
    for item in batch:
        pe = item['path_entities']
        pr = item['path_relations']
        pl = item['path_local_entities']
        
        # Pad path entities
        pad_len = max_path_len - pe.size(0)
        if pad_len > 0:
            pe = torch.cat([pe, torch.zeros(pad_len, dtype=torch.long)])
            pl = torch.cat([pl, torch.zeros(pad_len, dtype=torch.long)])
        
        # Pad path relations (one less than entities)
        max_rel_len = max_path_len - 1
        rel_pad_len = max_rel_len - pr.size(0)
        if rel_pad_len > 0:
            pr = torch.cat([pr, torch.zeros(rel_pad_len, dtype=torch.long)])
        
        path_entities.append(pe)
        path_relations.append(pr)
        path_local_entities.append(pl)
        path_lengths.append(item['path_length'])
    
    # ===== All paths (multi-path output) =====
    # Get max dimensions across batch
    max_num_paths = max(item['all_path_entities'].size(0) for item in batch)
    max_path_entity_len = max(item['all_path_entities'].size(1) for item in batch)
    max_path_rel_len = max(item['all_path_relations'].size(1) for item in batch)
    
    all_path_entities_list = []
    all_path_relations_list = []
    all_path_local_entities_list = []
    all_path_lengths_list = []
    num_paths_list = []
    
    for item in batch:
        ape = item['all_path_entities']  # [num_paths, path_len]
        apr = item['all_path_relations']  # [num_paths, rel_len]
        apl = item['all_path_local_entities']  # [num_paths, path_len]
        aplen = item['all_path_lengths']  # [num_paths]
        
        curr_num_paths = ape.size(0)
        curr_path_len = ape.size(1)
        curr_rel_len = apr.size(1)
        
        # Pad paths dimension
        if curr_num_paths < max_num_paths:
            pad_paths = max_num_paths - curr_num_paths
            ape = torch.cat([ape, torch.zeros(pad_paths, curr_path_len, dtype=torch.long)], dim=0)
            apr = torch.cat([apr, torch.zeros(pad_paths, curr_rel_len, dtype=torch.long)], dim=0)
            apl = torch.cat([apl, torch.zeros(pad_paths, curr_path_len, dtype=torch.long)], dim=0)
            aplen = torch.cat([aplen, torch.zeros(pad_paths, dtype=torch.long)])
        
        # Pad path length dimension
        if curr_path_len < max_path_entity_len:
            pad_len = max_path_entity_len - curr_path_len
            ape = torch.cat([ape, torch.zeros(max_num_paths, pad_len, dtype=torch.long)], dim=1)
            apl = torch.cat([apl, torch.zeros(max_num_paths, pad_len, dtype=torch.long)], dim=1)
        
        if curr_rel_len < max_path_rel_len:
            pad_len = max_path_rel_len - curr_rel_len
            apr = torch.cat([apr, torch.zeros(max_num_paths, pad_len, dtype=torch.long)], dim=1)
        
        all_path_entities_list.append(ape)
        all_path_relations_list.append(apr)
        all_path_local_entities_list.append(apl)
        all_path_lengths_list.append(aplen)
        num_paths_list.append(item['num_paths'])
    
    # Handle entity indices with padding
    max_q_entities = max(item['q_entity_indices'].size(0) for item in batch)
    max_a_entities = max(item['a_entity_indices'].size(0) for item in batch)
    
    q_entity_indices = []
    a_entity_indices = []
    
    for item in batch:
        qi = item['q_entity_indices']
        ai = item['a_entity_indices']
        
        if qi.size(0) < max_q_entities:
            qi = torch.cat([qi, torch.zeros(max_q_entities - qi.size(0), dtype=torch.long)])
        if ai.size(0) < max_a_entities:
            ai = torch.cat([ai, torch.zeros(max_a_entities - ai.size(0), dtype=torch.long)])
        
        q_entity_indices.append(qi)
        a_entity_indices.append(ai)
    
    return {
        'ids': ids,
        'question_input_ids': question_input_ids,
        'question_attention_mask': question_attention_mask,
        'graph_batch': graph_batch,
        'q_entity_indices': torch.stack(q_entity_indices),
        'a_entity_indices': torch.stack(a_entity_indices),
        # Single path (backward compatibility)
        'path_entities': torch.stack(path_entities),
        'path_relations': torch.stack(path_relations),
        'path_local_entities': torch.stack(path_local_entities),
        'path_lengths': torch.stack(path_lengths),
        # All paths (multi-path output)
        'all_path_entities': torch.stack(all_path_entities_list),      # [batch, max_paths, max_len]
        'all_path_relations': torch.stack(all_path_relations_list),    # [batch, max_paths, max_len-1]
        'all_path_local_entities': torch.stack(all_path_local_entities_list),  # [batch, max_paths, max_len]
        'all_path_lengths': torch.stack(all_path_lengths_list),        # [batch, max_paths]
        'num_paths': torch.stack(num_paths_list)                       # [batch]
    }


class KGPathDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for efficient data loading with parallel workers.
    
    Supports two modes:
    1. Pre-built vocabulary: Load unified vocab from vocab_path (recommended)
       - Ensures all entities from train/test/val are in vocabulary
       - Prevents UNK tokens during inference
    2. Build vocabulary: Build from training data only (legacy mode)
       - May cause UNK tokens for entities not in training data
    """
    
    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        vocab_path: Optional[str] = None,  # Path to pre-built vocabulary
        batch_size: int = 32,
        num_workers: int = 4,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_question_length: int = 64,
        max_path_length: int = 10,
        max_graph_nodes: int = 200,
        max_entities: int = 50000,
        max_relations: int = 5000
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
        self.max_path_length = max_path_length
        self.max_graph_nodes = max_graph_nodes
        self.max_entities = max_entities
        self.max_relations = max_relations
        
        self.vocab = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets and load/build vocabulary."""
        if stage == 'fit' or stage is None:
            # Load pre-built vocabulary if provided (recommended)
            if self.vocab_path and Path(self.vocab_path).exists():
                print(f"Loading pre-built vocabulary from {self.vocab_path}...")
                self.vocab = EntityRelationVocab.load(self.vocab_path)
                print(f"  Loaded {self.vocab.num_entities} entities, {self.vocab.num_relations} relations")
                build_vocab = False
            else:
                print("No pre-built vocabulary provided, building from training data...")
                print("  Warning: This may cause UNK tokens for entities not in training data")
                self.vocab = None
                build_vocab = True
            
            # Create training dataset
            # training=True: randomly sample from all paths for data augmentation
            self.train_dataset = KGPathDataset(
                self.train_path,
                vocab=self.vocab,
                tokenizer_name=self.tokenizer_name,
                max_question_length=self.max_question_length,
                max_path_length=self.max_path_length,
                max_graph_nodes=self.max_graph_nodes,
                max_entities=self.max_entities,
                max_relations=self.max_relations,
                build_vocab=build_vocab,
                training=True  # Randomly sample paths during training
            )
            
            # Use the vocab from dataset if we built it
            if build_vocab:
                self.vocab = self.train_dataset.vocab
            
            if self.val_path:
                # training=False: use first path for consistent evaluation
                self.val_dataset = KGPathDataset(
                    self.val_path,
                    vocab=self.vocab,
                    tokenizer_name=self.tokenizer_name,
                    max_question_length=self.max_question_length,
                    max_path_length=self.max_path_length,
                    max_graph_nodes=self.max_graph_nodes,
                    max_entities=self.max_entities,
                    max_relations=self.max_relations,
                    build_vocab=False,
                    training=False  # Use first path for consistent evaluation
                )
        
        if stage == 'test' or stage is None:
            if self.test_path and self.vocab:
                self.test_dataset = KGPathDataset(
                    self.test_path,
                    vocab=self.vocab,
                    tokenizer_name=self.tokenizer_name,
                    max_question_length=self.max_question_length,
                    max_path_length=self.max_path_length,
                    max_graph_nodes=self.max_graph_nodes,
                    max_entities=self.max_entities,
                    max_relations=self.max_relations,
                    build_vocab=False,
                    training=False  # Use first path for consistent evaluation
                )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_kg_batch,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_kg_batch,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_kg_batch,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

