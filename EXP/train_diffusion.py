"""
Train Diffusion Model: Train the content-aware path diffusion model.
(Raw PyTorch version, no Lightning dependency)

This script:
1. Loads WebQSP data and vocab
2. Prepares dataset of (question, path) pairs
3. Trains diffusion model components
4. Saves checkpoints in format compatible with DiffusionRanker
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import warnings
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

# Add Core to path
CORE_PATH = Path(__file__).parent.parent / "Core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))
    
# Import Core components (No PL dependency)
try:
    from Core.modules.diffusion import PathDiffusionTransformer, DiscreteDiffusion
except ImportError as e:
    print(f"Error importing Core modules: {e}")
    sys.exit(1)


class QuestionEncoder(nn.Module):
    """Encode questions using pretrained transformer."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 256,
        freeze: bool = False
    ):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.proj = nn.Linear(self.hidden_size, output_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.proj(outputs.last_hidden_state)
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        return sequence_output, pooled_output


class DiffusionModel(nn.Module):
    """
    Wrapper for diffusion components.
    Combination of QuestionEncoder, PathDiffusionTransformer, DiscreteDiffusion.
    """
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 256,
        max_path_length: int = 10,
        predict_entities: bool = True
    ):
        super().__init__()
        
        self.question_encoder = QuestionEncoder(output_dim=hidden_dim)
        
        self.denoiser = PathDiffusionTransformer(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            max_path_length=max_path_length,
            predict_entities=predict_entities,
            question_dim=hidden_dim
        )
        
        self.diffusion = DiscreteDiffusion(
            num_entities=num_entities,
            num_relations=num_relations,
            num_timesteps=1000
        )
        
    def forward(self, batch):
        # 1. Encode question
        q_emb, _ = self.question_encoder(
            batch['question_input_ids'], 
            batch['question_attention_mask']
        )
        
        # 2. Add noise
        batch_size = q_emb.shape[0]
        device = q_emb.device
        
        # Sample random timestep
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=device).long()
        
        target_entities = batch['target_entities']
        target_relations = batch['target_relations']
        
        noisy_entities, noisy_relations = self.diffusion.q_sample(
            target_entities, target_relations, t
        )
        
        # 3. Denoise
        path_mask = batch.get('path_mask', None)
        
        entity_logits, relation_logits = self.denoiser(
            noisy_entities,
            noisy_relations,
            t,
            q_emb,
            path_mask=path_mask,
            question_mask=batch['question_attention_mask'] == 0 # Key padding mask (True where pad)
        )
        
        # 4. Compute Loss
        loss_dict = self.diffusion.compute_loss(
            entity_logits,
            relation_logits,
            target_entities,
            target_relations,
            path_mask=path_mask
        )
        
        return loss_dict['loss']


class DiffusionDataset(Dataset):
    """Dataset for diffusion training."""
    
    def __init__(
        self,
        questions: List[str],
        paths: List[Dict[str, Any]],
        tokenizer: Any,
        relation2idx: Dict[str, int],
        entity2idx: Dict[str, int],
        max_path_length: int = 10,
        max_question_length: int = 64
    ):
        self.questions = questions
        self.paths = paths
        self.tokenizer = tokenizer
        self.relation2idx = relation2idx
        self.entity2idx = entity2idx
        self.max_path_length = max_path_length
        self.max_question_length = max_question_length
        self.pad_token_id = tokenizer.pad_token_id
        
        self.rel_unk = relation2idx.get('<UNK>', 0)
        self.ent_unk = entity2idx.get('<UNK>', 0)
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        path_data = self.paths[idx]
        
        q_enc = self.tokenizer(
            question,
            max_length=self.max_question_length,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        
        relations = path_data.get('relations', [])
        entities = path_data.get('entities', [])
        
        rel_ids = [self.relation2idx.get(r, self.rel_unk) for r in relations]
        ent_ids = [self.entity2idx.get(e, self.ent_unk) for e in entities]
        
        # Ensure consistency: len(relations) should be len(entities) - 1
        # We drive truncation by entities
        if not ent_ids: # Handle empty case
            ent_ids = [self.ent_unk]
            rel_ids = []
            
        ent_ids = ent_ids[:self.max_path_length]
        L = len(ent_ids)
        
        # Relations must be exactly L-1
        if L > 1:
            rel_ids = rel_ids[:L-1]
            # Pad if shorter (shouldn't happen for valid paths usually)
            if len(rel_ids) < L-1:
                rel_ids = rel_ids + [self.rel_unk] * ((L-1) - len(rel_ids))
        else:
            rel_ids = []
            
        return {
            'q_input_ids': q_enc['input_ids'].squeeze(0),
            'q_attention_mask': q_enc['attention_mask'].squeeze(0),
            'rel_ids': torch.tensor(rel_ids, dtype=torch.long),
            'ent_ids': torch.tensor(ent_ids, dtype=torch.long)
        }


def collate_fn(batch):
    """Collate batch with padding."""
    q_input_ids = [b['q_input_ids'] for b in batch]
    q_attention_mask = [b['q_attention_mask'] for b in batch]
    rel_ids = [b['rel_ids'] for b in batch]
    ent_ids = [b['ent_ids'] for b in batch]
    
    q_input_ids_padded = pad_sequence(q_input_ids, batch_first=True, padding_value=0)
    q_attention_mask_padded = pad_sequence(q_attention_mask, batch_first=True, padding_value=0)
    
    # 0 is PAD
    rel_ids_padded = pad_sequence(rel_ids, batch_first=True, padding_value=0)
    ent_ids_padded = pad_sequence(ent_ids, batch_first=True, padding_value=0)
    
    # Path mask based on ENTITIES
    # [B, L]
    path_mask = (ent_ids_padded != 0).long()
    
    return {
        'question_input_ids': q_input_ids_padded,
        'question_attention_mask': q_attention_mask_padded,
        'target_relations': rel_ids_padded,
        'target_entities': ent_ids_padded,
        'path_mask': path_mask
    }


def load_vocab(vocab_path: Path) -> Tuple[Dict, Dict]:
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        rel2idx = vocab.get('relation2idx', vocab.get('relation2id', {}))
        ent2idx = vocab.get('entity2idx', vocab.get('entity2id', {}))
        return rel2idx, ent2idx
    except Exception as e:
        print(f"Error loading vocab from {vocab_path}: {e}")
        try:
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            rel2idx = vocab.get('relation2idx', vocab.get('relation2id', {}))
            ent2idx = vocab.get('entity2idx', vocab.get('entity2id', {}))
            return rel2idx, ent2idx
        except:
            return {}, {}


def prepare_data(parquet_path: Path) -> Tuple[List[str], List[Dict]]:
    df = pd.read_parquet(parquet_path)
    questions = []
    paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {parquet_path.name}"):
        q = row['question']
        p_str = row['paths']
        if isinstance(p_str, str):
            try:
                p_list = json.loads(p_str)
            except:
                p_list = []
        else:
            p_list = p_str if p_str else []
        if not p_list: continue
        
        # Use first valid path
        for p in p_list:
            if p.get('relations'):
                questions.append(q)
                paths.append(p)
                break 
    return questions, paths


def train(args):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load Vocab
    vocab_file = data_path / "vocab.json"
    rel2idx, ent2idx = load_vocab(vocab_file)
    if not rel2idx:
        print("Error: Empty vocabulary!")
        return
    print(f"Vocab: {len(rel2idx)} relations, {len(ent2idx)} entities")
    
    # Prepare Data
    train_q, train_p = prepare_data(data_path / "train.parquet")
    val_q, val_p = prepare_data(data_path / "val.parquet")
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    train_dataset = DiffusionDataset(train_q, train_p, tokenizer, rel2idx, ent2idx, max_path_length=args.max_path_len)
    val_dataset = DiffusionDataset(val_q, val_p, tokenizer, rel2idx, ent2idx, max_path_length=args.max_path_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Model
    model = DiffusionModel(
        num_entities=len(ent2idx) + 1,
        num_relations=len(rel2idx) + 1,
        hidden_dim=256,
        max_path_length=args.max_path_len,
        predict_entities=True
    ).to(device)
    
    # Lower LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    
    # Train Loop
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            
            val = loss.item()
            import math
            if not math.isnan(val): # Only add if not NaN
                train_loss += val
            else:
                # Optionally warn
                pass
            pbar.set_postfix({'loss': val})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        valid_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss = model(batch)
                if not math.isnan(loss.item()):
                    val_loss += loss.item()
                    valid_batches += 1
        
        avg_val_loss = val_loss / valid_batches if valid_batches > 0 else float('nan')
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        # Save state always
        state = {
            'state_dict': model.state_dict(),
            'config': {
                'num_entities': len(ent2idx) + 1,
                'num_relations': len(rel2idx) + 1,
                'hidden_dim': 256,
                'max_path_length': args.max_path_len
            }
        }
        # Save last
        torch.save(state, out_path / "model.pt")
        
        if not math.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Copy to best (or just keep valid model.pt)
            print(f"Saved best model (Val Loss: {best_val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Data/webqsp_final")
    parser.add_argument("--output_dir", default="EXP/models/diffusion")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_path_len", type=int, default=10)
    
    args = parser.parse_args()
    try:
        train(args)
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
