"""
Path Ranker Model: Question + Candidate Paths -> Path Ranking

Uses better text embeddings to rank candidate relation paths given a question.
Input: Question text + Candidate relation paths (not raw graph)
Output: Ranking/selection of the best path that answers the question

Architecture:
1. Question Encoder: Better embedding model (BGE-M3, E5-large, or SBERT)
2. Path Encoder: Encodes relation paths as text sequences
3. Scoring: Cross-attention or dot product scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import pytorch_lightning as pl
except ImportError:
    class _LightningModule(nn.Module):
        def save_hyperparameters(self, *args, **kwargs):
            pass

        def log(self, *args, **kwargs):
            pass

    class _PL:
        LightningModule = _LightningModule

    pl = _PL()
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple, List, Any
import math


class PathRankerModel(pl.LightningModule):
    """
    Path Ranker that scores candidate relation paths for a question.
    
    Uses bi-encoder architecture:
    - Encode question with text encoder
    - Encode each candidate path as text (relations joined with " -> ")
    - Score paths via cosine similarity or learned scorer
    """
    
    def __init__(
        self,
        encoder_name: str = "BAAI/bge-base-en-v1.5",  # Better embedding model
        hidden_dim: int = 768,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 50000,
        freeze_encoder: bool = False,  # Fine-tune the encoder
        temperature: float = 0.05,  # For contrastive loss
        max_candidates: int = 100,  # Max candidate paths per sample
        max_path_length: int = 8,   # Max relations in a path
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_candidates = max_candidates
        self.max_path_length = max_path_length
        
        # Load encoder and tokenizer
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder_hidden_size = self.encoder.config.hidden_size
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Projection layer if encoder hidden size differs
        if self.encoder_hidden_size != hidden_dim:
            self.projection = nn.Linear(self.encoder_hidden_size, hidden_dim)
        else:
            self.projection = nn.Identity()
        
        # Path scorer (optional, can also use cosine similarity)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # Loss
        self.loss_fn = nn.CrossEntropyLoss()
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text using the encoder model."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Mean pooling
        hidden = outputs.last_hidden_state  # [B, seq_len, hidden]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        # Project
        return self.projection(pooled)
    
    def path_to_text(self, path: List[str]) -> str:
        """Convert relation path to text for encoding."""
        # Replace underscores and dots with spaces for better tokenization
        relations = [r.replace('.', ' ').replace('_', ' ') for r in path]
        return " -> ".join(relations)
    
    def forward(
        self,
        question_input_ids: torch.Tensor,      # [B, seq_len]
        question_attention_mask: torch.Tensor,  # [B, seq_len]
        path_input_ids: torch.Tensor,           # [B, num_candidates, seq_len]
        path_attention_mask: torch.Tensor,      # [B, num_candidates, seq_len]
        candidate_mask: torch.Tensor,           # [B, num_candidates] - 1 for valid, 0 for padding
        labels: Optional[torch.Tensor] = None,  # [B] - index of correct path
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for path ranking.
        
        Returns:
            logits: [B, num_candidates] - scores for each candidate
            loss: scalar loss (if labels provided)
        """
        B, num_candidates, seq_len = path_input_ids.shape
        
        # Encode question
        question_emb = self.encode_text(
            question_input_ids, question_attention_mask
        )  # [B, hidden_dim]
        
        # Encode all paths (flatten, encode, reshape)
        path_input_ids_flat = path_input_ids.view(B * num_candidates, seq_len)
        path_attention_mask_flat = path_attention_mask.view(B * num_candidates, seq_len)
        
        path_emb_flat = self.encode_text(
            path_input_ids_flat, path_attention_mask_flat
        )  # [B * num_candidates, hidden_dim]
        
        path_emb = path_emb_flat.view(B, num_candidates, -1)  # [B, num_candidates, hidden_dim]
        
        # Score paths - option 1: dot product / cosine similarity
        question_emb_norm = F.normalize(question_emb, dim=-1)
        path_emb_norm = F.normalize(path_emb, dim=-1)
        
        logits = torch.bmm(
            path_emb_norm,
            question_emb_norm.unsqueeze(-1)
        ).squeeze(-1)  # [B, num_candidates]
        
        # Apply temperature
        logits = logits / self.temperature
        
        # Mask invalid candidates
        logits = logits.masked_fill(~candidate_mask.bool(), float('-inf'))
        
        outputs = {'logits': logits}
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs['loss'] = loss
        
        return outputs
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            path_input_ids=batch['path_input_ids'],
            path_attention_mask=batch['path_attention_mask'],
            candidate_mask=batch['candidate_mask'],
            labels=batch['labels'],
        )
        
        self.log('train/loss', outputs['loss'], prog_bar=True)
        
        # Compute accuracy
        preds = outputs['logits'].argmax(dim=-1)
        acc = (preds == batch['labels']).float().mean()
        self.log('train/acc', acc, prog_bar=True)
        
        return outputs['loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            path_input_ids=batch['path_input_ids'],
            path_attention_mask=batch['path_attention_mask'],
            candidate_mask=batch['candidate_mask'],
            labels=batch['labels'],
        )
        
        self.log('val/loss', outputs['loss'], prog_bar=True, sync_dist=True)
        
        # Compute accuracy
        preds = outputs['logits'].argmax(dim=-1)
        acc = (preds == batch['labels']).float().mean()
        self.log('val/acc', acc, prog_bar=True, sync_dist=True)
        
        return outputs['loss']
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            path_input_ids=batch['path_input_ids'],
            path_attention_mask=batch['path_attention_mask'],
            candidate_mask=batch['candidate_mask'],
            labels=batch['labels'],
        )
        
        self.log('test/loss', outputs['loss'], prog_bar=True, sync_dist=True)
        
        preds = outputs['logits'].argmax(dim=-1)
        acc = (preds == batch['labels']).float().mean()
        self.log('test/acc', acc, prog_bar=True, sync_dist=True)
        
        return outputs['loss']
    
    @torch.no_grad()
    def rank_paths(
        self,
        question: str,
        candidate_paths: List[List[str]],
        top_k: int = 10,
    ) -> List[Tuple[List[str], float]]:
        """
        Rank candidate paths for a question.
        
        Returns:
            List of (path, score) tuples sorted by score descending
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Encode question
        q_enc = self.tokenizer(
            question,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).to(device)
        
        question_emb = self.encode_text(
            q_enc['input_ids'], q_enc['attention_mask']
        )  # [1, hidden]
        question_emb_norm = F.normalize(question_emb, dim=-1)
        
        # Encode paths in batches
        path_texts = [self.path_to_text(p) for p in candidate_paths]
        
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(path_texts), batch_size):
            batch_texts = path_texts[i:i+batch_size]
            p_enc = self.tokenizer(
                batch_texts,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            ).to(device)
            
            path_emb = self.encode_text(
                p_enc['input_ids'], p_enc['attention_mask']
            )
            path_emb_norm = F.normalize(path_emb, dim=-1)
            
            scores = (path_emb_norm @ question_emb_norm.T).squeeze(-1)
            all_scores.extend(scores.cpu().tolist())
        
        # Sort by score
        ranked = sorted(
            zip(candidate_paths, all_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked[:top_k]
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }


if __name__ == '__main__':
    # Test the model
    print("Testing PathRankerModel...")
    
    model = PathRankerModel(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller for testing
        hidden_dim=384,
    )
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test ranking
    question = "What is the capital of France?"
    candidates = [
        ["location.country.capital"],
        ["people.person.nationality", "location.country.capital"],
        ["film.film.country"],
        ["music.artist.origin"],
    ]
    
    ranked = model.rank_paths(question, candidates, top_k=4)
    print(f"\nQuestion: {question}")
    print("Ranked paths:")
    for path, score in ranked:
        print(f"  {score:.4f}: {path}")
