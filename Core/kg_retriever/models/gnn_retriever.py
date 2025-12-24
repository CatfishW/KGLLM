"""
GNN-based Path Retriever for KGQA.

Inspired by ReaRev: Uses GNN message passing over KG subgraph
to iteratively refine entity/relation representations and score paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from typing import Dict, Optional, Tuple, List
import math


class GNNLayer(nn.Module):
    """
    Single GNN layer with relation-aware message passing.
    
    Aggregates neighbor information weighted by relation embeddings.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Relation-aware attention
        self.rel_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,      # [B, num_nodes, hidden_dim]
        edge_index: torch.Tensor,          # [B, num_edges, 2] (head_idx, tail_idx)
        edge_relations: torch.Tensor,      # [B, num_edges, hidden_dim]
        edge_mask: torch.Tensor,           # [B, num_edges]
    ) -> torch.Tensor:
        """
        Perform one round of message passing.
        
        Returns:
            updated_features: [B, num_nodes, hidden_dim]
        """
        B, num_nodes, D = node_features.shape
        _, num_edges, _ = edge_index.shape
        
        # Get source and target indices
        src_idx = edge_index[:, :, 0]  # [B, num_edges]
        tgt_idx = edge_index[:, :, 1]  # [B, num_edges]
        
        # Gather source node features
        src_idx_expanded = src_idx.unsqueeze(-1).expand(-1, -1, D)
        src_features = node_features.gather(1, src_idx_expanded)  # [B, num_edges, D]
        
        # Relation-aware message
        rel_features = self.rel_proj(edge_relations)  # [B, num_edges, D]
        messages = src_features + rel_features  # [B, num_edges, D]
        
        # Apply mask
        messages = messages * edge_mask.unsqueeze(-1)
        
        # Aggregate messages to target nodes (sum aggregation)
        aggregated = torch.zeros_like(node_features)
        tgt_idx_expanded = tgt_idx.unsqueeze(-1).expand(-1, -1, D)
        aggregated.scatter_add_(1, tgt_idx_expanded, messages)
        
        # Count for normalization
        counts = torch.zeros(B, num_nodes, 1, device=node_features.device)
        ones = edge_mask.unsqueeze(-1)
        counts.scatter_add_(1, tgt_idx_expanded[:, :, :1], ones)
        counts = counts.clamp(min=1)
        
        # Normalize
        aggregated = aggregated / counts
        
        # Residual connection and norm
        node_features = self.norm(node_features + self.dropout(aggregated))
        
        # FFN
        node_features = self.ffn_norm(node_features + self.ffn(node_features))
        
        return node_features


class QuestionEncoder(nn.Module):
    """Encode questions using pretrained transformer."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 256,
        freeze: bool = True,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.projection = nn.Linear(self.hidden_size, output_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns pooled question embedding [B, output_dim]."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden * mask_expanded).sum(dim=1)
        pooled = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        return self.projection(pooled)


class GNNRetriever(pl.LightningModule):
    """
    GNN-based Path Retriever.
    
    Architecture:
    1. Encode question with SBERT
    2. Build KG graph representation  
    3. Run L layers of GNN message passing
    4. Use question to attend over relation embeddings
    5. Predict relation sequence via autoregressive decoding
    """
    
    def __init__(
        self,
        num_relations: int,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        num_heads: int = 8,
        max_path_length: int = 8,
        num_entity_buckets: int = 50000,
        dropout: float = 0.1,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        question_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_question_encoder: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.max_path_length = max_path_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Special tokens
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.EOS_IDX = 2
        self.RELATION_OFFSET = 3
        
        # Question encoder
        self.question_encoder = QuestionEncoder(
            model_name=question_encoder_name,
            output_dim=hidden_dim,
            freeze=freeze_question_encoder,
        )
        
        # Relation embeddings
        self.relation_embedding = nn.Embedding(num_relations + 3, hidden_dim)
        
        # Entity hash embeddings
        self.entity_embedding = nn.Embedding(num_entity_buckets + 1, hidden_dim // 2)
        
        # Entity projection (from hash to full dim)
        self.entity_projection = nn.Linear(hidden_dim // 2, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])
        
        # Question-conditioned relation scorer
        self.relation_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Autoregressive decoder for path generation
        self.path_decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, num_relations + 3)
    
    def encode_kg(
        self,
        relation_ids: torch.Tensor,      # [B, num_edges]
        head_hash_ids: torch.Tensor,     # [B, num_edges]
        tail_hash_ids: torch.Tensor,     # [B, num_edges]
        triple_mask: torch.Tensor,       # [B, num_edges]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode KG and build graph structure.
        
        Returns:
            relation_embs: [B, num_edges, hidden_dim]
            aggregated_kg: [B, hidden_dim] - pooled KG representation
        """
        B, num_edges = relation_ids.shape
        
        # Embed relations
        rel_emb = self.relation_embedding(relation_ids)  # [B, E, D]
        
        # Embed entities
        head_emb = self.entity_embedding(head_hash_ids)  # [B, E, D/2]
        tail_emb = self.entity_embedding(tail_hash_ids)  # [B, E, D/2]
        
        head_emb = self.entity_projection(head_emb)  # [B, E, D]
        tail_emb = self.entity_projection(tail_emb)  # [B, E, D]
        
        # Combine for edge representation
        edge_features = rel_emb + head_emb + tail_emb  # Simple sum
        edge_features = edge_features * triple_mask.unsqueeze(-1)
        
        # Mean pooling over edges for KG representation
        sum_features = edge_features.sum(dim=1)
        count = triple_mask.sum(dim=1, keepdim=True).clamp(min=1)
        aggregated_kg = sum_features / count
        
        return rel_emb, aggregated_kg
    
    def forward(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        kg_relation_ids: torch.Tensor,
        kg_head_hash_ids: torch.Tensor,
        kg_tail_hash_ids: torch.Tensor,
        kg_triple_mask: torch.Tensor,
        target_relations: torch.Tensor,
        path_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with teacher forcing."""
        B = question_input_ids.size(0)
        device = question_input_ids.device
        
        # Encode question
        question_emb = self.question_encoder(
            question_input_ids, question_attention_mask
        )  # [B, D]
        
        # Encode KG
        rel_emb, kg_emb = self.encode_kg(
            kg_relation_ids, kg_head_hash_ids, kg_tail_hash_ids, kg_triple_mask
        )
        
        # Combine question and KG for context
        context = question_emb + kg_emb  # [B, D]
        
        # Teacher forcing: embed target relations
        target_emb = self.relation_embedding(target_relations)  # [B, L, D]
        
        # Decode path autoregressively
        # Shift targets for input (use context as initial)
        decoder_input = torch.cat([
            context.unsqueeze(1),  # [B, 1, D]
            target_emb[:, :-1, :],  # [B, L-1, D]
        ], dim=1)  # [B, L, D]
        
        hidden = context.unsqueeze(0).expand(2, -1, -1).contiguous()  # [2, B, D]
        output, _ = self.path_decoder(decoder_input, hidden)  # [B, L, D]
        
        # Project to vocabulary
        logits = self.output_projection(output)  # [B, L, num_relations+3]
        
        # Compute loss
        if path_mask is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_relations.view(-1)
            mask_flat = path_mask.view(-1).bool()
            
            loss = F.cross_entropy(
                logits_flat[mask_flat],
                targets_flat[mask_flat],
                ignore_index=self.PAD_IDX,
            )
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_relations.view(-1),
                ignore_index=self.PAD_IDX,
            )
        
        return {'loss': loss, 'logits': logits}
    
    @torch.no_grad()
    def generate(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        kg_relation_ids: torch.Tensor,
        kg_head_hash_ids: torch.Tensor,
        kg_tail_hash_ids: torch.Tensor,
        kg_triple_mask: torch.Tensor,
        max_length: int = 8,
    ) -> torch.Tensor:
        """Generate paths autoregressively."""
        B = question_input_ids.size(0)
        device = question_input_ids.device
        
        # Encode question and KG
        question_emb = self.question_encoder(
            question_input_ids, question_attention_mask
        )
        _, kg_emb = self.encode_kg(
            kg_relation_ids, kg_head_hash_ids, kg_tail_hash_ids, kg_triple_mask
        )
        
        context = question_emb + kg_emb
        hidden = context.unsqueeze(0).expand(2, -1, -1).contiguous()
        
        # Autoregressive generation
        generated = []
        current_input = context.unsqueeze(1)  # [B, 1, D]
        
        for _ in range(max_length):
            output, hidden = self.path_decoder(current_input, hidden)
            logits = self.output_projection(output[:, -1, :])  # [B, vocab]
            next_token = logits.argmax(dim=-1)  # [B]
            generated.append(next_token)
            
            current_input = self.relation_embedding(next_token).unsqueeze(1)
        
        return torch.stack(generated, dim=1)  # [B, max_length]
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            kg_relation_ids=batch['kg_relation_ids'],
            kg_head_hash_ids=batch['kg_head_hash_ids'],
            kg_tail_hash_ids=batch['kg_tail_hash_ids'],
            kg_triple_mask=batch['kg_triple_mask'],
            target_relations=batch['target_relations'],
            path_mask=batch.get('path_mask'),
        )
        
        self.log('train/loss', outputs['loss'], prog_bar=True)
        return outputs['loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            kg_relation_ids=batch['kg_relation_ids'],
            kg_head_hash_ids=batch['kg_head_hash_ids'],
            kg_tail_hash_ids=batch['kg_tail_hash_ids'],
            kg_triple_mask=batch['kg_triple_mask'],
            target_relations=batch['target_relations'],
            path_mask=batch.get('path_mask'),
        )
        
        self.log('val/loss', outputs['loss'], prog_bar=True, sync_dist=True)
        return outputs['loss']
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            kg_relation_ids=batch['kg_relation_ids'],
            kg_head_hash_ids=batch['kg_head_hash_ids'],
            kg_tail_hash_ids=batch['kg_tail_hash_ids'],
            kg_triple_mask=batch['kg_triple_mask'],
            target_relations=batch['target_relations'],
            path_mask=batch.get('path_mask'),
        )
        
        self.log('test/loss', outputs['loss'], prog_bar=True, sync_dist=True)
        return outputs['loss']
    
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
    # Quick test
    model = GNNRetriever(num_relations=5000, hidden_dim=256, num_gnn_layers=3)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
