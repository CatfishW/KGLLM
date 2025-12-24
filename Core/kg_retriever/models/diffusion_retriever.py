"""
KG-Conditioned Diffusion Retriever for Path Generation.

Architecture:
1. Question Encoder: Frozen MiniLM for question embedding
2. KG Encoder: Linear compression of KG triples into fixed-size features
3. Diffusion Transformer: Generates relation paths conditioned on question + KG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple, List, Any
import math


class KGEncoder(nn.Module):
    """
    Efficiently encode knowledge graph triples into a fixed-size feature vector.
    
    Optimized approach for FULL KG:
    1. Each triple (h, r, t) → linear embed: rel_emb + head_hash + tail_hash
    2. Fast mean pooling over all triples (no attention for speed)
    3. Linear projection to get KG feature vector
    
    This handles 10K+ triples efficiently.
    """
    
    def __init__(
        self,
        num_relations: int,
        hidden_dim: int = 256,
        num_entity_buckets: int = 50000,  # Larger for full KG
        kg_feature_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kg_feature_dim = kg_feature_dim
        self.num_entity_buckets = num_entity_buckets
        
        # Lightweight embeddings
        emb_dim = hidden_dim // 2  # Smaller for efficiency
        
        # Relation embeddings
        self.relation_embedding = nn.Embedding(num_relations + 2, emb_dim)  # +2 for PAD, UNK
        
        # Entity hash embeddings (lightweight)
        self.entity_embedding = nn.Embedding(num_entity_buckets + 1, emb_dim // 2)  # +1 for PAD
        
        # Fast triple encoder: single linear layer
        triple_input_dim = emb_dim + 2 * (emb_dim // 2)  # rel + head + tail
        self.triple_projection = nn.Linear(triple_input_dim, hidden_dim)
        
        # Final KG feature projection with LayerNorm
        self.kg_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, kg_feature_dim),
            nn.GELU(),
            nn.Linear(kg_feature_dim, kg_feature_dim),
        )
    
    def forward(
        self,
        relation_ids: torch.Tensor,      # [B, num_triples]
        head_hash_ids: torch.Tensor,     # [B, num_triples]
        tail_hash_ids: torch.Tensor,     # [B, num_triples]
        triple_mask: torch.Tensor,       # [B, num_triples] - 1 for valid, 0 for padding
    ) -> torch.Tensor:
        """
        Encode KG triples into a fixed-size feature vector.
        
        Optimized for full KG with 10K+ triples.
        
        Returns:
            kg_features: [B, kg_feature_dim]
        """
        # Embed relations and entity hashes
        rel_emb = self.relation_embedding(relation_ids)       # [B, T, emb_dim]
        head_emb = self.entity_embedding(head_hash_ids)       # [B, T, emb_dim//2]
        tail_emb = self.entity_embedding(tail_hash_ids)       # [B, T, emb_dim//2]
        
        # Concatenate and project triples (fast linear)
        triple_input = torch.cat([rel_emb, head_emb, tail_emb], dim=-1)
        triple_features = self.triple_projection(triple_input)  # [B, T, hidden_dim]
        
        # Apply mask and mean pooling (fast, handles variable length)
        triple_mask_expanded = triple_mask.unsqueeze(-1)  # [B, T, 1]
        masked_features = triple_features * triple_mask_expanded
        
        # Sum and normalize by count
        sum_features = masked_features.sum(dim=1)  # [B, hidden_dim]
        count = triple_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        pooled = sum_features / count  # [B, hidden_dim]
        
        # Project to KG feature
        kg_features = self.kg_projection(pooled)  # [B, kg_feature_dim]
        
        return kg_features


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
        
        # Project to output dimension
        self.projection = nn.Linear(self.hidden_size, output_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sequence_output: [B, seq_len, output_dim]
            pooled_output: [B, output_dim]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, seq_len, hidden_size]
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden * mask_expanded).sum(dim=1)
        pooled = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        # Project
        sequence_output = self.projection(hidden)
        pooled_output = self.projection(pooled)
        
        return sequence_output, pooled_output


class DiffusionTransformer(nn.Module):
    """
    Transformer for discrete diffusion over relation sequences.
    Conditioned on question + KG features.
    """
    
    def __init__(
        self,
        num_relations: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_path_length: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.max_path_length = max_path_length
        
        # Relation embedding for path positions
        self.relation_embedding = nn.Embedding(num_relations + 3, hidden_dim)  # +3 for PAD, UNK, MASK
        self.position_embedding = nn.Embedding(max_path_length, hidden_dim)
        
        # Timestep embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Condition projection (question + KG → hidden)
        self.condition_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, num_relations + 3)
        
        # Special token indices
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.MASK_IDX = 2
    
    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embedding(emb)
    
    def forward(
        self,
        noisy_relations: torch.Tensor,  # [B, path_length]
        timesteps: torch.Tensor,         # [B]
        question_features: torch.Tensor, # [B, hidden_dim]
        kg_features: torch.Tensor,       # [B, hidden_dim]
        path_mask: Optional[torch.Tensor] = None,  # [B, path_length]
    ) -> torch.Tensor:
        """
        Predict clean relation logits from noisy input.
        
        Returns:
            logits: [B, path_length, num_relations + 3]
        """
        B, path_length = noisy_relations.shape
        
        # Embed noisy relations + positions
        rel_emb = self.relation_embedding(noisy_relations)  # [B, L, hidden]
        pos_ids = torch.arange(path_length, device=noisy_relations.device)
        pos_emb = self.position_embedding(pos_ids).unsqueeze(0)  # [1, L, hidden]
        
        # Timestep embedding
        time_emb = self.get_timestep_embedding(timesteps)  # [B, hidden]
        
        # Condition on question + KG
        condition = torch.cat([question_features, kg_features], dim=-1)  # [B, hidden*2]
        condition = self.condition_projection(condition)  # [B, hidden]
        
        # Combine embeddings
        x = rel_emb + pos_emb + time_emb.unsqueeze(1) + condition.unsqueeze(1)
        
        # Causal mask for autoregressive-style generation
        causal_mask = torch.triu(
            torch.ones(path_length, path_length, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        
        # Transformer
        x = self.transformer(x, mask=causal_mask)
        
        # Output logits
        logits = self.output_projection(x)  # [B, L, num_relations + 3]
        
        return logits


class KGDiffusionRetriever(pl.LightningModule):
    """
    Full model for KG-conditioned path retrieval using diffusion.
    
    Input: Question + Knowledge Graph
    Output: Sequence of relations forming the reasoning path
    """
    
    def __init__(
        self,
        num_relations: int,
        hidden_dim: int = 256,
        kg_feature_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_diffusion_steps: int = 20,
        max_path_length: int = 8,
        num_entity_buckets: int = 10000,
        dropout: float = 0.1,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        question_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_question_encoder: bool = True,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_diffusion_steps = num_diffusion_steps
        self.max_path_length = max_path_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.label_smoothing = label_smoothing
        
        # Question encoder
        self.question_encoder = QuestionEncoder(
            model_name=question_encoder_name,
            output_dim=hidden_dim,
            freeze=freeze_question_encoder,
        )
        
        # KG encoder
        self.kg_encoder = KGEncoder(
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            num_entity_buckets=num_entity_buckets,
            kg_feature_dim=kg_feature_dim,
        )
        
        # Diffusion transformer
        self.diffusion = DiffusionTransformer(
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_path_length=max_path_length,
            dropout=dropout,
        )
        
        # Special tokens
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.MASK_IDX = 2
    
    def add_noise(
        self,
        clean_relations: torch.Tensor,  # [B, L]
        timesteps: torch.Tensor,         # [B]
    ) -> torch.Tensor:
        """
        Add noise to clean relation sequence for forward diffusion.
        
        Uses discrete masking: randomly replace tokens with MASK based on timestep.
        """
        B, L = clean_relations.shape
        
        # Noise probability increases with timestep
        noise_prob = timesteps.float() / self.num_diffusion_steps  # [B]
        noise_prob = noise_prob.unsqueeze(-1).expand(-1, L)  # [B, L]
        
        # Random mask
        mask = torch.rand_like(noise_prob) < noise_prob
        
        # Replace with MASK token
        noisy = clean_relations.clone()
        noisy[mask] = self.MASK_IDX
        
        return noisy
    
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
        """
        Training forward pass.
        """
        B = question_input_ids.size(0)
        device = question_input_ids.device
        
        # Encode question
        _, question_features = self.question_encoder(
            question_input_ids, question_attention_mask
        )  # [B, hidden_dim]
        
        # Encode KG
        kg_features = self.kg_encoder(
            kg_relation_ids, kg_head_hash_ids, kg_tail_hash_ids, kg_triple_mask
        )  # [B, kg_feature_dim]
        
        # Sample random timesteps
        timesteps = torch.randint(1, self.num_diffusion_steps + 1, (B,), device=device)
        
        # Add noise to target relations
        noisy_relations = self.add_noise(target_relations, timesteps)
        
        # Predict clean relations
        logits = self.diffusion(
            noisy_relations, timesteps, question_features, kg_features, path_mask
        )  # [B, L, num_relations + 3]
        
        # Compute loss
        if path_mask is not None:
            # Only compute loss on valid positions
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_relations.view(-1)
            mask_flat = path_mask.view(-1).bool()
            
            loss = F.cross_entropy(
                logits_flat[mask_flat],
                targets_flat[mask_flat],
                ignore_index=self.PAD_IDX,
                label_smoothing=self.label_smoothing,
            )
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_relations.view(-1),
                ignore_index=self.PAD_IDX,
                label_smoothing=self.label_smoothing,
            )
        
        return {
            'loss': loss,
            'logits': logits,
        }
    
    @torch.no_grad()
    def generate(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        kg_relation_ids: torch.Tensor,
        kg_head_hash_ids: torch.Tensor,
        kg_tail_hash_ids: torch.Tensor,
        kg_triple_mask: torch.Tensor,
        path_length: int = 4,
        num_samples: int = 3,
    ) -> torch.Tensor:
        """
        Generate relation paths via iterative denoising.
        
        Returns:
            generated_paths: [B, num_samples, path_length]
        """
        B = question_input_ids.size(0)
        device = question_input_ids.device
        
        # Encode question and KG
        _, question_features = self.question_encoder(
            question_input_ids, question_attention_mask
        )
        kg_features = self.kg_encoder(
            kg_relation_ids, kg_head_hash_ids, kg_tail_hash_ids, kg_triple_mask
        )
        
        # Expand for multiple samples
        question_features = question_features.unsqueeze(1).expand(-1, num_samples, -1)
        question_features = question_features.reshape(B * num_samples, -1)
        kg_features = kg_features.unsqueeze(1).expand(-1, num_samples, -1)
        kg_features = kg_features.reshape(B * num_samples, -1)
        
        # Start with all MASK tokens
        x = torch.full(
            (B * num_samples, path_length),
            self.MASK_IDX,
            dtype=torch.long,
            device=device,
        )
        
        # Iterative denoising
        for t in range(self.num_diffusion_steps, 0, -1):
            timesteps = torch.full((B * num_samples,), t, dtype=torch.long, device=device)
            
            logits = self.diffusion(x, timesteps, question_features, kg_features)
            
            # Sample from logits
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B * num_samples, -1)
            
            # Only update MASK positions
            mask_positions = (x == self.MASK_IDX)
            
            # Probabilistically unmask based on timestep
            unmask_prob = 1.0 / t
            unmask = torch.rand_like(x.float()) < unmask_prob
            update = mask_positions & unmask
            
            x = torch.where(update, sampled, x)
        
        # Final pass: fill any remaining MASK tokens
        timesteps = torch.ones((B * num_samples,), dtype=torch.long, device=device)
        logits = self.diffusion(x, timesteps, question_features, kg_features)
        final = logits.argmax(dim=-1)
        x = torch.where(x == self.MASK_IDX, final, x)
        
        return x.view(B, num_samples, path_length)
    
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
        # Separate parameters for different learning rates
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
        
        # Linear warmup then cosine decay
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }


if __name__ == '__main__':
    # Quick test
    model = KGDiffusionRetriever(num_relations=5000, hidden_dim=256)
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    B, num_triples, path_len = 2, 100, 4
    dummy_batch = {
        'question_input_ids': torch.randint(0, 1000, (B, 32)),
        'question_attention_mask': torch.ones(B, 32),
        'kg_relation_ids': torch.randint(0, 5000, (B, num_triples)),
        'kg_head_hash_ids': torch.randint(0, 10000, (B, num_triples)),
        'kg_tail_hash_ids': torch.randint(0, 10000, (B, num_triples)),
        'kg_triple_mask': torch.ones(B, num_triples),
        'target_relations': torch.randint(3, 5000, (B, path_len)),
        'path_mask': torch.ones(B, path_len),
    }
    
    outputs = model(**dummy_batch)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
