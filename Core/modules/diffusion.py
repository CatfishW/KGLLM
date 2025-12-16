"""
Discrete Diffusion Model for Path Generation.

Based on D3PM (Discrete Denoising Diffusion Probabilistic Models) adapted
for generating sequences of (entity, relation) pairs as reasoning paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from einops import rearrange


def generate_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate causal attention mask.
    
    Returns a boolean mask where True means "block this attention".
    Position i can only attend to positions 0..i (inclusive).
    
    Args:
        seq_len: Length of sequence
        device: Device to create mask on
        
    Returns:
        [seq_len, seq_len] boolean mask (True = blocked)
    """
    # Upper triangular matrix (excluding diagonal) = positions that should be blocked
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PathTransformerBlock(nn.Module):
    """Transformer block for path denoising."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # AdaLN modulation for conditioning on time
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get modulation parameters from timestep
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(time_emb).chunk(6, dim=-1)
        
        # Self-attention with AdaLN
        # Force float32 for LayerNorm to avoid instability in mixed precision
        with torch.cuda.amp.autocast(enabled=False):
            x_norm = self.norm1(x.float()).type_as(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with AdaLN
        with torch.cuda.amp.autocast(enabled=False):
            x_norm = self.norm2(x.float()).type_as(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention for conditioning on question and graph."""
    
    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(context_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True
        )
        
        self.proj = nn.Linear(context_dim, hidden_dim) if context_dim != hidden_dim else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_norm = self.norm_q(x)
        context_norm = self.norm_kv(context)
        
        attn_out, _ = self.cross_attn(
            x_norm, 
            context_norm, 
            context_norm,
            key_padding_mask=context_mask
        )
        
        return x + attn_out


class PathDiffusionTransformer(nn.Module):
    """
    Transformer-based denoising network for discrete path diffusion.
    
    Takes noisy path tokens and predicts the clean path.
    Conditioned on: question encoding, timestep.
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        question_dim: int = 256,  # Dimension of question encoder output
        max_path_length: int = 20,
        dropout: float = 0.1,
        predict_entities: bool = True,
        use_causal_attention: bool = True  # Enable causal masking for autoregressive-style generation
    ):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        self.predict_entities = predict_entities
        self.use_causal_attention = use_causal_attention
        
        # Entity and relation embeddings for path tokens
        if predict_entities:
            self.entity_embedding = nn.Embedding(num_entities, hidden_dim, padding_idx=0)
        else:
            self.entity_embedding = None
            
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim, padding_idx=0)
        
        # Position embedding for sequence positions
        # If predict_entities is False, sequence length is roughly half (just relations)
        self.pos_embedding = nn.Embedding(max_path_length * 2, hidden_dim)
        
        # Token type embedding (0=entity, 1=relation)
        if predict_entities:
            self.type_embedding = nn.Embedding(2, hidden_dim)
        else:
            self.type_embedding = None
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Project question to hidden_dim
        self.question_proj = nn.Linear(question_dim, hidden_dim)
        
        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                'self_attn': PathTransformerBlock(hidden_dim, num_heads, dropout=dropout),
                'cross_attn_q': CrossAttentionBlock(hidden_dim, hidden_dim, num_heads, dropout)
            }))
        
        # Output heads
        self.norm_out = nn.LayerNorm(hidden_dim)
        
        if predict_entities:
            self.entity_head = nn.Linear(hidden_dim, num_entities)
        else:
            self.entity_head = None
            
        self.relation_head = nn.Linear(hidden_dim, num_relations)
    
    def forward(
        self,
        noisy_entities: Optional[torch.Tensor],  # [B, L] or None
        noisy_relations: torch.Tensor,  # [B, L-1]
        timesteps: torch.Tensor,  # [B]
        question_encoding: torch.Tensor,  # [B, seq_len, question_dim]
        path_mask: Optional[torch.Tensor] = None,  # [B, L]
        question_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Predict clean path from noisy path.
        
        Returns:
            entity_logits: [B, L, num_entities] or None
            relation_logits: [B, L-1, num_relations]
        """
        B = noisy_relations.shape[0]
        device = noisy_relations.device
        
        # Embed relations
        relation_emb = self.relation_embedding(noisy_relations)  # [B, L-1, D]
        
        if self.predict_entities and noisy_entities is not None:
            L = noisy_entities.shape[1]
            # Embed entities
            entity_emb = self.entity_embedding(noisy_entities)  # [B, L, D]
            
            # Interleave entities and relations: e0, r0, e1, r1, ..., eL
            seq_len = 2 * L - 1
            x = torch.zeros(B, seq_len, self.hidden_dim, device=device)
            
            # Place entities at even positions
            x[:, 0::2, :] = entity_emb
            # Place relations at odd positions
            x[:, 1::2, :] = relation_emb
            
            # Add type embeddings
            types = torch.zeros(seq_len, dtype=torch.long, device=device)
            types[0::2] = 0  # entities
            types[1::2] = 1  # relations
            x = x + self.type_embedding(types).unsqueeze(0)
            
            # Create mask
            if path_mask is not None:
                seq_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
                seq_mask[:, 0::2] = ~path_mask
                seq_mask[:, 1::2] = ~path_mask[:, :-1]
            else:
                seq_mask = None
                
        else:
            # Relation only mode
            x = relation_emb
            seq_len = x.shape[1]
            
            # Create mask (using relation part of path_mask)
            if path_mask is not None:
                # path_mask is [B, L] (entities). Relations are L-1.
                # We assume path_mask corresponds to valid entities.
                # Valid relations are one less.
                seq_mask = ~path_mask[:, :-1].bool()
                
                # Fix for causal attention: ensure first position is always valid
                # When all positions are masked combined with causal mask, softmax produces NaN
                # Setting first position to False (not padding) ensures at least self-attention works
                seq_mask[:, 0] = False
            else:
                seq_mask = None
        
        # Add position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        # Clamp to avoid index error if length changed
        positions = positions.clamp(max=self.pos_embedding.num_embeddings - 1)
        x = x + self.pos_embedding(positions)
        
        # Time embedding
        time_emb = self.time_mlp(timesteps)  # [B, D]
        
        # Project context
        question_ctx = self.question_proj(question_encoding)
        
        # Generate causal attention mask if enabled
        if self.use_causal_attention:
            causal_mask = generate_causal_mask(seq_len, device=device)
        else:
            causal_mask = None
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block['self_attn'](x, time_emb, key_padding_mask=seq_mask, attn_mask=causal_mask)
            x = block['cross_attn_q'](x, question_ctx, context_mask=question_mask)
        
        x = self.norm_out(x)
        
        if self.predict_entities:
            # Extract entity and relation representations
            entity_repr = x[:, 0::2, :]  # [B, L, D]
            relation_repr = x[:, 1::2, :]  # [B, L-1, D]
            
            entity_logits = self.entity_head(entity_repr)
            relation_logits = self.relation_head(relation_repr)
        else:
            entity_logits = None
            relation_logits = self.relation_head(x)
        
        return entity_logits, relation_logits


class DiscreteDiffusion(nn.Module):
    """
    Discrete Diffusion Process for path generation.
    
    Implements forward diffusion (adding noise) and training objective
    for the denoising network.
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_timesteps: int = 1000,
        noise_schedule: str = 'cosine'
    ):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_timesteps = num_timesteps
        
        # Create noise schedule (probability of masking at each timestep)
        if noise_schedule == 'cosine':
            # Cosine schedule - slower corruption at start
            steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
            alpha_bar = torch.cos(((steps / num_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = torch.clamp(betas, max=0.999)
        else:  # linear
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        
        # Mask token indices (from vocab: entity MASK=4, relation MASK=2)
        # These are fixed indices based on EntityRelationVocab definition
        self.entity_mask_idx = 4  # "<MASK>" token index for entities
        self.relation_mask_idx = 2  # "<MASK>" token index for relations
    
    def q_sample(
        self,
        x_entities: torch.Tensor,
        x_relations: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: corrupt clean data by randomly masking tokens.
        
        At timestep t, each token is masked with probability 1 - alpha_cumprod[t].
        """
        B = x_entities.shape[0]
        device = x_entities.device
        
        # Get corruption probability for each sample
        mask_prob = 1 - self.alpha_cumprod[t]  # [B]
        
        # Random mask for entities
        entity_mask = torch.rand_like(x_entities.float()) < mask_prob.unsqueeze(1)
        noisy_entities = x_entities.clone()
        noisy_entities[entity_mask] = self.entity_mask_idx
        
        # Random mask for relations
        relation_mask = torch.rand_like(x_relations.float()) < mask_prob.unsqueeze(1)
        noisy_relations = x_relations.clone()
        noisy_relations[relation_mask] = self.relation_mask_idx
        
        return noisy_entities, noisy_relations
    
    def compute_loss(
        self,
        entity_logits: Optional[torch.Tensor],
        relation_logits: torch.Tensor,
        target_entities: torch.Tensor,
        target_relations: torch.Tensor,
        path_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cross-entropy loss for denoising prediction.
        """
        # Always compute the loss in float32 to avoid FP16 overflow under AMP
        relation_logits_flat = relation_logits.view(-1, self.num_relations).float()
        target_relations_flat = target_relations.view(-1)
        
        def _safe_cross_entropy(logits, targets, mask=None, ignore_indices=None):
            """Compute CE that gracefully handles the all-masked case."""
            if mask is not None:
                mask = mask.bool()
                if not mask.any():
                    return logits.new_zeros(())
                logits = logits[mask]
                targets = targets[mask]
            
            # Ignore padding (0) and mask tokens (2 for relations, 4 for entities)
            if ignore_indices is None:
                ignore_indices = [0]  # Default: only ignore padding
            # Filter out ignored indices from targets
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            for idx in ignore_indices:
                valid_mask = valid_mask & (targets != idx)
            
            if not valid_mask.any():
                return logits.new_zeros(())
            
            logits = logits[valid_mask]
            targets = targets[valid_mask]
            return F.cross_entropy(logits, targets)
        
        if path_mask is not None:
            relation_mask_flat = path_mask[:, :-1].reshape(-1)
            relation_loss = _safe_cross_entropy(
                relation_logits_flat, target_relations_flat, relation_mask_flat,
                ignore_indices=[0, 2]  # Ignore PAD (0) and MASK (2)
            )
        else:
            relation_loss = _safe_cross_entropy(
                relation_logits_flat, target_relations_flat,
                ignore_indices=[0, 2]  # Ignore PAD (0) and MASK (2)
            )
            
        if entity_logits is not None:
            entity_logits_flat = entity_logits.view(-1, self.num_entities).float()
            target_entities_flat = target_entities.view(-1)
            
            if path_mask is not None:
                entity_mask_flat = path_mask.view(-1)
                entity_loss = _safe_cross_entropy(
                    entity_logits_flat, target_entities_flat, entity_mask_flat,
                    ignore_indices=[0, 4]  # Ignore PAD (0) and MASK (4)
                )
            else:
                entity_loss = _safe_cross_entropy(
                    entity_logits_flat, target_entities_flat,
                    ignore_indices=[0, 4]  # Ignore PAD (0) and MASK (4)
                )
        else:
            entity_loss = torch.tensor(0.0, device=relation_logits.device)
        
        total_loss = entity_loss + relation_loss
        
        return {
            'loss': total_loss,
            'entity_loss': entity_loss,
            'relation_loss': relation_loss
        }
    
    @torch.no_grad()
    def sample(
        self,
        denoiser: PathDiffusionTransformer,
        question_encoding: torch.Tensor,
        path_length: int,
        question_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate paths by iterative denoising.
        
        Starts from fully masked sequence and gradually unmasks.
        """
        B = question_encoding.shape[0]
        device = question_encoding.device
        
        # Start with fully masked sequence
        # Only initialize entities if we're predicting them
        if denoiser.predict_entities:
            entities = torch.full((B, path_length), self.entity_mask_idx, device=device)
        else:
            # If not predicting entities, use a dummy tensor (will be ignored)
            entities = torch.zeros((B, path_length), dtype=torch.long, device=device)
        relations = torch.full((B, path_length - 1), self.relation_mask_idx, device=device)
        
        # Iteratively denoise
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Get predictions
            entity_logits, relation_logits = denoiser(
                entities if denoiser.predict_entities else None, 
                relations, 
                t_batch,
                question_encoding,
                question_mask=question_mask
            )
            
            # Check and fix NaN/inf in logits
            if torch.isnan(relation_logits).any() or torch.isinf(relation_logits).any():
                relation_logits = torch.where(
                    torch.isnan(relation_logits) | torch.isinf(relation_logits),
                    torch.zeros_like(relation_logits),
                    relation_logits
                )
            
            if entity_logits is not None:
                if torch.isnan(entity_logits).any() or torch.isinf(entity_logits).any():
                    entity_logits = torch.where(
                        torch.isnan(entity_logits) | torch.isinf(entity_logits),
                        torch.zeros_like(entity_logits),
                        entity_logits
                    )
            
            # Sample from predictions (with temperature)
            if temperature > 0:
                # Clamp logits to prevent numerical instability
                relation_logits_clamped = torch.clamp(relation_logits, min=-50.0, max=50.0)
                
                relation_probs = F.softmax(relation_logits_clamped / temperature, dim=-1)
                
                # Check for NaN or inf and replace with uniform distribution if needed
                if torch.isnan(relation_probs).any() or torch.isinf(relation_probs).any():
                    # Fall back to uniform distribution if probabilities are invalid
                    relation_probs = torch.ones_like(relation_probs) / self.num_relations
                
                # Ensure probabilities are valid (non-negative, sum to 1)
                relation_probs = torch.clamp(relation_probs, min=1e-8, max=1.0)
                relation_probs = relation_probs / relation_probs.sum(dim=-1, keepdim=True)
                
                relation_samples = torch.multinomial(
                    relation_probs.view(-1, self.num_relations), 1
                ).view(B, path_length - 1)
                
                if entity_logits is not None:
                    # Clamp logits to prevent numerical instability
                    entity_logits_clamped = torch.clamp(entity_logits, min=-50.0, max=50.0)
                    entity_probs = F.softmax(entity_logits_clamped / temperature, dim=-1)
                    
                    # Check for NaN or inf and replace with uniform distribution if needed
                    if torch.isnan(entity_probs).any() or torch.isinf(entity_probs).any():
                        # Fall back to uniform distribution if probabilities are invalid
                        entity_probs = torch.ones_like(entity_probs) / self.num_entities
                    
                    # Ensure probabilities are valid (non-negative, sum to 1)
                    entity_probs = torch.clamp(entity_probs, min=1e-8, max=1.0)
                    entity_probs = entity_probs / entity_probs.sum(dim=-1, keepdim=True)
                    
                    entity_samples = torch.multinomial(
                        entity_probs.view(-1, self.num_entities), 1
                    ).view(B, path_length)
            else:
                relation_samples = relation_logits.argmax(dim=-1)
                if entity_logits is not None:
                    entity_samples = entity_logits.argmax(dim=-1)
            
            # Determine which positions to update based on schedule
            if t > 0:
                # Gradually unmask
                unmask_prob = 1 - self.alpha_cumprod[t-1] / self.alpha_cumprod[t]
                relation_unmask = torch.rand(B, path_length - 1, device=device) < unmask_prob
                
                relations = torch.where(
                    relation_unmask & (relations == self.relation_mask_idx),
                    relation_samples,
                    relations
                )
                
                if entity_logits is not None:
                    entity_unmask = torch.rand(B, path_length, device=device) < unmask_prob
                    entities = torch.where(
                        entity_unmask & (entities == self.entity_mask_idx),
                        entity_samples,
                        entities
                    )
            else:
                # Final step: unmask everything
                relations = relation_samples
                if entity_logits is not None:
                    entities = entity_samples
        
        # If not predicting entities, return empty entities (all zeros)
        if not denoiser.predict_entities:
            entities = torch.zeros_like(entities)
        
        # Post-process: trim padding and stop at PAD tokens
        # PAD token index is 0 for both entities and relations
        pad_idx = 0
        
        # For relations: find first PAD token and trim
        for b in range(B):
            rel_seq = relations[b]
            # Find first PAD token
            pad_positions = (rel_seq == pad_idx).nonzero(as_tuple=True)[0]
            if len(pad_positions) > 0:
                # Trim at first PAD
                first_pad = pad_positions[0].item()
                relations[b, first_pad:] = pad_idx
        
        return entities, relations

