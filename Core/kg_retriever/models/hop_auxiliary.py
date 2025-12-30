"""
Hop-wise Auxiliary Loss Module for Multi-Hop Path Reasoning.

Provides supervision at each hop level, encouraging the model to correctly
identify intermediate relations in multi-hop paths, not just the final answer.

Key Innovation: Progressive hop supervision with curriculum-style weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class HopAuxiliaryLoss(nn.Module):
    """
    Computes hop-wise auxiliary losses for multi-hop path reasoning.
    
    Core Idea: Supervise intermediate path prefixes to encourage
    correct early hop identification. This helps with:
    1. Multi-hop reasoning accuracy
    2. Path interpretability
    3. Gradient flow to early layers
    
    Loss at hop k: CE(score(path[:k]), label)
    Total: Σ λ_k * L_hop_k
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        max_hops: int = 4,
        loss_weights: Optional[List[float]] = None,
        temperature: float = 1.0,
        loss_type: str = 'cross_entropy',  # 'cross_entropy', 'margin', 'plackett_luce'
        margin: float = 1.0,
        cumulative: bool = True,  # If True, hop_k loss uses cumulative score up to hop k
    ):
        """
        Args:
            hidden_dim: Dimension of embeddings
            max_hops: Maximum number of hops to supervise
            loss_weights: Weights for each hop loss [λ_1, λ_2, ..., λ_max_hops]
                         If None, uses progressive weighting
            temperature: Temperature for softmax
            loss_type: Type of loss ('cross_entropy', 'margin', 'plackett_luce')
            margin: Margin for margin loss
            cumulative: If True, use cumulative scores; if False, use per-hop scores
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_hops = max_hops
        self.temperature = temperature
        self.loss_type = loss_type
        self.margin = margin
        self.cumulative = cumulative
        
        # Default: Progressive weighting (later hops matter more)
        if loss_weights is None:
            # [0.1, 0.2, 0.3, 0.4] for 4 hops
            weights = [(i + 1) / sum(range(1, max_hops + 1)) for i in range(max_hops)]
            loss_weights = weights
        
        assert len(loss_weights) == max_hops, f"Expected {max_hops} weights, got {len(loss_weights)}"
        self.register_buffer('loss_weights', torch.tensor(loss_weights))
        
        # Per-hop scoring heads
        self.hop_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * (i + 1) if cumulative else hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )
            for i in range(max_hops)
        ])
    
    def compute_hop_scores(
        self,
        hop_embeddings: torch.Tensor,  # [B, C, max_hops, D]
        question_embed: torch.Tensor,  # [B, D]
        hop_mask: torch.Tensor,        # [B, C, max_hops]
    ) -> torch.Tensor:
        """
        Compute per-hop scores for each candidate path.
        
        Args:
            hop_embeddings: Per-hop embeddings [B, C, max_hops, D]
            question_embed: Pooled question embedding [B, D]
            hop_mask: Which hops are valid [B, C, max_hops]
            
        Returns:
            hop_scores: [B, C, max_hops]
        """
        B, C, max_hops, D = hop_embeddings.shape
        device = hop_embeddings.device
        
        hop_scores = []
        
        for h in range(self.max_hops):
            if self.cumulative:
                # Concatenate embeddings from hop 0 to h
                cumulative_emb = hop_embeddings[:, :, :h+1, :].reshape(B, C, -1)  # [B, C, (h+1)*D]
            else:
                # Just use this hop's embedding
                cumulative_emb = hop_embeddings[:, :, h, :]  # [B, C, D]
            
            # Score with hop-specific head
            score = self.hop_scorers[h](cumulative_emb).squeeze(-1)  # [B, C]
            hop_scores.append(score)
        
        hop_scores = torch.stack(hop_scores, dim=-1)  # [B, C, max_hops]
        
        # Mask invalid hops
        hop_scores = hop_scores.masked_fill(~hop_mask.bool(), -65504.0)
        
        return hop_scores
    
    def forward(
        self,
        hop_embeddings: torch.Tensor,  # [B, C, max_hops, D]
        question_embed: torch.Tensor,  # [B, D]
        labels: torch.Tensor,          # [B] - correct path index
        candidate_mask: torch.Tensor,  # [B, C]
        hop_mask: torch.Tensor,        # [B, C, max_hops]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute hop-wise auxiliary losses.
        
        Args:
            hop_embeddings: Per-hop embeddings [B, C, max_hops, D]
            question_embed: Pooled question embedding [B, D]
            labels: Index of correct path [B]
            candidate_mask: Valid candidates [B, C]
            hop_mask: Valid hops for each candidate [B, C, max_hops]
            
        Returns:
            total_loss: Weighted sum of per-hop losses
            loss_dict: Individual losses for logging
        """
        B, C, max_hops, D = hop_embeddings.shape
        device = hop_embeddings.device
        
        # Compute per-hop scores
        hop_scores = self.compute_hop_scores(hop_embeddings, question_embed, hop_mask)
        
        losses = {}
        total_loss = torch.tensor(0.0, device=device)
        
        for h in range(self.max_hops):
            # Get scores for this hop: [B, C]
            scores_h = hop_scores[:, :, h]
            
            # Apply candidate mask
            scores_h = scores_h.masked_fill(~candidate_mask.bool(), -65504.0)
            
            # Check if this hop is valid for all samples
            # A hop is valid if the correct path has this hop
            correct_hop_valid = hop_mask[torch.arange(B, device=device), labels, h]
            
            if correct_hop_valid.sum() == 0:
                # No valid samples for this hop
                losses[f'hop_{h+1}_loss'] = torch.tensor(0.0, device=device)
                continue
            
            # Scale by temperature
            scores_h = scores_h / self.temperature
            
            if self.loss_type == 'cross_entropy':
                # Standard cross-entropy
                loss_h = F.cross_entropy(scores_h, labels, reduction='none')
                # Only count samples where this hop is valid
                loss_h = (loss_h * correct_hop_valid.float()).sum() / correct_hop_valid.float().sum().clamp(min=1)
                
            elif self.loss_type == 'margin':
                # Margin loss: correct score should exceed others by margin
                batch_idx = torch.arange(B, device=device)
                correct_scores = scores_h[batch_idx, labels]  # [B]
                
                # max(0, margin - (correct - others))
                margin_loss = F.relu(self.margin - (correct_scores.unsqueeze(1) - scores_h))
                margin_loss = margin_loss.masked_fill(~candidate_mask.bool(), 0)
                margin_loss[batch_idx, labels] = 0  # Don't penalize correct position
                
                loss_h = margin_loss.sum(dim=1).mean()
                
            elif self.loss_type == 'plackett_luce':
                # Plackett-Luce listwise loss
                loss_h = F.cross_entropy(scores_h, labels, reduction='none')
                loss_h = (loss_h * correct_hop_valid.float()).sum() / correct_hop_valid.float().sum().clamp(min=1)
            
            losses[f'hop_{h+1}_loss'] = loss_h
            total_loss = total_loss + self.loss_weights[h] * loss_h
        
        losses['total_hop_aux_loss'] = total_loss
        
        # Also compute per-hop accuracy for logging
        with torch.no_grad():
            for h in range(self.max_hops):
                scores_h = hop_scores[:, :, h]
                preds = scores_h.argmax(dim=-1)
                correct_hop_valid = hop_mask[torch.arange(B, device=device), labels, h]
                
                if correct_hop_valid.sum() > 0:
                    acc = ((preds == labels) * correct_hop_valid.float()).sum() / correct_hop_valid.float().sum()
                    losses[f'hop_{h+1}_acc'] = acc
        
        return total_loss, losses


class ProgressiveHopLoss(nn.Module):
    """
    Progressive Hop Loss with Curriculum Learning.
    
    Starts by focusing on early hops, then gradually shifts focus to later hops
    as training progresses. This implements a curriculum that first learns
    easy (early) hops before tackling harder (later) hops.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        max_hops: int = 4,
        warmup_steps: int = 1000,
        curriculum_steps: int = 10000,
        final_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_hops = max_hops
        self.warmup_steps = warmup_steps
        self.curriculum_steps = curriculum_steps
        
        # Final weights at end of curriculum
        if final_weights is None:
            final_weights = [0.1, 0.2, 0.3, 0.4][:max_hops]
        self.register_buffer('final_weights', torch.tensor(final_weights))
        
        # Initial weights: focus on early hops
        initial_weights = [0.4, 0.3, 0.2, 0.1][:max_hops]
        self.register_buffer('initial_weights', torch.tensor(initial_weights))
        
        # Simple per-hop scorers
        self.hop_scorers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(max_hops)
        ])
        
        self.register_buffer('current_step', torch.tensor(0))
    
    def get_current_weights(self) -> torch.Tensor:
        """Get interpolated weights based on current training step."""
        step = self.current_step.item()
        
        if step < self.warmup_steps:
            # Warmup: use initial weights
            return self.initial_weights
        elif step < self.warmup_steps + self.curriculum_steps:
            # Curriculum: interpolate
            progress = (step - self.warmup_steps) / self.curriculum_steps
            return (1 - progress) * self.initial_weights + progress * self.final_weights
        else:
            # Post-curriculum: use final weights
            return self.final_weights
    
    def step(self):
        """Call this after each training step to update curriculum."""
        self.current_step += 1
    
    def forward(
        self,
        hop_embeddings: torch.Tensor,  # [B, C, max_hops, D]
        labels: torch.Tensor,          # [B]
        candidate_mask: torch.Tensor,  # [B, C]
        hop_mask: torch.Tensor,        # [B, C, max_hops]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute progressive hop loss with curriculum weighting."""
        B, C, max_hops, D = hop_embeddings.shape
        device = hop_embeddings.device
        
        weights = self.get_current_weights()
        losses = {'curriculum_weights': weights.tolist()}
        total_loss = torch.tensor(0.0, device=device)
        
        for h in range(self.max_hops):
            # Score using hop embedding
            hop_emb = hop_embeddings[:, :, h, :]  # [B, C, D]
            scores = self.hop_scorers[h](hop_emb).squeeze(-1)  # [B, C]
            
            # Mask and compute loss
            scores = scores.masked_fill(~candidate_mask.bool(), -65504.0)
            
            loss_h = F.cross_entropy(scores, labels, reduction='mean')
            losses[f'hop_{h+1}_loss'] = loss_h
            
            total_loss = total_loss + weights[h] * loss_h
        
        losses['total'] = total_loss
        return total_loss, losses


if __name__ == '__main__':
    print("Testing HopAuxiliaryLoss...")
    
    B, C, max_hops, D = 4, 10, 4, 128
    
    hop_aux = HopAuxiliaryLoss(
        hidden_dim=D,
        max_hops=max_hops,
        loss_weights=[0.1, 0.2, 0.3, 0.4],
        cumulative=True,
    )
    
    hop_embeds = torch.randn(B, C, max_hops, D)
    question = torch.randn(B, D)
    labels = torch.randint(0, C, (B,))
    candidate_mask = torch.ones(B, C, dtype=torch.bool)
    hop_mask = torch.ones(B, C, max_hops, dtype=torch.bool)
    
    # Make some hops invalid for testing
    hop_mask[:, :, 3] = False  # No 4th hop for any path
    
    total_loss, loss_dict = hop_aux(
        hop_embeds, question, labels, candidate_mask, hop_mask
    )
    
    print(f"Total auxiliary loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
    
    print("\nTesting ProgressiveHopLoss...")
    
    prog_loss = ProgressiveHopLoss(hidden_dim=D, max_hops=max_hops)
    
    print("Initial weights:", prog_loss.get_current_weights().tolist())
    
    # Simulate training steps
    for _ in range(5000):
        prog_loss.step()
    
    print("After 5000 steps:", prog_loss.get_current_weights().tolist())
