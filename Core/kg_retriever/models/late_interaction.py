"""
Late Interaction Module for ColBERT-style Token-Level Matching.

Implements MaxSim scoring: for each query token, find max similarity
across all document/path tokens, then sum for final relevance score.

This approach provides fine-grained matching while allowing pre-computation
of document embeddings for efficient inference.

Reference: ColBERTv2 (Santhanam et al., 2022), Jina-ColBERT-v2 (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LateInteractionScorer(nn.Module):
    """
    Efficient MaxSim scoring for ColBERT-style late interaction.
    
    Computes token-level similarity between query and document tokens,
    using MaxSim aggregation for relevance scoring.
    
    Features:
    - Batch-efficient MaxSim computation
    - Learned temperature parameter
    - Mask-aware scoring for variable-length inputs
    - Optional L2 normalization for cosine similarity
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        temperature: float = 0.02,
        learnable_temperature: bool = False,
        normalize: bool = True,
    ):
        """
        Args:
            hidden_dim: Dimension of token embeddings
            temperature: Temperature for scaling similarities
            learnable_temperature: If True, temperature is learned
            normalize: If True, L2 normalize embeddings before similarity
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.normalize = normalize
        
        if learnable_temperature:
            # Log-space for numerical stability
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))
    
    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)
    
    def forward(
        self,
        query_embeds: torch.Tensor,     # [B, Tq, D]
        path_embeds: torch.Tensor,      # [B, C, Tp, D]
        query_mask: Optional[torch.Tensor] = None,   # [B, Tq]
        path_mask: Optional[torch.Tensor] = None,    # [B, C, Tp]
    ) -> torch.Tensor:
        """
        Compute MaxSim scores between queries and candidate paths.
        
        MaxSim: For each query token, find the maximum similarity across
        all path tokens. Sum these max similarities for the final score.
        
        Args:
            query_embeds: Query token embeddings [B, Tq, D]
            path_embeds: Path token embeddings [B, C, Tp, D]
            query_mask: Mask for query tokens [B, Tq]
            path_mask: Mask for path tokens [B, C, Tp]
            
        Returns:
            scores: Relevance scores [B, C]
        """
        B, Tq, D = query_embeds.shape
        _, C, Tp, _ = path_embeds.shape
        
        # Normalize embeddings for cosine similarity
        if self.normalize:
            query_embeds = F.normalize(query_embeds, p=2, dim=-1)
            path_embeds = F.normalize(path_embeds, p=2, dim=-1)
        
        # Compute similarity matrix: [B, C, Tq, Tp]
        # Reshape for batch matmul
        query_expanded = query_embeds.unsqueeze(1)  # [B, 1, Tq, D]
        path_transposed = path_embeds.transpose(-1, -2)  # [B, C, D, Tp]
        
        # [B, 1, Tq, D] @ [B, C, D, Tp] -> [B, C, Tq, Tp]
        similarity = torch.matmul(query_expanded, path_transposed)
        
        # Apply path mask before taking max
        if path_mask is not None:
            # path_mask: [B, C, Tp] -> [B, C, 1, Tp]
            path_mask_expanded = path_mask.unsqueeze(2)
            similarity = similarity.masked_fill(~path_mask_expanded.bool(), -65504.0)  # float16 safe
        
        # MaxSim: max over path tokens for each query token
        # [B, C, Tq, Tp] -> [B, C, Tq]
        max_sim, _ = similarity.max(dim=-1)
        
        # Apply query mask and compute mean (not sum) over query tokens
        # Mean gives scores in [0, 1] range like PathRanker's cosine similarity
        if query_mask is not None:
            # query_mask: [B, Tq] -> [B, 1, Tq]
            query_mask_expanded = query_mask.unsqueeze(1)
            max_sim = max_sim.masked_fill(~query_mask_expanded.bool(), 0.0)
            # Mean over valid tokens only
            valid_count = query_mask_expanded.float().sum(dim=-1).clamp(min=1.0)  # [B, 1]
            scores = max_sim.sum(dim=-1) / valid_count  # [B, C]
        else:
            scores = max_sim.mean(dim=-1)  # [B, C]
        
        # Scale by temperature (like PathRanker divides by 0.05)
        scores = scores / self.temperature
        
        return scores


class HierarchicalLateInteraction(nn.Module):
    """
    Hierarchical Late Interaction for Multi-Hop Paths.
    
    Extends MaxSim scoring to multi-hop paths by:
    1. Computing per-hop MaxSim scores
    2. Aggregating hop scores with learned weights
    3. Supporting hop-level attention
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        max_hops: int = 4,
        temperature: float = 0.02,
        normalize: bool = True,
        hop_aggregation: str = 'weighted_sum',  # 'weighted_sum', 'attention', 'max'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_hops = max_hops
        self.normalize = normalize
        self.hop_aggregation = hop_aggregation
        
        self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))
        
        if hop_aggregation == 'weighted_sum':
            # Learnable weights for each hop
            self.hop_weights = nn.Parameter(torch.ones(max_hops) / max_hops)
        elif hop_aggregation == 'attention':
            # Query-conditioned hop attention
            self.hop_attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
    
    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)
    
    def forward(
        self,
        query_embeds: torch.Tensor,      # [B, Tq, D]
        hop_embeds: torch.Tensor,        # [B, C, max_hops, Th, D] - per-hop token embeds
        query_pooled: torch.Tensor,      # [B, D] - pooled query for attention
        query_mask: Optional[torch.Tensor] = None,   # [B, Tq]
        hop_mask: Optional[torch.Tensor] = None,     # [B, C, max_hops]
        hop_token_mask: Optional[torch.Tensor] = None,  # [B, C, max_hops, Th]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hierarchical late interaction scores.
        
        Args:
            query_embeds: Query token embeddings [B, Tq, D]
            hop_embeds: Per-hop token embeddings [B, C, max_hops, Th, D]
            query_pooled: Pooled query embedding [B, D]
            query_mask: Query token mask [B, Tq]
            hop_mask: Hop validity mask [B, C, max_hops]
            hop_token_mask: Token mask per hop [B, C, max_hops, Th]
            
        Returns:
            final_scores: Aggregated scores [B, C]
            hop_scores: Per-hop scores [B, C, max_hops]
        """
        B, Tq, D = query_embeds.shape
        _, C, max_hops, Th, _ = hop_embeds.shape
        
        if self.normalize:
            query_embeds = F.normalize(query_embeds, p=2, dim=-1)
            hop_embeds = F.normalize(hop_embeds, p=2, dim=-1)
        
        # Compute per-hop MaxSim scores
        hop_scores_list = []
        
        for h in range(max_hops):
            # Get hop h embeddings: [B, C, Th, D]
            hop_h_embeds = hop_embeds[:, :, h, :, :]
            
            # Compute MaxSim for this hop
            # [B, 1, Tq, D] @ [B, C, D, Th] -> [B, C, Tq, Th]
            query_expanded = query_embeds.unsqueeze(1)
            hop_transposed = hop_h_embeds.transpose(-1, -2)
            sim = torch.matmul(query_expanded, hop_transposed)
            
            # Apply hop token mask
            if hop_token_mask is not None:
                token_mask = hop_token_mask[:, :, h, :].unsqueeze(2)  # [B, C, 1, Th]
                sim = sim.masked_fill(~token_mask.bool(), -65504.0)
            
            # Max over hop tokens
            max_sim, _ = sim.max(dim=-1)  # [B, C, Tq]
            
            # Apply query mask
            if query_mask is not None:
                qmask = query_mask.unsqueeze(1)
                max_sim = max_sim.masked_fill(~qmask.bool(), 0.0)
            
            # Sum over query tokens
            hop_score = max_sim.sum(dim=-1)  # [B, C]
            hop_scores_list.append(hop_score)
        
        # Stack hop scores: [B, C, max_hops]
        hop_scores = torch.stack(hop_scores_list, dim=-1)
        
        # Apply hop mask
        if hop_mask is not None:
            hop_scores = hop_scores.masked_fill(~hop_mask.bool(), 0.0)
        
        # Aggregate hop scores
        if self.hop_aggregation == 'weighted_sum':
            weights = F.softmax(self.hop_weights, dim=0)  # [max_hops]
            if hop_mask is not None:
                # Renormalize weights based on valid hops
                masked_weights = weights.unsqueeze(0).unsqueeze(0) * hop_mask.float()
                masked_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-9)
                final_scores = (hop_scores * masked_weights).sum(dim=-1)
            else:
                final_scores = (hop_scores * weights).sum(dim=-1)
                
        elif self.hop_aggregation == 'attention':
            # Query-conditioned attention over hops
            # Expand query for each hop
            query_exp = query_pooled.unsqueeze(1).unsqueeze(2).expand(-1, C, max_hops, -1)
            
            # Get mean hop representation
            hop_pooled = hop_embeds.mean(dim=3)  # [B, C, max_hops, D]
            
            # Concatenate and compute attention
            attn_input = torch.cat([query_exp, hop_pooled], dim=-1)  # [B, C, max_hops, 2D]
            attn_weights = self.hop_attention(attn_input).squeeze(-1)  # [B, C, max_hops]
            
            if hop_mask is not None:
                attn_weights = attn_weights.masked_fill(~hop_mask.bool(), -65504.0)
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            final_scores = (hop_scores * attn_weights).sum(dim=-1)
            
        elif self.hop_aggregation == 'max':
            if hop_mask is not None:
                hop_scores = hop_scores.masked_fill(~hop_mask.bool(), -65504.0)
            final_scores, _ = hop_scores.max(dim=-1)
        
        # Scale by temperature
        final_scores = final_scores / self.temperature
        hop_scores = hop_scores / self.temperature
        
        return final_scores, hop_scores


def maxsim_batch(
    query_embeds: torch.Tensor,
    doc_embeds: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Standalone MaxSim function for simple use cases.
    
    Args:
        query_embeds: [B, Tq, D]
        doc_embeds: [B, C, Td, D]
        query_mask: [B, Tq]
        doc_mask: [B, C, Td]
        normalize: Whether to L2 normalize embeddings
        
    Returns:
        scores: [B, C]
    """
    if normalize:
        query_embeds = F.normalize(query_embeds, p=2, dim=-1)
        doc_embeds = F.normalize(doc_embeds, p=2, dim=-1)
    
    # [B, 1, Tq, D] @ [B, C, D, Td] -> [B, C, Tq, Td]
    sim = torch.matmul(query_embeds.unsqueeze(1), doc_embeds.transpose(-1, -2))
    
    if doc_mask is not None:
        sim = sim.masked_fill(~doc_mask.unsqueeze(2).bool(), -65504.0)
    
    # Max over doc tokens
    max_sim, _ = sim.max(dim=-1)  # [B, C, Tq]
    
    if query_mask is not None:
        max_sim = max_sim.masked_fill(~query_mask.unsqueeze(1).bool(), 0.0)
    
    # Sum over query tokens
    return max_sim.sum(dim=-1)


if __name__ == '__main__':
    print("Testing LateInteractionScorer...")
    
    scorer = LateInteractionScorer(hidden_dim=128, temperature=0.02)
    
    B, Tq, C, Tp, D = 2, 16, 10, 32, 128
    
    query = torch.randn(B, Tq, D)
    paths = torch.randn(B, C, Tp, D)
    query_mask = torch.ones(B, Tq, dtype=torch.bool)
    path_mask = torch.ones(B, C, Tp, dtype=torch.bool)
    
    scores = scorer(query, paths, query_mask, path_mask)
    print(f"Scores shape: {scores.shape}")  # [B, C]
    print(f"Scores range: [{scores.min():.2f}, {scores.max():.2f}]")
    
    print("\nTesting HierarchicalLateInteraction...")
    
    hier_scorer = HierarchicalLateInteraction(hidden_dim=128, max_hops=4)
    
    max_hops, Th = 4, 8
    hop_embeds = torch.randn(B, C, max_hops, Th, D)
    query_pooled = torch.randn(B, D)
    hop_mask = torch.ones(B, C, max_hops, dtype=torch.bool)
    
    final_scores, hop_scores = hier_scorer(
        query, hop_embeds, query_pooled, query_mask, hop_mask
    )
    print(f"Final scores shape: {final_scores.shape}")  # [B, C]
    print(f"Hop scores shape: {hop_scores.shape}")  # [B, C, max_hops]
