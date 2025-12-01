"""
KG Path Autoregressive Model - GPT-style autoregressive transformer for relation chain generation.

Architecture:
- Question Encoder: Pretrained transformer for question understanding
- Autoregressive Decoder: Causal transformer decoder for relation chain generation
- Supports multiple decoding strategies: greedy, beam search, nucleus sampling, top-k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from typing import Dict, Optional, Tuple, Any, List
import os
import math
import json

# Reuse QuestionEncoder from diffusion model
from kg_path_diffusion import QuestionEncoder


class AutoregressiveDecoder(nn.Module):
    """Causal transformer decoder for autoregressive relation chain generation."""
    
    def __init__(
        self,
        num_relations: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_path_length: int = 25,
        dropout: float = 0.1,
        question_dim: int = 256
    ):
        super().__init__()
        
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        
        # Relation embeddings
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        
        # Position embeddings
        self.pos_embedding = nn.Embedding(max_path_length, hidden_dim)
        
        # Cross-attention to question
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Self-attention layers (causal)
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.norm3_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Projection to relation vocabulary
        self.relation_head = nn.Linear(hidden_dim, num_relations)
        
        # Project question to hidden_dim if needed
        if question_dim != hidden_dim:
            self.question_proj = nn.Linear(question_dim, hidden_dim)
        else:
            self.question_proj = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        relation_ids: torch.Tensor,  # [B, seq_len]
        question_seq: torch.Tensor,  # [B, q_len, question_dim]
        question_mask: Optional[torch.Tensor] = None,  # [B, q_len] True for padding
        relation_mask: Optional[torch.Tensor] = None  # [B, seq_len] True for padding
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            relation_ids: Relation token IDs [B, seq_len]
            question_seq: Question sequence embeddings [B, q_len, question_dim]
            question_mask: Question padding mask (True = padding)
            relation_mask: Relation padding mask (True = padding)
        
        Returns:
            relation_logits: [B, seq_len, num_relations]
        """
        B, seq_len = relation_ids.shape
        device = relation_ids.device
        
        # Embed relations
        relation_emb = self.relation_embedding(relation_ids)  # [B, seq_len, hidden_dim]
        
        # Add position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)
        x = relation_emb + pos_emb
        x = self.dropout(x)
        
        # Project question
        question_proj = self.question_proj(question_seq)  # [B, q_len, hidden_dim]
        
        # Create causal mask for self-attention
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        
        # Apply decoder layers
        for i in range(len(self.self_attn_layers)):
            # Self-attention with causal masking
            residual = x
            x = self.norm1_layers[i](x)
            attn_out, _ = self.self_attn_layers[i](
                x, x, x,
                attn_mask=causal_mask,
                key_padding_mask=relation_mask
            )
            x = residual + self.dropout(attn_out)
            
            # Cross-attention to question
            residual = x
            x = self.norm2_layers[i](x)
            cross_attn_out, _ = self.cross_attn_layers[i](
                x, question_proj, question_proj,
                key_padding_mask=question_mask
            )
            x = residual + self.dropout(cross_attn_out)
            
            # Feed-forward
            residual = x
            x = self.norm3_layers[i](x)
            ffn_out = self.ffn_layers[i](x)
            x = residual + ffn_out
        
        # Project to relation vocabulary
        relation_logits = self.relation_head(x)  # [B, seq_len, num_relations]
        
        return relation_logits
    
    @torch.no_grad()
    def generate(
        self,
        question_seq: torch.Tensor,  # [B, q_len, question_dim]
        question_mask: Optional[torch.Tensor] = None,
        max_length: int = 25,
        decoding_strategy: str = "greedy",
        beam_width: int = 5,
        top_p: float = 0.9,
        top_k: int = 50,
        temperature: float = 1.0,
        length_penalty: float = 0.6,
        use_contrastive_search: bool = False,
        contrastive_penalty_alpha: float = 0.6,
        contrastive_top_k: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate relation chains autoregressively.
        
        Returns:
            relation_ids: [B, max_length] Generated relation IDs
            relation_logits: [B, max_length, num_relations] Logits for each position
        """
        B = question_seq.shape[0]
        device = question_seq.device
        
        # Project question
        question_proj = self.question_proj(question_seq)
        
        if decoding_strategy == "beam_search":
            return self._generate_beam_search(
                question_proj, question_mask, max_length, beam_width, length_penalty, temperature
            )
        elif decoding_strategy == "nucleus":
            return self._generate_nucleus(
                question_proj, question_mask, max_length, top_p, temperature
            )
        elif decoding_strategy == "top_k":
            return self._generate_top_k(
                question_proj, question_mask, max_length, top_k, temperature
            )
        elif use_contrastive_search:
            return self._generate_contrastive(
                question_proj, question_mask, max_length, contrastive_penalty_alpha, contrastive_top_k, temperature
            )
        else:
            return self._generate_greedy(
                question_proj, question_mask, max_length, temperature
            )
    
    def _generate_greedy(
        self,
        question_proj: torch.Tensor,
        question_mask: Optional[torch.Tensor],
        max_length: int,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding."""
        B = question_proj.shape[0]
        device = question_proj.device
        
        # Start with empty sequence (will generate first token)
        relation_ids = torch.zeros(B, 0, dtype=torch.long, device=device)
        all_logits = []
        
        for step in range(max_length):
            # If empty, use a dummy token to get first prediction
            if relation_ids.shape[1] == 0:
                dummy_input = torch.zeros(B, 1, dtype=torch.long, device=device)
            else:
                dummy_input = relation_ids
            
            # Get logits for current sequence
            logits = self.forward(dummy_input, question_proj, question_mask)
            if relation_ids.shape[1] == 0:
                next_logits = logits[:, 0, :] / temperature  # [B, num_relations]
            else:
                next_logits = logits[:, -1, :] / temperature  # [B, num_relations]
            
            # Mask out padding token (index 0)
            next_logits[:, 0] = float('-inf')
            
            # Greedy selection
            next_ids = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            relation_ids = torch.cat([relation_ids, next_ids], dim=1)
            all_logits.append(next_logits.unsqueeze(1))
        
        all_logits = torch.cat(all_logits, dim=1)  # [B, max_length, num_relations]
        return relation_ids, all_logits
    
    def _generate_beam_search(
        self,
        question_proj: torch.Tensor,
        question_mask: Optional[torch.Tensor],
        max_length: int,
        beam_width: int,
        length_penalty: float,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Beam search decoding."""
        B = question_proj.shape[0]
        device = question_proj.device
        
        # Expand question for beam search
        question_proj_expanded = question_proj.unsqueeze(1).expand(-1, beam_width, -1, -1)
        question_proj_expanded = question_proj_expanded.reshape(B * beam_width, *question_proj.shape[1:])
        
        if question_mask is not None:
            question_mask_expanded = question_mask.unsqueeze(1).expand(-1, beam_width, -1)
            question_mask_expanded = question_mask_expanded.reshape(B * beam_width, -1)
        else:
            question_mask_expanded = None
        
        # Initialize beams
        relation_ids = torch.zeros(B * beam_width, 1, dtype=torch.long, device=device)
        beam_scores = torch.zeros(B, beam_width, device=device)
        beam_scores[:, 1:] = float('-inf')
        beam_scores = beam_scores.view(-1)
        
        finished = torch.zeros(B * beam_width, dtype=torch.bool, device=device)
        
        for step in range(max_length):
            logits = self.forward(relation_ids, question_proj_expanded, question_mask_expanded)
            next_logits = logits[:, -1, :] / temperature  # [B*beam, num_relations]
            
            # Apply length penalty
            length_penalty_factor = ((step + 1) / 5.0) ** length_penalty
            next_logprobs = F.log_softmax(next_logits, dim=-1) / length_penalty_factor
            
            # Reshape for beam search
            next_logprobs = next_logprobs.view(B, beam_width, -1)  # [B, beam, vocab]
            
            # Get top candidates
            vocab_size = next_logprobs.shape[-1]
            top_logprobs, top_indices = torch.topk(
                next_logprobs.view(B, -1), beam_width * 2, dim=-1
            )
            
            # Select beams
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update beams (simplified - full implementation would track best beams)
            # For simplicity, use top beam_width
            selected_beams = beam_indices[:, :beam_width]
            selected_tokens = token_indices[:, :beam_width]
            
            # Update relation_ids (simplified version)
            new_relation_ids = []
            for b in range(B):
                for k in range(beam_width):
                    beam_idx = selected_beams[b, k].item()
                    token_idx = selected_tokens[b, k].item()
                    prev_seq = relation_ids[b * beam_width + beam_idx]
                    new_seq = torch.cat([prev_seq, token_idx.unsqueeze(0).unsqueeze(0)])
                    new_relation_ids.append(new_seq)
            
            relation_ids = torch.cat(new_relation_ids, dim=0)
        
        # Return best beam for each sample
        best_beams = relation_ids[:B * beam_width:beam_width]  # Take first beam (simplified)
        return best_beams[:, 1:], None  # Remove initial token
    
    def _generate_nucleus(
        self,
        question_proj: torch.Tensor,
        question_mask: Optional[torch.Tensor],
        max_length: int,
        top_p: float,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nucleus (top-p) sampling."""
        B = question_proj.shape[0]
        device = question_proj.device
        
        relation_ids = torch.zeros(B, 0, dtype=torch.long, device=device)
        all_logits = []
        
        for step in range(max_length):
            if relation_ids.shape[1] == 0:
                dummy_input = torch.zeros(B, 1, dtype=torch.long, device=device)
            else:
                dummy_input = relation_ids
            
            logits = self.forward(dummy_input, question_proj, question_mask)
            if relation_ids.shape[1] == 0:
                next_logits = logits[:, 0, :] / temperature
            else:
                next_logits = logits[:, -1, :] / temperature
            
            # Mask out padding token
            next_logits[:, 0] = float('-inf')
            
            # Nucleus sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Find cutoff
            sorted_indices_to_remove = cumprobs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Create mask
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_ids = torch.multinomial(probs, 1)
            relation_ids = torch.cat([relation_ids, next_ids], dim=1)
            all_logits.append(next_logits.unsqueeze(1))
        
        all_logits = torch.cat(all_logits, dim=1)
        return relation_ids, all_logits
    
    def _generate_top_k(
        self,
        question_proj: torch.Tensor,
        question_mask: Optional[torch.Tensor],
        max_length: int,
        top_k: int,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k sampling."""
        B = question_proj.shape[0]
        device = question_proj.device
        
        relation_ids = torch.zeros(B, 0, dtype=torch.long, device=device)
        all_logits = []
        
        for step in range(max_length):
            if relation_ids.shape[1] == 0:
                dummy_input = torch.zeros(B, 1, dtype=torch.long, device=device)
            else:
                dummy_input = relation_ids
            
            logits = self.forward(dummy_input, question_proj, question_mask)
            if relation_ids.shape[1] == 0:
                next_logits = logits[:, 0, :] / temperature
            else:
                next_logits = logits[:, -1, :] / temperature
            
            # Mask out padding token
            next_logits[:, 0] = float('-inf')
            
            # Top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_logits, min(top_k, next_logits.shape[-1]), dim=-1)
            top_k_mask = torch.zeros_like(next_logits, dtype=torch.bool)
            top_k_mask.scatter_(-1, top_k_indices, True)
            next_logits[~top_k_mask] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_ids = torch.multinomial(probs, 1)
            relation_ids = torch.cat([relation_ids, next_ids], dim=1)
            all_logits.append(next_logits.unsqueeze(1))
        
        all_logits = torch.cat(all_logits, dim=1)
        return relation_ids, all_logits
    
    def _generate_contrastive(
        self,
        question_proj: torch.Tensor,
        question_mask: Optional[torch.Tensor],
        max_length: int,
        penalty_alpha: float,
        top_k: int,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Contrastive search decoding."""
        B = question_proj.shape[0]
        device = question_proj.device
        
        relation_ids = torch.zeros(B, 0, dtype=torch.long, device=device)
        all_logits = []
        
        for step in range(max_length):
            if relation_ids.shape[1] == 0:
                dummy_input = torch.zeros(B, 1, dtype=torch.long, device=device)
            else:
                dummy_input = relation_ids
            
            logits = self.forward(dummy_input, question_proj, question_mask)
            if relation_ids.shape[1] == 0:
                next_logits = logits[:, 0, :] / temperature
            else:
                next_logits = logits[:, -1, :] / temperature
            
            # Mask out padding token
            next_logits[:, 0] = float('-inf')
            
            # Get top-k candidates
            top_k_logits, top_k_indices = torch.topk(next_logits, min(top_k, next_logits.shape[-1]), dim=-1)
            
            # Apply repetition penalty
            for b in range(B):
                for token_id in relation_ids[b].unique():
                    if token_id.item() > 0:  # Skip padding
                        next_logits[b, token_id] -= penalty_alpha
            
            # Sample from top-k
            top_k_mask = torch.zeros_like(next_logits, dtype=torch.bool)
            top_k_mask.scatter_(-1, top_k_indices, True)
            next_logits[~top_k_mask] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_ids = torch.multinomial(probs, 1)
            relation_ids = torch.cat([relation_ids, next_ids], dim=1)
            all_logits.append(next_logits.unsqueeze(1))
        
        all_logits = torch.cat(all_logits, dim=1)
        return relation_ids, all_logits


class KGPathAutoregressiveModel(nn.Module):
    """
    Autoregressive model for KG Path Generation.
    
    Architecture:
    1. Question Encoder: Pretrained transformer for question understanding
    2. Autoregressive Decoder: Causal transformer decoder for relation chain generation
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 256,
        question_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_question_encoder: bool = False,
        num_layers: int = 6,
        num_heads: int = 8,
        max_path_length: int = 25,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        
        # Question encoder
        self.question_encoder = QuestionEncoder(
            model_name=question_encoder_name,
            output_dim=hidden_dim,
            freeze=freeze_question_encoder
        )
        
        # Autoregressive decoder
        self.decoder = AutoregressiveDecoder(
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_path_length=max_path_length,
            dropout=dropout,
            question_dim=hidden_dim
        )
    
    def encode_inputs(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode question inputs."""
        question_seq, question_pooled = self.question_encoder(
            question_input_ids, question_attention_mask
        )
        return question_seq, question_pooled
    
    def forward(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        target_relations: torch.Tensor,
        relation_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            question_input_ids: [B, q_len]
            question_attention_mask: [B, q_len]
            target_relations: [B, seq_len] Relation IDs
            relation_mask: [B, seq_len] True for padding
        
        Returns:
            Dictionary with 'loss' and 'relation_loss'
        """
        # Encode question
        question_seq, _ = self.encode_inputs(question_input_ids, question_attention_mask)
        question_mask = ~question_attention_mask.bool()  # True for padding
        
        # Get logits
        relation_logits = self.decoder(
            target_relations,
            question_seq,
            question_mask=question_mask,
            relation_mask=relation_mask
        )  # [B, seq_len, num_relations]
        
        # Compute loss
        if relation_mask is not None:
            # Mask out padding tokens
            valid_mask = ~relation_mask  # True for valid tokens
            if valid_mask.any():
                flat_logits = relation_logits[valid_mask]  # [N, num_relations]
                flat_targets = target_relations[valid_mask]  # [N]
                loss = F.cross_entropy(flat_logits, flat_targets)
            else:
                loss = torch.tensor(0.0, device=relation_logits.device, requires_grad=True)
        else:
            loss = F.cross_entropy(
                relation_logits.reshape(-1, self.num_relations),
                target_relations.reshape(-1)
            )
        
        return {
            'loss': loss,
            'relation_loss': loss
        }
    
    def forward_multipath(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        all_target_relations: torch.Tensor,
        all_path_lengths: torch.Tensor,
        num_paths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-path training forward pass.
        """
        B = question_input_ids.shape[0]
        max_paths = all_target_relations.shape[1]
        rel_len = all_target_relations.shape[2]
        device = question_input_ids.device
        
        # Encode question once
        question_seq, _ = self.encode_inputs(question_input_ids, question_attention_mask)
        question_mask = ~question_attention_mask.bool()
        
        # Expand question for all paths
        question_seq_expanded = question_seq.unsqueeze(1).expand(-1, max_paths, -1, -1)
        question_seq_expanded = question_seq_expanded.reshape(B * max_paths, *question_seq.shape[1:])
        question_mask_expanded = question_mask.unsqueeze(1).expand(-1, max_paths, -1)
        question_mask_expanded = question_mask_expanded.reshape(B * max_paths, -1)
        
        # Flatten paths
        flat_relations = all_target_relations.reshape(B * max_paths, rel_len)
        
        # Create relation mask
        rel_lengths = (all_path_lengths - 3).clamp(min=0).clamp(max=rel_len)
        rel_lengths_flat = rel_lengths.reshape(B * max_paths)
        relation_mask = torch.arange(rel_len, device=device).unsqueeze(0) >= rel_lengths_flat.unsqueeze(1)
        
        # Valid path mask
        valid_path_mask = torch.arange(max_paths, device=device).unsqueeze(0) < num_paths.unsqueeze(1)
        valid_path_mask_flat = valid_path_mask.reshape(B * max_paths)
        
        # Forward pass
        relation_logits = self.decoder(
            flat_relations,
            question_seq_expanded,
            question_mask=question_mask_expanded,
            relation_mask=relation_mask
        )
        
        # Compute loss per path
        losses_per_path = []
        for i in range(B * max_paths):
            if not valid_path_mask_flat[i]:
                losses_per_path.append(torch.tensor(0.0, device=device, requires_grad=True))
                continue
            
            r_mask = relation_mask[i]
            if r_mask.sum() == 0:
                losses_per_path.append(torch.tensor(0.0, device=device, requires_grad=True))
                continue
            
            valid_mask = ~r_mask
            if valid_mask.any():
                r_logits = relation_logits[i][valid_mask]
                r_targets = flat_relations[i][valid_mask]
                r_loss = F.cross_entropy(r_logits, r_targets)
                losses_per_path.append(r_loss)
            else:
                losses_per_path.append(torch.tensor(0.0, device=device, requires_grad=True))
        
        losses_per_path = torch.stack(losses_per_path).reshape(B, max_paths)
        valid_mask = valid_path_mask.float()
        losses_per_path = losses_per_path * valid_mask
        
        num_paths_float = num_paths.float().clamp(min=1)
        sample_losses = losses_per_path.sum(dim=1) / num_paths_float
        total_loss = sample_losses.mean()
        
        return {
            'loss': total_loss,
            'relation_loss': total_loss,
            'num_paths_avg': num_paths.float().mean()
        }
    
    @torch.no_grad()
    def generate(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        path_length: int = 10,
        decoding_strategy: str = "greedy",
        beam_width: int = 5,
        top_p: float = 0.9,
        top_k: int = 50,
        temperature: float = 1.0,
        length_penalty: float = 0.6,
        use_contrastive_search: bool = False,
        contrastive_penalty_alpha: float = 0.6,
        contrastive_top_k: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate relation chains."""
        question_seq, _ = self.encode_inputs(question_input_ids, question_attention_mask)
        question_mask = ~question_attention_mask.bool()
        
        return self.decoder.generate(
            question_seq,
            question_mask=question_mask,
            max_length=path_length,
            decoding_strategy=decoding_strategy,
            beam_width=beam_width,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            use_contrastive_search=use_contrastive_search,
            contrastive_penalty_alpha=contrastive_penalty_alpha,
            contrastive_top_k=contrastive_top_k
        )
    
    @torch.no_grad()
    def generate_multiple(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        num_paths: int = 5,
        path_length: int = 10,
        temperature: float = 1.0,
        decoding_strategy: str = "greedy",
        beam_width: int = 5,
        top_p: float = 0.9,
        top_k: int = 50,
        length_penalty: float = 0.6,
        diversity_penalty: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate multiple diverse paths."""
        B = question_input_ids.shape[0]
        device = question_input_ids.device
        
        all_relations = []
        
        for path_idx in range(num_paths):
            # Adjust temperature for diversity
            if path_idx > 0 and diversity_penalty > 0:
                curr_temp = temperature * (1 + diversity_penalty * path_idx)
            else:
                curr_temp = temperature
            
            relations, _ = self.generate(
                question_input_ids,
                question_attention_mask,
                path_length=path_length,
                decoding_strategy=decoding_strategy,
                beam_width=beam_width,
                top_p=top_p,
                top_k=top_k,
                temperature=curr_temp,
                length_penalty=length_penalty
            )
            
            all_relations.append(relations)
        
        # Stack: [B, num_paths, path_length]
        all_relations = torch.stack(all_relations, dim=1)
        
        # Return dummy entities (not used in relation-only mode)
        dummy_entities = torch.zeros(B, num_paths, path_length, dtype=torch.long, device=device)
        
        return dummy_entities, all_relations


class KGPathAutoregressiveLightning(pl.LightningModule):
    """PyTorch Lightning module for Autoregressive KG Path Generation."""
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 256,
        question_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_question_encoder: bool = False,
        num_layers: int = 6,
        num_heads: int = 8,
        max_path_length: int = 25,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        use_multipath_training: bool = True,
        # Decoding parameters
        decoding_strategy: str = "beam_search",
        beam_width: int = 5,
        top_p: float = 0.9,
        top_k: int = 50,
        temperature: float = 1.0,
        length_penalty: float = 0.6,
        use_contrastive_search: bool = False,
        contrastive_penalty_alpha: float = 0.6,
        contrastive_top_k: int = 4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = KGPathAutoregressiveModel(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            question_encoder_name=question_encoder_name,
            freeze_question_encoder=freeze_question_encoder,
            num_layers=num_layers,
            num_heads=num_heads,
            max_path_length=max_path_length,
            dropout=dropout
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.use_multipath_training = use_multipath_training
        
        # Decoding parameters
        self.decoding_strategy = decoding_strategy
        self.beam_width = beam_width
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.use_contrastive_search = use_contrastive_search
        self.contrastive_penalty_alpha = contrastive_penalty_alpha
        self.contrastive_top_k = contrastive_top_k
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        if self.use_multipath_training and 'all_path_relations' in batch:
            return self.forward_multipath(batch)
        else:
            return self.forward_single(batch)
    
    def forward_single(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Single path forward."""
        # Create relation mask from path_lengths if available
        relation_mask = None
        if 'path_lengths' in batch:
            max_len = batch['path_relations'].shape[1]
            lengths = batch['path_lengths']
            relation_mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        return self.model(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            target_relations=batch['path_relations'],
            relation_mask=relation_mask
        )
    
    def forward_multipath(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Multi-path forward."""
        return self.model.forward_multipath(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            all_target_relations=batch['all_path_relations'],
            all_path_lengths=batch['all_path_lengths'],
            num_paths=batch['num_paths']
        )
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
        
        outputs = self(batch)
        
        self.log('train/loss', outputs['loss'], prog_bar=True, sync_dist=True)
        self.log('train/relation_loss', outputs['relation_loss'], sync_dist=True)
        
        if 'num_paths_avg' in outputs:
            self.log('train/num_paths_avg', outputs['num_paths_avg'], sync_dist=True)
        
        return outputs['loss']
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        
        self.log('val/loss', outputs['loss'], prog_bar=True, sync_dist=True)
        self.log('val/relation_loss', outputs['relation_loss'], sync_dist=True)
        
        if 'num_paths_avg' in outputs:
            self.log('val/num_paths_avg', outputs['num_paths_avg'], sync_dist=True)
        
        return outputs['loss']
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> Dict[str, Any]:
        """
        Load checkpoint with parameter matching.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: If True, raise error on mismatches. If False, skip mismatched parameters.
        
        Returns:
            Dictionary with loading statistics
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'model.' prefix if present (PyTorch Lightning format)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                cleaned_key = key[6:]  # Remove 'model.' prefix
                cleaned_state_dict[cleaned_key] = value
            else:
                cleaned_state_dict[key] = value
        
        # Get current model state dict
        model_state = self.model.state_dict()
        
        # Statistics
        loaded = 0
        skipped = 0
        missing = []
        
        # Match and load parameters
        for name, param in model_state.items():
            if name in cleaned_state_dict:
                checkpoint_param = cleaned_state_dict[name]
                
                # Check if shapes match
                if param.shape == checkpoint_param.shape:
                    try:
                        param.data.copy_(checkpoint_param)
                        loaded += 1
                    except Exception as e:
                        if strict:
                            raise RuntimeError(f"Failed to load parameter {name}: {e}")
                        skipped += 1
                else:
                    if strict:
                        raise RuntimeError(
                            f"Shape mismatch for parameter {name}: "
                            f"model has {param.shape}, checkpoint has {checkpoint_param.shape}"
                        )
                    skipped += 1
            else:
                missing.append(name)
                if strict:
                    raise RuntimeError(f"Parameter {name} not found in checkpoint")
        
        stats = {
            'loaded': loaded,
            'skipped': skipped,
            'missing': len(missing),
            'missing_params': missing[:10]
        }
        
        return stats
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight', 'norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

