"""
Flow Matching based path generator.

Implements a rectified-flow style objective (Lipman et al., 2022) for
discrete reasoning paths by operating in the continuous embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .diffusion import (
    SinusoidalPositionEmbeddings,
    PathTransformerBlock,
    CrossAttentionBlock,
)


class FlowMatchingTransformer(nn.Module):
    """Transformer that predicts velocity fields for entity/relation tokens."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        question_dim: int = 256,
        graph_dim: int = 256,
        max_path_length: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length

        # Token embeddings
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim, padding_idx=0)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim, padding_idx=0)

        # Positional / type encodings
        self.pos_embedding = nn.Embedding(max_path_length * 2, hidden_dim)
        self.type_embedding = nn.Embedding(2, hidden_dim)

        # Time conditioning
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Context projections
        self.question_proj = nn.Linear(question_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)

        # Transformer stack
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "self_attn": PathTransformerBlock(
                            hidden_dim, num_heads, dropout=dropout
                        ),
                        "cross_attn_q": CrossAttentionBlock(
                            hidden_dim, hidden_dim, num_heads, dropout
                        ),
                        "cross_attn_g": CrossAttentionBlock(
                            hidden_dim, hidden_dim, num_heads, dropout
                        ),
                    }
                )
            )

        self.norm_out = nn.LayerNorm(hidden_dim)

        # Velocity heads
        self.entity_velocity = nn.Linear(hidden_dim, hidden_dim)
        self.relation_velocity = nn.Linear(hidden_dim, hidden_dim)

        # Optional logits for auxiliary CE loss / decoding
        self.entity_head = nn.Linear(hidden_dim, num_entities)
        self.relation_head = nn.Linear(hidden_dim, num_relations)

    def embed_entities(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.entity_embedding(tokens)

    def embed_relations(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.relation_embedding(tokens)

    def decode_entities(self, representations: torch.Tensor) -> torch.Tensor:
        return self.entity_head(representations)

    def decode_relations(self, representations: torch.Tensor) -> torch.Tensor:
        return self.relation_head(representations)

    def forward(
        self,
        entity_inputs: torch.Tensor,
        relation_inputs: torch.Tensor,
        timesteps: torch.Tensor,
        question_encoding: torch.Tensor,
        graph_node_encoding: torch.Tensor,
        path_mask: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
        graph_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            entity_inputs: [B, L, D] noisy embeddings
            relation_inputs: [B, L-1, D] noisy embeddings
            timesteps: [B], values in [0, 1]
            question_encoding: [B, seq_len, question_dim]
            graph_node_encoding: [B, num_nodes, graph_dim]
        Returns:
            entity_velocity, relation_velocity, entity_logits, relation_logits
        """
        B, L, _ = entity_inputs.shape
        device = entity_inputs.device
        seq_len = 2 * L - 1

        x = torch.zeros(B, seq_len, self.hidden_dim, device=device)
        x[:, 0::2, :] = entity_inputs
        x[:, 1::2, :] = relation_inputs

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(max=self.max_path_length * 2 - 1)
        x = x + self.pos_embedding(positions)

        types = torch.zeros(seq_len, dtype=torch.long, device=device)
        types[0::2] = 0
        types[1::2] = 1
        x = x + self.type_embedding(types).unsqueeze(0)

        time_emb = self.time_mlp(timesteps)
        question_ctx = self.question_proj(question_encoding)
        graph_ctx = self.graph_proj(graph_node_encoding)

        if path_mask is not None:
            seq_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
            seq_mask[:, 0::2] = ~path_mask
            seq_mask[:, 1::2] = ~path_mask[:, :-1]
        else:
            seq_mask = None

        for block in self.blocks:
            x = block["self_attn"](x, time_emb, key_padding_mask=seq_mask)
            x = block["cross_attn_q"](x, question_ctx, context_mask=question_mask)
            x = block["cross_attn_g"](x, graph_ctx, context_mask=graph_mask)

        x = self.norm_out(x)
        entity_repr = x[:, 0::2, :]
        relation_repr = x[:, 1::2, :]

        entity_velocity = self.entity_velocity(entity_repr)
        relation_velocity = self.relation_velocity(relation_repr)
        entity_logits = self.entity_head(entity_repr)
        relation_logits = self.relation_head(relation_repr)

        return entity_velocity, relation_velocity, entity_logits, relation_logits


class FlowMatchingPathGenerator(nn.Module):
    """Flow matching objective wrapper for training & sampling."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        question_dim: int = 256,
        graph_dim: int = 256,
        max_path_length: int = 20,
        dropout: float = 0.1,
        num_integration_steps: int = 32,
        ce_weight: float = 0.1,
    ):
        super().__init__()
        self.transformer = FlowMatchingTransformer(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            question_dim=question_dim,
            graph_dim=graph_dim,
            max_path_length=max_path_length,
            dropout=dropout,
        )
        self.hidden_dim = hidden_dim
        self.num_integration_steps = num_integration_steps
        self.ce_weight = ce_weight

    def _interpolate(
        self, clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        while t.dim() < clean.dim():
            t = t.unsqueeze(-1)
        return (1 - t) * noise + t * clean

    def _masked_mse(
        self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            return F.mse_loss(pred, target)
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
        diff = (pred - target) ** 2
        diff = diff * mask
        denom = mask.sum().clamp(min=1.0)
        return diff.sum() / denom

    def forward(
        self,
        target_entities: torch.Tensor,
        target_relations: torch.Tensor,
        question_encoding: torch.Tensor,
        graph_node_encoding: torch.Tensor,
        path_mask: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
    ):
        B = target_entities.shape[0]
        device = target_entities.device
        t = torch.rand(B, device=device)

        clean_entity = self.transformer.embed_entities(target_entities)
        clean_relation = self.transformer.embed_relations(target_relations)

        noise_entity = torch.randn_like(clean_entity)
        noise_relation = torch.randn_like(clean_relation)

        xt_entity = self._interpolate(clean_entity, noise_entity, t)
        xt_relation = self._interpolate(clean_relation, noise_relation, t)

        entity_vel, relation_vel, entity_logits, relation_logits = self.transformer(
            xt_entity,
            xt_relation,
            t,
            question_encoding,
            graph_node_encoding,
            path_mask=path_mask,
            question_mask=question_mask,
        )

        entity_target = clean_entity - noise_entity
        relation_target = clean_relation - noise_relation

        ent_mask = path_mask if path_mask is not None else None
        rel_mask = path_mask[:, :-1] if path_mask is not None else None

        entity_loss = self._masked_mse(entity_vel, entity_target, ent_mask)
        relation_loss = self._masked_mse(relation_vel, relation_target, rel_mask)

        if self.ce_weight > 0:
            ce_entities = F.cross_entropy(
                entity_logits.view(-1, self.transformer.num_entities),
                target_entities.view(-1),
                ignore_index=0,
            )
            ce_relations = F.cross_entropy(
                relation_logits.view(-1, self.transformer.num_relations),
                target_relations.view(-1),
                ignore_index=0,
            )
            aux_loss = (ce_entities + ce_relations) * 0.5
        else:
            aux_loss = torch.tensor(0.0, device=device)

        total_loss = entity_loss + relation_loss + self.ce_weight * aux_loss

        return {
            "loss": total_loss,
            "entity_loss": entity_loss,
            "relation_loss": relation_loss,
            "aux_loss": aux_loss.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        question_encoding: torch.Tensor,
        graph_node_encoding: torch.Tensor,
        path_length: int,
        question_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
    ):
        """Integrate velocity field to obtain discrete paths."""
        B = question_encoding.shape[0]
        device = question_encoding.device
        steps = num_steps or self.num_integration_steps

        entity_state = torch.randn(B, path_length, self.hidden_dim, device=device)
        relation_state = torch.randn(B, path_length - 1, self.hidden_dim, device=device)

        time_grid = torch.linspace(0, 1, steps + 1, device=device)
        for i in range(steps):
            t_val = torch.full((B,), time_grid[i], device=device)
            entity_vel, relation_vel, _, _ = self.transformer(
                entity_state,
                relation_state,
                t_val,
                question_encoding,
                graph_node_encoding,
                question_mask=question_mask,
            )
            dt = time_grid[i + 1] - time_grid[i]
            entity_state = entity_state + entity_vel * dt
            relation_state = relation_state + relation_vel * dt

        entity_logits = self.transformer.decode_entities(entity_state)
        relation_logits = self.transformer.decode_relations(relation_state)

        if temperature > 0:
            entity_probs = F.softmax(entity_logits / temperature, dim=-1)
            relation_probs = F.softmax(relation_logits / temperature, dim=-1)
            entity_tokens = torch.multinomial(
                entity_probs.view(-1, self.transformer.num_entities), 1
            ).view(B, path_length)
            relation_tokens = torch.multinomial(
                relation_probs.view(-1, self.transformer.num_relations), 1
            ).view(B, path_length - 1)
        else:
            entity_tokens = entity_logits.argmax(dim=-1)
            relation_tokens = relation_logits.argmax(dim=-1)

        return entity_tokens, relation_tokens


