"""
KG Path GNN + Decoder Model - Hybrid model combining GNN encoder with autoregressive decoder.

Architecture:
- Question Encoder: Pretrained transformer for question understanding
- GNN Encoder: Graph Attention Network to encode knowledge graph structure
- Autoregressive Decoder: Causal transformer decoder for relation chain generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from typing import Dict, Optional, Tuple, Any, List
import os
import math

from .base import QuestionEncoder


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha: float = 0.2  # LeakyReLU negative slope
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Linear transformations for each head
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * self.head_dim, 1)))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(
        self,
        node_features: torch.Tensor,  # [N, in_dim]
        edge_index: torch.Tensor,  # [2, E] (source, target)
        edge_attr: Optional[torch.Tensor] = None  # [E, edge_dim] (optional relation embeddings)
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            node_features: Node feature matrix [N, in_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Optional edge attributes [E, edge_dim]
        
        Returns:
            Output node features [N, out_dim]
        """
        N = node_features.shape[0]
        device = node_features.device
        
        # Transform features
        h = self.W(node_features)  # [N, out_dim]
        h = h.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        
        # Compute attention scores
        # For each edge (i, j), compute attention from i to j
        if edge_index.shape[1] > 0:
            source, target = edge_index[0], edge_index[1]
            
            # Get source and target features
            h_source = h[source]  # [E, num_heads, head_dim]
            h_target = h[target]  # [E, num_heads, head_dim]
            
            # Concatenate for attention computation
            h_concat = torch.cat([h_source, h_target], dim=-1)  # [E, num_heads, 2*head_dim]
            
            # Compute attention scores
            attention_scores = torch.matmul(h_concat, self.a)  # [E, num_heads, 1]
            attention_scores = attention_scores.squeeze(-1)  # [E, num_heads]
            attention_scores = self.leaky_relu(attention_scores)
            
            # Apply edge attributes if provided (e.g., relation embeddings)
            if edge_attr is not None:
                # Simple addition of edge attributes to attention
                edge_attr_expanded = edge_attr.unsqueeze(1).expand(-1, self.num_heads, -1)
                # Project edge attributes to scalar
                if edge_attr_expanded.shape[-1] == self.head_dim:
                    edge_contribution = (edge_attr_expanded * h_target).sum(dim=-1, keepdim=True)
                    attention_scores = attention_scores + edge_contribution.squeeze(-1)
            
            # Create attention matrix (sparse)
            attention_matrix = torch.zeros(N, N, self.num_heads, device=device)
            attention_matrix[source, target] = attention_scores
            
            # Apply softmax
            attention_matrix = F.softmax(attention_matrix, dim=1)  # [N, N, num_heads]
            attention_matrix = self.dropout(attention_matrix)
            
            # Aggregate neighbors
            h_out = torch.zeros(N, self.num_heads, self.head_dim, device=device)
            for i in range(N):
                neighbors = target[source == i]
                if len(neighbors) > 0:
                    attn_weights = attention_matrix[i, neighbors]  # [num_neighbors, num_heads]
                    neighbor_features = h[neighbors]  # [num_neighbors, num_heads, head_dim]
                    # Weighted sum
                    h_out[i] = (attn_weights.unsqueeze(-1) * neighbor_features).sum(dim=0)
                else:
                    h_out[i] = h[i]  # Self-connection (if no neighbors, use self)
        else:
            # No edges, just return transformed features
            h_out = h
        
        # Reshape and return
        h_out = h_out.view(N, self.out_dim)
        return h_out


class GNNEncoder(nn.Module):
    """Graph Neural Network encoder using Graph Attention Networks."""
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        gnn_type: str = "gat"
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        
        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == "gat":
                self.gnn_layers.append(
                    GraphAttentionLayer(
                        in_dim=hidden_dim if i > 0 else hidden_dim,
                        out_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout
                    )
                )
            else:
                # Simple GCN-style layer
                self.gnn_layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,  # [N, node_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: Optional[torch.Tensor] = None  # [E, edge_dim]
    ) -> torch.Tensor:
        """
        Encode graph structure.
        
        Returns:
            Encoded node features [N, hidden_dim]
        """
        # Project input
        x = self.input_proj(node_features)  # [N, hidden_dim]
        
        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            if self.gnn_type == "gat":
                x_new = layer(x, edge_index, edge_attr)
            else:
                # Simple message passing for GCN
                # Aggregate neighbors
                if edge_index.shape[1] > 0:
                    source, target = edge_index[0], edge_index[1]
                    # Simple mean aggregation
                    neighbor_agg = torch.zeros_like(x)
                    for node_idx in range(x.shape[0]):
                        neighbors = target[source == node_idx]
                        if len(neighbors) > 0:
                            neighbor_agg[node_idx] = x[neighbors].mean(dim=0)
                        else:
                            neighbor_agg[node_idx] = x[node_idx]
                    x_new = layer(x + neighbor_agg)
                else:
                    x_new = layer(x)
            
            # Residual connection
            x = x + self.dropout(x_new)
            x = self.layer_norm(x)
        
        return x


class AutoregressiveDecoder(nn.Module):
    """Causal transformer decoder (reused from autoregressive model)."""
    
    def __init__(
        self,
        num_relations: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_path_length: int = 25,
        dropout: float = 0.1,
        context_dim: int = 256
    ):
        super().__init__()
        
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        
        # Relation embeddings
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        
        # Position embeddings
        self.pos_embedding = nn.Embedding(max_path_length, hidden_dim)
        
        # Cross-attention to context (question + graph)
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
        
        # Project context to hidden_dim if needed
        if context_dim != hidden_dim:
            self.context_proj = nn.Linear(context_dim, hidden_dim)
        else:
            self.context_proj = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        relation_ids: torch.Tensor,  # [B, seq_len]
        context: torch.Tensor,  # [B, context_len, context_dim]
        context_mask: Optional[torch.Tensor] = None,  # [B, context_len] True for padding
        relation_mask: Optional[torch.Tensor] = None  # [B, seq_len] True for padding
    ) -> torch.Tensor:
        """Forward pass."""
        B, seq_len = relation_ids.shape
        device = relation_ids.device
        
        # Embed relations
        relation_emb = self.relation_embedding(relation_ids)
        
        # Add position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)
        x = relation_emb + pos_emb
        x = self.dropout(x)
        
        # Project context
        context_proj = self.context_proj(context)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        
        # Apply decoder layers
        for i in range(len(self.self_attn_layers)):
            # Self-attention
            residual = x
            x = self.norm1_layers[i](x)
            attn_out, _ = self.self_attn_layers[i](
                x, x, x,
                attn_mask=causal_mask,
                key_padding_mask=relation_mask
            )
            x = residual + self.dropout(attn_out)
            
            # Cross-attention to context
            residual = x
            x = self.norm2_layers[i](x)
            cross_attn_out, _ = self.cross_attn_layers[i](
                x, context_proj, context_proj,
                key_padding_mask=context_mask
            )
            x = residual + self.dropout(cross_attn_out)
            
            # Feed-forward
            residual = x
            x = self.norm3_layers[i](x)
            ffn_out = self.ffn_layers[i](x)
            x = residual + ffn_out
        
        # Project to relation vocabulary
        relation_logits = self.relation_head(x)
        
        return relation_logits
    
    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        max_length: int = 25,
        decoding_strategy: str = "greedy",
        beam_width: int = 5,
        top_p: float = 0.9,
        top_k: int = 50,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate relation chains (simplified greedy for now)."""
        B = context.shape[0]
        device = context.device
        
        context_proj = self.context_proj(context)
        
        relation_ids = torch.zeros(B, 0, dtype=torch.long, device=device)
        
        for step in range(max_length):
            if relation_ids.shape[1] == 0:
                dummy_input = torch.zeros(B, 1, dtype=torch.long, device=device)
            else:
                dummy_input = relation_ids
            
            logits = self.forward(dummy_input, context_proj, context_mask)
            if relation_ids.shape[1] == 0:
                next_logits = logits[:, 0, :] / temperature
            else:
                next_logits = logits[:, -1, :] / temperature
            
            # Mask out padding token
            next_logits[:, 0] = float('-inf')
            
            if decoding_strategy == "greedy":
                next_ids = next_logits.argmax(dim=-1, keepdim=True)
            elif decoding_strategy == "nucleus":
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumprobs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(next_logits, dim=-1)
                next_ids = torch.multinomial(probs, 1)
            else:
                next_ids = next_logits.argmax(dim=-1, keepdim=True)
            
            relation_ids = torch.cat([relation_ids, next_ids], dim=1)
        
        return relation_ids, None


class KGPathGNNDecoderModel(nn.Module):
    """
    GNN + Decoder model for KG Path Generation.
    
    Architecture:
    1. Question Encoder: Pretrained transformer
    2. GNN Encoder: Graph Attention Network to encode KG structure
    3. Autoregressive Decoder: Generate relation chains
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 256,
        question_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_question_encoder: bool = False,
        gnn_type: str = "gat",
        gnn_layers: int = 3,
        gnn_heads: int = 8,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        max_path_length: int = 25,
        dropout: float = 0.1,
        use_graph_structure: bool = True
    ):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        self.use_graph_structure = use_graph_structure
        
        # Question encoder
        self.question_encoder = QuestionEncoder(
            model_name=question_encoder_name,
            output_dim=hidden_dim,
            freeze=freeze_question_encoder
        )
        
        # Entity embeddings (for graph nodes)
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
        
        # Relation embeddings (for graph edges)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        
        # GNN encoder
        if use_graph_structure:
            self.gnn_encoder = GNNEncoder(
                node_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=gnn_layers,
                num_heads=gnn_heads,
                dropout=dropout,
                gnn_type=gnn_type
            )
        else:
            self.gnn_encoder = None
        
        # Autoregressive decoder
        self.decoder = AutoregressiveDecoder(
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            max_path_length=max_path_length,
            dropout=dropout,
            context_dim=hidden_dim * 2  # Question + Graph context
        )
        
        # Projection to combine question and graph context
        self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def encode_inputs(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        graph_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode question and optionally graph structure.
        
        Args:
            question_input_ids: [B, q_len]
            question_attention_mask: [B, q_len]
            graph_data: Optional dict with:
                - node_ids: [N] Entity IDs
                - edge_index: [2, E] Edge connectivity
                - edge_relation_ids: [E] Relation IDs
        
        Returns:
            context: [B, context_len, hidden_dim] Combined context
            question_pooled: [B, hidden_dim] Question representation
        """
        # Encode question
        question_seq, question_pooled = self.question_encoder(
            question_input_ids, question_attention_mask
        )
        
        if self.use_graph_structure and graph_data is not None:
            # Encode graph
            node_ids = graph_data['node_ids']  # [N]
            edge_index = graph_data['edge_index']  # [2, E]
            edge_relation_ids = graph_data.get('edge_relation_ids', None)  # [E]
            
            # Get node features
            node_features = self.entity_embedding(node_ids)  # [N, hidden_dim]
            
            # Get edge features if available
            edge_attr = None
            if edge_relation_ids is not None:
                edge_attr = self.relation_embedding(edge_relation_ids)  # [E, hidden_dim]
            
            # Encode with GNN
            graph_features = self.gnn_encoder(
                node_features,
                edge_index,
                edge_attr
            )  # [N, hidden_dim]
            
            # Pool graph features (mean pooling)
            graph_pooled = graph_features.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            graph_pooled = graph_pooled.expand(question_pooled.shape[0], -1)  # [B, hidden_dim]
        else:
            # No graph, use zero features
            graph_pooled = torch.zeros_like(question_pooled)
        
        # Combine question and graph
        combined_context = torch.cat([question_pooled, graph_pooled], dim=-1)  # [B, 2*hidden_dim]
        combined_context = self.context_proj(combined_context)  # [B, hidden_dim]
        
        # Expand to sequence format for decoder
        # Use question sequence as base and add graph context
        context_seq = question_seq  # [B, q_len, hidden_dim]
        graph_context = combined_context.unsqueeze(1)  # [B, 1, hidden_dim]
        context = torch.cat([context_seq, graph_context], dim=1)  # [B, q_len+1, hidden_dim]
        
        return context, question_pooled
    
    def forward(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        target_relations: torch.Tensor,
        relation_mask: Optional[torch.Tensor] = None,
        graph_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass."""
        # Encode inputs
        context, _ = self.encode_inputs(
            question_input_ids,
            question_attention_mask,
            graph_data
        )
        
        # Create context mask
        context_mask = None
        if question_attention_mask is not None:
            # Add one for graph context (always valid)
            context_mask = ~question_attention_mask.bool()  # True for padding
            graph_mask = torch.zeros(context_mask.shape[0], 1, dtype=torch.bool, device=context_mask.device)
            context_mask = torch.cat([context_mask, graph_mask], dim=1)
        
        # Get logits
        relation_logits = self.decoder(
            target_relations,
            context,
            context_mask=context_mask,
            relation_mask=relation_mask
        )
        
        # Compute loss
        if relation_mask is not None:
            valid_mask = ~relation_mask
            if valid_mask.any():
                flat_logits = relation_logits[valid_mask]
                flat_targets = target_relations[valid_mask]
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
        num_paths: torch.Tensor,
        graph_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Multi-path training."""
        B = question_input_ids.shape[0]
        max_paths = all_target_relations.shape[1]
        rel_len = all_target_relations.shape[2]
        device = question_input_ids.device
        
        # Encode inputs once
        context, _ = self.encode_inputs(
            question_input_ids,
            question_attention_mask,
            graph_data
        )
        
        # Expand context for all paths
        context_expanded = context.unsqueeze(1).expand(-1, max_paths, -1, -1)
        context_expanded = context_expanded.reshape(B * max_paths, *context.shape[1:])
        
        # Create context mask
        context_mask = None
        if question_attention_mask is not None:
            q_mask = ~question_attention_mask.bool()
            graph_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            context_mask_full = torch.cat([q_mask, graph_mask], dim=1)
            context_mask_expanded = context_mask_full.unsqueeze(1).expand(-1, max_paths, -1)
            context_mask_expanded = context_mask_expanded.reshape(B * max_paths, -1)
            context_mask = context_mask_expanded
        
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
            context_expanded,
            context_mask=context_mask,
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
        graph_data: Optional[Dict[str, torch.Tensor]] = None,
        decoding_strategy: str = "greedy",
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate relation chains."""
        context, _ = self.encode_inputs(
            question_input_ids,
            question_attention_mask,
            graph_data
        )
        
        context_mask = None
        if question_attention_mask is not None:
            q_mask = ~question_attention_mask.bool()
            graph_mask = torch.zeros(q_mask.shape[0], 1, dtype=torch.bool, device=q_mask.device)
            context_mask = torch.cat([q_mask, graph_mask], dim=1)
        
        return self.decoder.generate(
            context,
            context_mask=context_mask,
            max_length=path_length,
            decoding_strategy=decoding_strategy,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
    
    @torch.no_grad()
    def generate_multiple(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        num_paths: int = 5,
        path_length: int = 10,
        graph_data: Optional[Dict[str, torch.Tensor]] = None,
        temperature: float = 1.0,
        decoding_strategy: str = "greedy",
        diversity_penalty: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate multiple diverse paths."""
        B = question_input_ids.shape[0]
        device = question_input_ids.device
        
        all_relations = []
        
        for path_idx in range(num_paths):
            if path_idx > 0 and diversity_penalty > 0:
                curr_temp = temperature * (1 + diversity_penalty * path_idx)
            else:
                curr_temp = temperature
            
            relations, _ = self.generate(
                question_input_ids,
                question_attention_mask,
                path_length=path_length,
                graph_data=graph_data,
                decoding_strategy=decoding_strategy,
                temperature=curr_temp
            )
            
            all_relations.append(relations)
        
        all_relations = torch.stack(all_relations, dim=1)
        dummy_entities = torch.zeros(B, num_paths, path_length, dtype=torch.long, device=device)
        
        return dummy_entities, all_relations


class KGPathGNNDecoderLightning(pl.LightningModule):
    """PyTorch Lightning module for GNN + Decoder KG Path Generation."""
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 256,
        question_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_question_encoder: bool = False,
        gnn_type: str = "gat",
        gnn_layers: int = 3,
        gnn_heads: int = 8,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        max_path_length: int = 25,
        dropout: float = 0.1,
        use_graph_structure: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        use_multipath_training: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = KGPathGNNDecoderModel(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            question_encoder_name=question_encoder_name,
            freeze_question_encoder=freeze_question_encoder,
            gnn_type=gnn_type,
            gnn_layers=gnn_layers,
            gnn_heads=gnn_heads,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            max_path_length=max_path_length,
            dropout=dropout,
            use_graph_structure=use_graph_structure
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.use_multipath_training = use_multipath_training
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        if self.use_multipath_training and 'all_path_relations' in batch:
            return self.forward_multipath(batch)
        else:
            return self.forward_single(batch)
    
    def forward_single(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Single path forward."""
        relation_mask = None
        if 'path_lengths' in batch:
            max_len = batch['path_relations'].shape[1]
            lengths = batch['path_lengths']
            relation_mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Graph data is optional - if not provided, model will use zero graph features
        graph_data = None
        
        return self.model(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            target_relations=batch['path_relations'],
            relation_mask=relation_mask,
            graph_data=graph_data
        )
    
    def forward_multipath(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Multi-path forward."""
        graph_data = None
        
        return self.model.forward_multipath(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            all_target_relations=batch['all_path_relations'],
            all_path_lengths=batch['all_path_lengths'],
            num_paths=batch['num_paths'],
            graph_data=graph_data
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

