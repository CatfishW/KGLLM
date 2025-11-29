"""
Graph Encoder using PyTorch Geometric.
Efficient message passing for encoding KG subgraphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, 
    RGCNConv, 
    TransformerConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool
)
from torch_geometric.data import Batch
from typing import Optional, Tuple


class RelationalGraphEncoder(nn.Module):
    """
    Relational Graph Convolutional Network for encoding KG subgraphs.
    
    Uses R-GCN which is specifically designed for knowledge graphs
    with multiple relation types.
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        num_bases: int = 30,  # For weight decomposition in RGCN
        dropout: float = 0.1,
        use_entity_embeddings: bool = True
    ):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_entity_embeddings = use_entity_embeddings
        
        # Entity embeddings
        if use_entity_embeddings:
            self.entity_embedding = nn.Embedding(num_entities, entity_dim, padding_idx=0)
        else:
            self.entity_embedding = None
        
        # R-GCN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        in_dim = entity_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.layers.append(
                RGCNConv(
                    in_dim, 
                    out_dim, 
                    num_relations=num_relations,
                    num_bases=min(num_bases, num_relations),
                    aggr='mean'
                )
            )
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        node_input_ids: Optional[torch.Tensor] = None,
        node_attention_mask: Optional[torch.Tensor] = None,
        text_encoder: Optional[nn.Module] = None,
        chunk_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph nodes.
        
        Args:
            node_ids: [num_nodes] tensor of entity indices
            edge_index: [2, num_edges] tensor
            edge_type: [num_edges] tensor of relation indices
            batch: [num_nodes] tensor indicating which graph each node belongs to
            node_input_ids: [num_nodes, seq_len] tokenized text
            node_attention_mask: [num_nodes, seq_len] attention mask
            text_encoder: Encoder module to use if use_entity_embeddings is False
            chunk_size: Number of nodes to process at once when using text encoder (to avoid OOM)
        
        Returns:
            node_embeddings: [num_nodes, output_dim]
            graph_embedding: [batch_size, output_dim] (if batch is provided)
        """
        # Get initial embeddings
        if self.use_entity_embeddings:
            x = self.entity_embedding(node_ids)
        elif text_encoder is not None and node_input_ids is not None:
            # Use text encoder with chunked processing to avoid OOM for large graphs.
            # Disable gradients (and dropout) for node text encoding to prevent huge
            # computation graphs and GPU memory spikes when processing millions of nodes.
            num_nodes = node_input_ids.shape[0]
            text_encoder_was_training = getattr(text_encoder, "training", False)
            prev_grad_state = torch.is_grad_enabled()
            if text_encoder_was_training:
                text_encoder.eval()
            torch.set_grad_enabled(False)
            try:
                if num_nodes > chunk_size:
                    # Process in chunks
                    x_chunks = []
                    for i in range(0, num_nodes, chunk_size):
                        end_idx = min(i + chunk_size, num_nodes)
                        chunk_input_ids = node_input_ids[i:end_idx]
                        chunk_attention_mask = node_attention_mask[i:end_idx]
                        _, chunk_x = text_encoder(chunk_input_ids, chunk_attention_mask)
                        x_chunks.append(chunk_x.detach())
                    x = torch.cat(x_chunks, dim=0)
                else:
                    # Process all at once if small enough
                    _, x = text_encoder(node_input_ids, node_attention_mask)
                    x = x.detach()
            finally:
                torch.set_grad_enabled(prev_grad_state)
                if text_encoder_was_training:
                    text_encoder.train()
        else:
            raise ValueError("Must provide text_encoder and node inputs when use_entity_embeddings is False")
        
        # Apply R-GCN layers
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x_new = layer(x, edge_index, edge_type)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            
            # Residual connection if dimensions match
            if x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new
        
        # Get graph-level embedding if batch is provided
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
        
        return x, graph_embedding


class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer for encoding KG subgraphs.
    More expressive than RGCN, uses attention mechanism.
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Entity and relation embeddings
        self.entity_embedding = nn.Embedding(num_entities, entity_dim, padding_idx=0)
        self.relation_embedding = nn.Embedding(num_relations, entity_dim, padding_idx=0)
        
        # Graph Transformer layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        in_dim = entity_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.layers.append(
                TransformerConv(
                    in_dim,
                    out_dim // num_heads,
                    heads=num_heads,
                    edge_dim=entity_dim,  # Use relation embeddings as edge features
                    dropout=dropout,
                    concat=True
                )
            )
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph nodes using Graph Transformer.
        """
        # Get initial embeddings
        x = self.entity_embedding(node_ids)
        edge_attr = self.relation_embedding(edge_type)
        
        # Apply Transformer layers
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x_new = layer(x, edge_index, edge_attr)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            
            # Residual connection
            if x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new
        
        # Get graph-level embedding
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
        
        return x, graph_embedding


class HybridGraphEncoder(nn.Module):
    """
    Hybrid encoder combining RGCN for structure and Transformer for reasoning.
    Best of both worlds approach.
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_rgcn_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        num_bases: int = 30,
        dropout: float = 0.1,
        use_entity_embeddings: bool = True
    ):
        super().__init__()
        
        self.use_entity_embeddings = use_entity_embeddings
        
        # Entity and relation embeddings
        if use_entity_embeddings:
            self.entity_embedding = nn.Embedding(num_entities, entity_dim, padding_idx=0)
        else:
            self.entity_embedding = None
            
        self.relation_embedding = nn.Embedding(num_relations, entity_dim, padding_idx=0)
        
        # RGCN layers for structural encoding
        self.rgcn_layers = nn.ModuleList()
        self.rgcn_norms = nn.ModuleList()
        
        in_dim = entity_dim
        for i in range(num_rgcn_layers):
            self.rgcn_layers.append(
                RGCNConv(in_dim, hidden_dim, num_relations, num_bases=min(num_bases, num_relations))
            )
            self.rgcn_norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        
        # Transformer layers for reasoning
        self.transformer_layers = nn.ModuleList()
        self.transformer_norms = nn.ModuleList()
        
        for i in range(num_transformer_layers):
            out_dim = output_dim if i == num_transformer_layers - 1 else hidden_dim
            self.transformer_layers.append(
                TransformerConv(
                    hidden_dim, out_dim // num_heads, heads=num_heads,
                    edge_dim=entity_dim, dropout=dropout, concat=True
                )
            )
            self.transformer_norms.append(nn.LayerNorm(out_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
    
    def forward(
        self,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        node_input_ids: Optional[torch.Tensor] = None,
        node_attention_mask: Optional[torch.Tensor] = None,
        text_encoder: Optional[nn.Module] = None,
        chunk_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph with hybrid RGCN + Transformer architecture.
        """
        if self.use_entity_embeddings:
            x = self.entity_embedding(node_ids)
        elif text_encoder is not None and node_input_ids is not None:
            # Use text encoder with chunked processing to avoid OOM for large graphs.
            # Disable gradients/dropout during node text encoding to keep memory usage low.
            num_nodes = node_input_ids.shape[0]
            text_encoder_was_training = getattr(text_encoder, "training", False)
            prev_grad_state = torch.is_grad_enabled()
            if text_encoder_was_training:
                text_encoder.eval()
            torch.set_grad_enabled(False)
            try:
                if num_nodes > chunk_size:
                    # Process in chunks
                    x_chunks = []
                    for i in range(0, num_nodes, chunk_size):
                        end_idx = min(i + chunk_size, num_nodes)
                        chunk_input_ids = node_input_ids[i:end_idx]
                        chunk_attention_mask = node_attention_mask[i:end_idx]
                        _, chunk_x = text_encoder(chunk_input_ids, chunk_attention_mask)
                        x_chunks.append(chunk_x.detach())
                    x = torch.cat(x_chunks, dim=0)
                else:
                    # Process all at once if small enough
                    _, x = text_encoder(node_input_ids, node_attention_mask)
                    x = x.detach()
            finally:
                torch.set_grad_enabled(prev_grad_state)
                if text_encoder_was_training:
                    text_encoder.train()
        else:
            raise ValueError("Must provide text_encoder and node inputs when use_entity_embeddings is False")
            
        edge_attr = self.relation_embedding(edge_type)
        
        # RGCN layers
        for layer, norm in zip(self.rgcn_layers, self.rgcn_norms):
            x = layer(x, edge_index, edge_type)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Transformer layers
        for layer, norm in zip(self.transformer_layers, self.transformer_norms):
            x_new = layer(x, edge_index, edge_attr)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            if x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new
        
        # Graph pooling
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
        
        return x, graph_embedding

