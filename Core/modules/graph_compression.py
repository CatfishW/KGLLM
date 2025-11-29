"""
Graph Compression Module for Large Graphs.

Converts large graphs with many nodes into fixed-size low-dimensional embeddings
that can be efficiently used in diffusion models. This allows handling graphs
with very large numbers of nodes without quadratic attention complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat


class AttentionBasedGraphCompression(nn.Module):
    """
    Compress large graphs using learnable attention-based pooling.
    
    Uses a set of learnable query vectors to attend over all graph nodes,
    producing a fixed-size set of compressed embeddings.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_compressed_nodes: int = 64,  # Fixed number of compressed nodes
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_compressed_nodes = num_compressed_nodes
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Learnable query vectors for compression
        self.compression_queries = nn.Parameter(
            torch.randn(num_compressed_nodes, output_dim)
        )
        
        # Projection layers for attention
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(input_dim, output_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize queries
        nn.init.xavier_uniform_(self.compression_queries)
    
    def forward(
        self,
        node_embeddings: torch.Tensor,  # [B, num_nodes, input_dim]
        node_mask: Optional[torch.Tensor] = None  # [B, num_nodes] (True = padding)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress graph nodes into fixed-size embeddings.
        
        Args:
            node_embeddings: Node embeddings from graph encoder
            node_mask: Mask indicating padding nodes (True = padding)
        
        Returns:
            compressed_embeddings: [B, num_compressed_nodes, output_dim]
            attention_weights: [B, num_compressed_nodes, num_nodes] (optional, for visualization)
        """
        B, num_nodes, _ = node_embeddings.shape
        
        # Expand queries for batch: [num_compressed_nodes, output_dim] -> [B, num_compressed_nodes, output_dim]
        queries = repeat(self.compression_queries, 'n d -> b n d', b=B)
        
        # Project queries, keys, values
        Q = self.q_proj(queries)  # [B, num_compressed_nodes, output_dim]
        K = self.k_proj(node_embeddings)  # [B, num_nodes, output_dim]
        V = self.v_proj(node_embeddings)  # [B, num_nodes, output_dim]
        
        # Reshape for multi-head attention
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute attention scores
        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / (self.head_dim ** 0.5)
        
        # Apply mask if provided (mask out padding nodes)
        if node_mask is not None:
            # node_mask: [B, num_nodes], True = padding
            # We want to mask out padding, so we set scores to -inf where mask is True
            mask_expanded = node_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, num_nodes]
            scores = scores.masked_fill(mask_expanded, float('-inf'))
        
        # Softmax attention
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, num_compressed_nodes, num_nodes]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        compressed = torch.einsum('bhqk,bhkd->bhqd', attn_weights, V)  # [B, num_heads, num_compressed_nodes, head_dim]
        compressed = rearrange(compressed, 'b h n d -> b n (h d)')  # [B, num_compressed_nodes, output_dim]
        
        # Output projection and residual
        compressed = self.out_proj(compressed)
        compressed = self.norm(compressed)
        
        # Average attention weights across heads for visualization
        attn_weights_avg = attn_weights.mean(dim=1)  # [B, num_compressed_nodes, num_nodes]
        
        return compressed, attn_weights_avg


class ClusterBasedGraphCompression(nn.Module):
    """
    Compress graphs using learnable clustering/assignment.
    
    Uses a set of learnable cluster centers and soft assignment to compress
    the graph into a fixed number of cluster embeddings.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_clusters: int = 64,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_clusters = num_clusters
        self.temperature = temperature
        
        # Learnable cluster centers
        self.cluster_centers = nn.Parameter(
            torch.randn(num_clusters, output_dim)
        )
        
        # Projection to match dimensions
        self.node_proj = nn.Linear(input_dim, output_dim)
        self.cluster_proj = nn.Linear(output_dim, output_dim)
        
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize cluster centers
        nn.init.xavier_uniform_(self.cluster_centers)
    
    def forward(
        self,
        node_embeddings: torch.Tensor,  # [B, num_nodes, input_dim]
        node_mask: Optional[torch.Tensor] = None  # [B, num_nodes] (True = padding)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress graph nodes using soft clustering.
        
        Returns:
            compressed_embeddings: [B, num_clusters, output_dim]
            assignment_weights: [B, num_clusters, num_nodes]
        """
        B, num_nodes, _ = node_embeddings.shape
        
        # Project nodes
        node_proj = self.node_proj(node_embeddings)  # [B, num_nodes, output_dim]
        
        # Expand cluster centers for batch
        cluster_centers = repeat(self.cluster_centers, 'c d -> b c d', b=B)  # [B, num_clusters, output_dim]
        cluster_centers_proj = self.cluster_proj(cluster_centers)
        
        # Compute similarity between nodes and cluster centers
        # [B, num_clusters, output_dim] x [B, num_nodes, output_dim]^T -> [B, num_clusters, num_nodes]
        similarities = torch.bmm(cluster_centers_proj, node_proj.transpose(1, 2)) / self.temperature
        
        # Apply mask if provided
        if node_mask is not None:
            # Mask out padding nodes
            mask_expanded = node_mask.unsqueeze(1)  # [B, 1, num_nodes]
            similarities = similarities.masked_fill(mask_expanded, float('-inf'))
        
        # Soft assignment weights
        assignment_weights = F.softmax(similarities, dim=-1)  # [B, num_clusters, num_nodes]
        assignment_weights = self.dropout(assignment_weights)
        
        # Weighted aggregation of nodes to clusters
        # [B, num_clusters, num_nodes] x [B, num_nodes, output_dim] -> [B, num_clusters, output_dim]
        compressed = torch.bmm(assignment_weights, node_proj)
        
        # Add cluster center as residual
        compressed = compressed + cluster_centers_proj
        compressed = self.norm(compressed)
        
        return compressed, assignment_weights


class HierarchicalGraphCompression(nn.Module):
    """
    Hierarchical compression: first pool locally, then compress globally.
    
    This is more efficient for very large graphs as it reduces computation
    in multiple stages.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_compressed_nodes: int = 64,
        intermediate_pool_size: int = 256,  # First reduce to this many nodes
        compression_type: str = "attention",  # "attention" or "cluster"
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_compressed_nodes = num_compressed_nodes
        self.intermediate_pool_size = intermediate_pool_size
        self.compression_type = compression_type
        
        # First stage: reduce to intermediate size
        if compression_type == "attention":
            self.stage1 = AttentionBasedGraphCompression(
                input_dim, output_dim, intermediate_pool_size, num_heads, dropout
            )
        else:
            self.stage1 = ClusterBasedGraphCompression(
                input_dim, output_dim, intermediate_pool_size, dropout=dropout
            )
        
        # Second stage: reduce to final compressed size
        if compression_type == "attention":
            self.stage2 = AttentionBasedGraphCompression(
                output_dim, output_dim, num_compressed_nodes, num_heads, dropout
            )
        else:
            self.stage2 = ClusterBasedGraphCompression(
                output_dim, output_dim, num_compressed_nodes, dropout=dropout
            )
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Two-stage compression.
        
        Returns:
            compressed_embeddings: [B, num_compressed_nodes, output_dim]
            attention_weights: [B, num_compressed_nodes, num_nodes] (approximate)
        """
        # First stage compression
        intermediate, _ = self.stage1(node_embeddings, node_mask)
        
        # Second stage compression (no mask needed for intermediate)
        compressed, attn_weights = self.stage2(intermediate, None)
        
        return compressed, attn_weights


class GraphCompressionWrapper(nn.Module):
    """
    Wrapper that integrates graph compression with graph encoding.
    
    Takes raw graph data, encodes it, and compresses to fixed size.
    """
    
    def __init__(
        self,
        graph_encoder: nn.Module,  # The existing graph encoder
        input_dim: int,
        output_dim: int,
        num_compressed_nodes: int = 64,
        compression_method: str = "attention",  # "attention", "cluster", or "hierarchical"
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.graph_encoder = graph_encoder
        self.compression_method = compression_method
        
        # Create compression module based on method
        if compression_method == "attention":
            self.compressor = AttentionBasedGraphCompression(
                input_dim, output_dim, num_compressed_nodes, num_heads, dropout
            )
        elif compression_method == "cluster":
            self.compressor = ClusterBasedGraphCompression(
                input_dim, output_dim, num_compressed_nodes, dropout=dropout
            )
        elif compression_method == "hierarchical":
            self.compressor = HierarchicalGraphCompression(
                input_dim, output_dim, num_compressed_nodes,
                compression_type="attention", num_heads=num_heads, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown compression_method: {compression_method}")
    
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode and compress graph.
        
        Returns:
            compressed_embeddings: [B, num_compressed_nodes, output_dim]
            graph_pooled: [B, output_dim] (global graph embedding)
            node_mask: [B, num_compressed_nodes] (always False, since fixed size)
        """
        # First encode the graph using the existing encoder
        node_embeddings, graph_pooled = self.graph_encoder(
            node_ids, edge_index, edge_type, batch,
            node_input_ids, node_attention_mask, text_encoder, chunk_size
        )
        
        # Reshape node embeddings to [B, num_nodes, dim] format
        if batch is not None:
            batch_size = batch.max().item() + 1
            device = node_embeddings.device
            
            # Get number of nodes per graph
            nodes_per_graph = torch.bincount(batch, minlength=batch_size)
            max_nodes = nodes_per_graph.max().item()
            
            # Create padded tensor
            node_emb_padded = torch.zeros(
                batch_size, max_nodes, node_embeddings.shape[-1],
                device=device
            )
            
            # Create mask (True = padding)
            node_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)
            
            # Fill in node embeddings
            node_idx = 0
            for i in range(batch_size):
                n_nodes = nodes_per_graph[i].item()
                node_emb_padded[i, :n_nodes] = node_embeddings[node_idx:node_idx + n_nodes]
                node_mask[i, :n_nodes] = False
                node_idx += n_nodes
        else:
            # Single graph case
            batch_size = 1
            node_emb_padded = node_embeddings.unsqueeze(0)
            node_mask = torch.zeros(1, node_embeddings.shape[0], dtype=torch.bool, device=node_embeddings.device)
        
        # Compress to fixed size
        compressed, _ = self.compressor(node_emb_padded, node_mask)
        
        # Create mask for compressed (always False since it's fixed size)
        compressed_mask = torch.zeros(
            batch_size, compressed.shape[1],
            dtype=torch.bool, device=compressed.device
        )
        
        return compressed, graph_pooled, compressed_mask

