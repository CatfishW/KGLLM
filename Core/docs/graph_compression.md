# Graph Compression for Large Graphs

## Overview

The graph compression module enables handling very large graphs (with thousands of nodes) in diffusion-based path generation by compressing graphs into fixed-size low-dimensional embeddings. This avoids the quadratic complexity of attention mechanisms over large node sets.

## Architecture

The compression module provides three compression methods:

1. **Attention-Based Compression**: Uses learnable query vectors to attend over all graph nodes, producing a fixed-size set of compressed embeddings.
2. **Cluster-Based Compression**: Uses learnable cluster centers with soft assignment to compress graphs.
3. **Hierarchical Compression**: Two-stage compression that first reduces to an intermediate size, then to the final compressed size (most efficient for very large graphs).

## Usage

### Basic Usage

To enable graph compression, set `use_graph_compression=True` when initializing the model:

```python
from kg_path_diffusion import KGPathDiffusionModel

model = KGPathDiffusionModel(
    num_entities=50000,
    num_relations=5000,
    hidden_dim=256,
    use_graph_compression=True,  # Enable compression
    num_compressed_nodes=64,      # Fixed number of compressed nodes
    compression_method="attention"  # "attention", "cluster", or "hierarchical"
)
```

### Compression Methods

#### 1. Attention-Based (Recommended for most cases)

```python
model = KGPathDiffusionModel(
    # ... other parameters ...
    use_graph_compression=True,
    num_compressed_nodes=64,
    compression_method="attention"
)
```

- **Pros**: Most flexible, learns to attend to important nodes
- **Cons**: Slightly more compute than cluster-based
- **Best for**: Medium to large graphs (100-1000 nodes)

#### 2. Cluster-Based

```python
model = KGPathDiffusionModel(
    # ... other parameters ...
    use_graph_compression=True,
    num_compressed_nodes=64,
    compression_method="cluster"
)
```

- **Pros**: Faster, simpler computation
- **Cons**: Less flexible than attention
- **Best for**: Large graphs with clear structure

#### 3. Hierarchical (Recommended for very large graphs)

```python
model = KGPathDiffusionModel(
    # ... other parameters ...
    use_graph_compression=True,
    num_compressed_nodes=64,
    compression_method="hierarchical"
)
```

- **Pros**: Most efficient for very large graphs (1000+ nodes)
- **Cons**: More parameters, slightly more complex
- **Best for**: Very large graphs (1000+ nodes)

### Choosing num_compressed_nodes

- **Small graphs (< 100 nodes)**: 32-64 compressed nodes
- **Medium graphs (100-500 nodes)**: 64-128 compressed nodes
- **Large graphs (500-2000 nodes)**: 128-256 compressed nodes
- **Very large graphs (> 2000 nodes)**: 256-512 compressed nodes

Note: More compressed nodes = better representation but more computation. Start with 64 and increase if needed.

## Benefits

1. **Scalability**: Handle graphs with thousands of nodes without OOM errors
2. **Efficiency**: Fixed-size embeddings reduce attention complexity from O(n²) to O(k²) where k << n
3. **Flexibility**: Learnable compression adapts to different graph structures
4. **Backward Compatible**: Can be disabled to use original behavior

## Implementation Details

The compression happens after graph encoding:

1. Graph encoder produces node embeddings: `[num_nodes, hidden_dim]`
2. Compression module reduces to fixed size: `[num_compressed_nodes, hidden_dim]`
3. Diffusion model uses compressed embeddings for cross-attention

The compression is fully differentiable and learned end-to-end with the rest of the model.

## Example: Training with Compression

```python
from kg_path_diffusion import KGPathDiffusionLightning

# Initialize model with compression
model = KGPathDiffusionLightning(
    num_entities=50000,
    num_relations=5000,
    hidden_dim=256,
    use_graph_compression=True,
    num_compressed_nodes=128,
    compression_method="hierarchical"
)

# Training proceeds as normal
trainer.fit(model, train_dataloader, val_dataloader)
```

## Performance Considerations

- **Memory**: Compression reduces memory usage significantly for large graphs
- **Speed**: Attention-based is slightly slower than cluster-based, hierarchical is fastest for very large graphs
- **Quality**: Compression quality depends on `num_compressed_nodes` - more nodes = better quality but slower

## Troubleshooting

### Out of Memory with Compression

- Reduce `num_compressed_nodes`
- Use `compression_method="hierarchical"` for very large graphs
- Reduce `hidden_dim`

### Poor Path Generation Quality

- Increase `num_compressed_nodes`
- Try `compression_method="attention"` for better flexibility
- Check if compression is too aggressive (too few compressed nodes)

