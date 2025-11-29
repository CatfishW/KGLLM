# Deprecated Code

This folder contains code that is no longer used in the current model implementation.

## Files

### `graph_encoder.py`
- **Purpose**: Graph encoder using PyTorch Geometric (R-GCN, GAT, TransformerConv)
- **Reason for deprecation**: The current model does not use graph neural networks. It uses a transformer-based architecture (QuestionEncoder + PathDiffusionTransformer) without graph processing.
- **Dependencies**: `torch-geometric`, `torch-scatter`, `torch-sparse`, `torch-cluster`

### `debug_collate.py`
- **Purpose**: Debug script for testing dataset collation with PyTorch Geometric Batch
- **Reason for deprecation**: Uses torch_geometric.data.Batch which is no longer needed

### `test_compression_integration.py`
- **Purpose**: Test script for graph compression integration
- **Reason for deprecation**: Tests graph compression features that depend on torch_geometric

### `test_debug.py`
- **Purpose**: Debug/test script for model creation with graph encoder
- **Reason for deprecation**: Imports and tests the deprecated graph_encoder module

## Note

These files are kept for reference but are not part of the active codebase. The current model implementation is purely transformer-based and does not require PyTorch Geometric.

