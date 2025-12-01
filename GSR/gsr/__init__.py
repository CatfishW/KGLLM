"""
GSR (Generative Subgraph Retrieval) Module.

Provides subgraph ID generation and retrieval functionality.
"""

from .subgraph_index import SubgraphIndex, SubgraphPattern, build_subgraph_index_from_dataset

__all__ = [
    'SubgraphIndex',
    'SubgraphPattern',
    'build_subgraph_index_from_dataset'
]

