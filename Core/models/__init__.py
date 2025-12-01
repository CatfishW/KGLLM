"""
Models package for KG Path Generation.

This package contains all model implementations:
- diffusion: Discrete diffusion model
- autoregressive: GPT-style autoregressive transformer
- gnn_decoder: GNN encoder + autoregressive decoder hybrid
"""

from .base import QuestionEncoder
from .diffusion import KGPathDiffusionModel, KGPathDiffusionLightning
from .autoregressive import KGPathAutoregressiveModel, KGPathAutoregressiveLightning
from .gnn_decoder import KGPathGNNDecoderModel, KGPathGNNDecoderLightning
from .factory import create_model

__all__ = [
    'QuestionEncoder',
    'KGPathDiffusionModel',
    'KGPathDiffusionLightning',
    'KGPathAutoregressiveModel',
    'KGPathAutoregressiveLightning',
    'KGPathGNNDecoderModel',
    'KGPathGNNDecoderLightning',
    'create_model',
]

