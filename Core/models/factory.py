"""
Model Factory - Creates appropriate model based on configuration.

Supports:
- diffusion: Original diffusion model
- autoregressive: GPT-style autoregressive transformer
- gnn_decoder: GNN encoder + autoregressive decoder hybrid
"""

from typing import Dict, Any, Optional
import pytorch_lightning as pl


def create_model(
    model_type: str,
    num_entities: int,
    num_relations: int,
    config: Dict[str, Any]
) -> pl.LightningModule:
    """
    Create a model based on model_type.
    
    Args:
        model_type: One of "diffusion", "autoregressive", "gnn_decoder"
        num_entities: Number of entities in vocabulary
        num_relations: Number of relations in vocabulary
        config: Configuration dictionary with model parameters
    
    Returns:
        PyTorch Lightning module
    """
    if model_type == "diffusion":
        from .diffusion import KGPathDiffusionLightning
        
        return KGPathDiffusionLightning(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=config.get('hidden_dim', 256),
            question_encoder_name=config.get('question_encoder', 'sentence-transformers/all-MiniLM-L6-v2'),
            freeze_question_encoder=config.get('freeze_question_encoder', False),
            num_diffusion_layers=config.get('num_diffusion_layers', 6),
            num_heads=config.get('num_heads', 8),
            num_diffusion_steps=config.get('num_diffusion_steps', 1000),
            max_path_length=config.get('max_path_length', 25),
            dropout=config.get('dropout', 0.1),
            learning_rate=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
            warmup_steps=config.get('warmup_steps', 1000),
            max_steps=config.get('max_steps', 100000),
            use_entity_embeddings=config.get('use_entity_embeddings', True),
            predict_entities=config.get('predict_entities', True),
            use_causal_attention=config.get('use_causal_attention', True),
            predict_hop_count=config.get('predict_hop_count', True),
            hop_count_loss_weight=config.get('hop_count_loss_weight', 0.5)
        )
    
    elif model_type == "autoregressive":
        from .autoregressive import KGPathAutoregressiveLightning
        
        return KGPathAutoregressiveLightning(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=config.get('hidden_dim', 256),
            question_encoder_name=config.get('question_encoder', 'sentence-transformers/all-MiniLM-L6-v2'),
            freeze_question_encoder=config.get('freeze_question_encoder', False),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            max_path_length=config.get('max_path_length', 25),
            dropout=config.get('dropout', 0.1),
            learning_rate=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
            warmup_steps=config.get('warmup_steps', 1000),
            max_steps=config.get('max_steps', 100000),
            use_multipath_training=config.get('augment_paths', True),
            decoding_strategy=config.get('decoding_strategy', 'beam_search'),
            beam_width=config.get('beam_width', 5),
            top_p=config.get('top_p', 0.9),
            top_k=config.get('top_k', 50),
            temperature=config.get('temperature', 1.0),
            length_penalty=config.get('length_penalty', 0.6),
            use_contrastive_search=config.get('use_contrastive_search', False),
            contrastive_penalty_alpha=config.get('contrastive_penalty_alpha', 0.6),
            contrastive_top_k=config.get('contrastive_top_k', 4)
        )
    
    elif model_type == "gnn_decoder":
        from .gnn_decoder import KGPathGNNDecoderLightning
        
        return KGPathGNNDecoderLightning(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=config.get('hidden_dim', 256),
            question_encoder_name=config.get('question_encoder', 'sentence-transformers/all-MiniLM-L6-v2'),
            freeze_question_encoder=config.get('freeze_question_encoder', False),
            gnn_type=config.get('gnn_type', 'gat'),
            gnn_layers=config.get('gnn_layers', 3),
            gnn_heads=config.get('gnn_heads', 8),
            decoder_layers=config.get('decoder_layers', 6),
            decoder_heads=config.get('decoder_heads', 8),
            max_path_length=config.get('max_path_length', 25),
            dropout=config.get('dropout', 0.1),
            use_graph_structure=config.get('use_graph_structure', True),
            learning_rate=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
            warmup_steps=config.get('warmup_steps', 1000),
            max_steps=config.get('max_steps', 100000),
            use_multipath_training=config.get('augment_paths', True)
        )
    
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported types: 'diffusion', 'autoregressive', 'gnn_decoder'"
        )

