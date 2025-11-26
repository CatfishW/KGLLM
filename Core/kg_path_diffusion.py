"""
KG Path Diffusion Model - Main Model Implementation.

Combines:
- Question encoder (pretrained transformer)
- Graph encoder (RGCN/Transformer)
- Discrete diffusion for path generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from typing import Dict, Optional, Tuple, Any
from torch_geometric.data import Batch

from modules.graph_encoder import HybridGraphEncoder, RelationalGraphEncoder
from modules.diffusion import PathDiffusionTransformer, DiscreteDiffusion


class QuestionEncoder(nn.Module):
    """Encode questions using pretrained transformer."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 256,
        freeze: bool = False
    ):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.proj = nn.Linear(self.hidden_size, output_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode question.
        
        Returns:
            sequence_output: [B, seq_len, output_dim]
            pooled_output: [B, output_dim]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.proj(outputs.last_hidden_state)
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        return sequence_output, pooled_output


class KGPathDiffusionModel(nn.Module):
    """
    Full model for KG Path Generation using Discrete Diffusion.
    
    Architecture:
    1. Question Encoder: Pretrained transformer for question understanding
    2. Graph Encoder: RGCN/Transformer for KG subgraph encoding
    3. Path Diffusion: Discrete diffusion model for path generation
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 256,
        question_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_question_encoder: bool = False,
        graph_encoder_type: str = "hybrid",  # "rgcn", "transformer", "hybrid"
        num_graph_layers: int = 3,
        num_diffusion_layers: int = 6,
        num_heads: int = 8,
        num_diffusion_steps: int = 1000,
        max_path_length: int = 20,
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
        
        # Graph encoder
        if graph_encoder_type == "hybrid":
            self.graph_encoder = HybridGraphEncoder(
                num_entities=num_entities,
                num_relations=num_relations,
                entity_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_rgcn_layers=num_graph_layers // 2,
                num_transformer_layers=num_graph_layers - num_graph_layers // 2,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.graph_encoder = RelationalGraphEncoder(
                num_entities=num_entities,
                num_relations=num_relations,
                entity_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_graph_layers,
                dropout=dropout
            )
        
        # Path diffusion model
        self.diffusion = DiscreteDiffusion(
            num_entities=num_entities,
            num_relations=num_relations,
            num_timesteps=num_diffusion_steps
        )
        
        self.denoiser = PathDiffusionTransformer(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            num_layers=num_diffusion_layers,
            num_heads=num_heads,
            question_dim=hidden_dim,
            graph_dim=hidden_dim,
            max_path_length=max_path_length,
            dropout=dropout
        )
    
    def encode_inputs(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        graph_batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode question and graph inputs.
        
        Returns:
            question_seq: [B, seq_len, hidden_dim]
            question_pooled: [B, hidden_dim]
            graph_node_emb: [B, max_nodes, hidden_dim]
            graph_pooled: [B, hidden_dim]
        """
        # Encode question
        question_seq, question_pooled = self.question_encoder(
            question_input_ids, question_attention_mask
        )
        
        # Encode graph
        node_emb, graph_pooled = self.graph_encoder(
            graph_batch.node_ids,
            graph_batch.edge_index,
            graph_batch.edge_type,
            graph_batch.batch
        )
        
        # Reshape node embeddings to [B, max_nodes, hidden_dim]
        batch_size = question_input_ids.shape[0]
        device = question_input_ids.device
        
        # Get number of nodes per graph
        nodes_per_graph = torch.bincount(graph_batch.batch, minlength=batch_size)
        max_nodes = nodes_per_graph.max().item()
        
        # Create padded tensor
        graph_node_emb = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=device)
        
        # Fill in node embeddings
        node_idx = 0
        for i in range(batch_size):
            n_nodes = nodes_per_graph[i].item()
            graph_node_emb[i, :n_nodes] = node_emb[node_idx:node_idx + n_nodes]
            node_idx += n_nodes
        
        return question_seq, question_pooled, graph_node_emb, graph_pooled
    
    def forward(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        graph_batch: Batch,
        target_entities: torch.Tensor,
        target_relations: torch.Tensor,
        path_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            question_input_ids: [B, seq_len]
            question_attention_mask: [B, seq_len]
            graph_batch: PyG Batch object
            target_entities: [B, path_len]
            target_relations: [B, path_len-1]
            path_mask: [B, path_len] boolean mask (True for valid positions)
        
        Returns:
            Dictionary with loss and metrics
        """
        B = question_input_ids.shape[0]
        device = question_input_ids.device
        
        # Encode inputs
        question_seq, question_pooled, graph_node_emb, graph_pooled = self.encode_inputs(
            question_input_ids, question_attention_mask, graph_batch
        )
        
        # Sample random timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        
        # Add noise to target paths
        noisy_entities, noisy_relations = self.diffusion.q_sample(
            target_entities, target_relations, t
        )
        
        # Predict clean paths
        entity_logits, relation_logits = self.denoiser(
            noisy_entities, noisy_relations, t,
            question_seq, graph_node_emb,
            path_mask=path_mask,
            question_mask=~question_attention_mask.bool()
        )
        
        # Compute loss
        losses = self.diffusion.compute_loss(
            entity_logits, relation_logits,
            target_entities, target_relations,
            path_mask=path_mask
        )
        
        return losses
    
    @torch.no_grad()
    def generate(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        graph_batch: Batch,
        path_length: int = 10,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate reasoning paths.
        
        Returns:
            entities: [B, path_length]
            relations: [B, path_length-1]
        """
        # Encode inputs
        question_seq, question_pooled, graph_node_emb, graph_pooled = self.encode_inputs(
            question_input_ids, question_attention_mask, graph_batch
        )
        
        # Generate via diffusion sampling
        entities, relations = self.diffusion.sample(
            self.denoiser,
            question_seq, graph_node_emb,
            path_length=path_length,
            question_mask=~question_attention_mask.bool(),
            temperature=temperature
        )
        
        return entities, relations


class KGPathDiffusionLightning(pl.LightningModule):
    """
    PyTorch Lightning module for training KG Path Diffusion.
    
    Features:
    - Automatic mixed precision (AMP)
    - Gradient checkpointing
    - Multi-GPU training with DDP
    - Learning rate scheduling
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 256,
        question_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_question_encoder: bool = False,
        graph_encoder_type: str = "hybrid",
        num_graph_layers: int = 3,
        num_diffusion_layers: int = 6,
        num_heads: int = 8,
        num_diffusion_steps: int = 1000,
        max_path_length: int = 20,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = KGPathDiffusionModel(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            question_encoder_name=question_encoder_name,
            freeze_question_encoder=freeze_question_encoder,
            graph_encoder_type=graph_encoder_type,
            num_graph_layers=num_graph_layers,
            num_diffusion_layers=num_diffusion_layers,
            num_heads=num_heads,
            num_diffusion_steps=num_diffusion_steps,
            max_path_length=max_path_length,
            dropout=dropout
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.model(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            graph_batch=batch['graph_batch'],
            target_entities=batch['path_entities'],
            target_relations=batch['path_relations'],
            path_mask=batch.get('path_mask')
        )
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        
        self.log('train/loss', outputs['loss'], prog_bar=True, sync_dist=True)
        self.log('train/entity_loss', outputs['entity_loss'], sync_dist=True)
        self.log('train/relation_loss', outputs['relation_loss'], sync_dist=True)
        
        return outputs['loss']
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        
        self.log('val/loss', outputs['loss'], prog_bar=True, sync_dist=True)
        self.log('val/entity_loss', outputs['entity_loss'], sync_dist=True)
        self.log('val/relation_loss', outputs['relation_loss'], sync_dist=True)
        
        return outputs['loss']
    
    def configure_optimizers(self):
        # Separate parameters for different learning rates
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
        
        # Linear warmup then cosine decay
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


# Import math for lr_lambda
import math

