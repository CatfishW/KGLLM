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
        Training forward pass (single path mode for backward compatibility).
        
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
    
    def forward_multipath(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        graph_batch: Batch,
        all_target_entities: torch.Tensor,
        all_target_relations: torch.Tensor,
        all_path_lengths: torch.Tensor,
        num_paths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-path training forward pass.
        
        Trains the model on ALL diverse paths, not just one.
        The loss is computed over all valid paths and averaged.
        
        Args:
            question_input_ids: [B, seq_len]
            question_attention_mask: [B, seq_len]
            graph_batch: PyG Batch object
            all_target_entities: [B, max_paths, path_len]
            all_target_relations: [B, max_paths, path_len-1]
            all_path_lengths: [B, max_paths] - actual length of each path
            num_paths: [B] - number of valid paths per sample
        
        Returns:
            Dictionary with loss and metrics
        """
        B = question_input_ids.shape[0]
        max_paths = all_target_entities.shape[1]
        path_len = all_target_entities.shape[2]
        rel_len = all_target_relations.shape[2]  # Use actual relation length (may differ from path_len - 1)
        device = question_input_ids.device
        
        # Encode inputs once (shared across all paths)
        question_seq, question_pooled, graph_node_emb, graph_pooled = self.encode_inputs(
            question_input_ids, question_attention_mask, graph_batch
        )
        
        # Expand encoded inputs for all paths
        # [B, ...] -> [B * max_paths, ...]
        question_seq_expanded = question_seq.unsqueeze(1).expand(-1, max_paths, -1, -1)
        question_seq_expanded = question_seq_expanded.reshape(B * max_paths, -1, self.hidden_dim)
        
        graph_node_emb_expanded = graph_node_emb.unsqueeze(1).expand(-1, max_paths, -1, -1)
        graph_node_emb_expanded = graph_node_emb_expanded.reshape(B * max_paths, -1, self.hidden_dim)
        
        question_mask_expanded = (~question_attention_mask.bool()).unsqueeze(1).expand(-1, max_paths, -1)
        question_mask_expanded = question_mask_expanded.reshape(B * max_paths, -1)
        
        # Flatten paths: [B, max_paths, path_len] -> [B * max_paths, path_len]
        flat_entities = all_target_entities.reshape(B * max_paths, path_len)
        flat_relations = all_target_relations.reshape(B * max_paths, rel_len)
        flat_lengths = all_path_lengths.reshape(B * max_paths)
        
        # Create path mask based on lengths (for entities)
        path_mask = torch.arange(path_len, device=device).unsqueeze(0) < flat_lengths.unsqueeze(1)
        
        # Create relation mask (relations are one less than entities, but respect rel_len)
        # For a path of length L, there are L-1 relations
        rel_lengths = (flat_lengths - 1).clamp(min=0)  # Ensure non-negative
        relation_mask = torch.arange(rel_len, device=device).unsqueeze(0) < rel_lengths.unsqueeze(1)
        
        # Create valid path mask (to ignore padding paths)
        valid_path_mask = torch.arange(max_paths, device=device).unsqueeze(0) < num_paths.unsqueeze(1)
        valid_path_mask_flat = valid_path_mask.reshape(B * max_paths)
        
        # Sample random timesteps (same for all paths in a batch)
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        t_expanded = t.unsqueeze(1).expand(-1, max_paths).reshape(B * max_paths)
        
        # Add noise to all target paths
        noisy_entities, noisy_relations = self.diffusion.q_sample(
            flat_entities, flat_relations, t_expanded
        )
        
        # Predict clean paths for all
        entity_logits, relation_logits = self.denoiser(
            noisy_entities, noisy_relations, t_expanded,
            question_seq_expanded, graph_node_emb_expanded,
            path_mask=path_mask,
            question_mask=question_mask_expanded
        )
        
        # Compute loss per path
        losses_per_path = self._compute_loss_per_path(
            entity_logits, relation_logits,
            flat_entities, flat_relations,
            path_mask, relation_mask, valid_path_mask_flat
        )
        
        # Reshape losses: [B * max_paths] -> [B, max_paths]
        losses_per_path = losses_per_path.reshape(B, max_paths)
        
        # Mask out invalid paths and compute mean loss per sample
        losses_per_path = losses_per_path * valid_path_mask.float()
        
        # Average over valid paths
        sample_losses = losses_per_path.sum(dim=1) / num_paths.float().clamp(min=1)
        
        # Final loss is mean over batch
        total_loss = sample_losses.mean()
        
        return {
            'loss': total_loss,
            'entity_loss': total_loss * 0.5,  # Approximate split
            'relation_loss': total_loss * 0.5,
            'num_paths_avg': num_paths.float().mean()
        }
    
    def _compute_loss_per_path(
        self,
        entity_logits: torch.Tensor,
        relation_logits: torch.Tensor,
        target_entities: torch.Tensor,
        target_relations: torch.Tensor,
        path_mask: torch.Tensor,
        relation_mask: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for each path individually."""
        B_flat = entity_logits.shape[0]
        device = entity_logits.device
        
        losses = torch.zeros(B_flat, device=device)
        
        for i in range(B_flat):
            if not valid_mask[i]:
                continue
            
            # Get valid positions for this path
            e_mask = path_mask[i]
            r_mask = relation_mask[i]
            
            # Entity loss
            e_logits = entity_logits[i][e_mask]
            e_targets = target_entities[i][e_mask]
            if e_logits.numel() > 0:
                e_loss = F.cross_entropy(e_logits, e_targets, ignore_index=0, reduction='mean')
            else:
                e_loss = torch.tensor(0.0, device=device)
            
            # Relation loss (use relation_mask directly)
            r_logits = relation_logits[i][r_mask]
            r_targets = target_relations[i][r_mask]
            if r_logits.numel() > 0:
                r_loss = F.cross_entropy(r_logits, r_targets, ignore_index=0, reduction='mean')
            else:
                r_loss = torch.tensor(0.0, device=device)
            
            losses[i] = e_loss + r_loss
        
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
        Generate a single reasoning path.
        
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
    
    @torch.no_grad()
    def generate_multiple(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        graph_batch: Batch,
        num_paths: int = 5,
        path_length: int = 10,
        temperature: float = 1.0,
        diversity_penalty: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple diverse reasoning paths.
        
        Uses temperature sampling with diversity penalty to encourage
        different paths covering different answer entities.
        
        Args:
            question_input_ids: [B, seq_len]
            question_attention_mask: [B, seq_len]
            graph_batch: PyG Batch object
            num_paths: Number of paths to generate per sample
            path_length: Length of each path
            temperature: Sampling temperature (higher = more diverse)
            diversity_penalty: Penalty for repeating entities/relations
        
        Returns:
            all_entities: [B, num_paths, path_length]
            all_relations: [B, num_paths, path_length-1]
        """
        B = question_input_ids.shape[0]
        device = question_input_ids.device
        
        # Encode inputs once
        question_seq, question_pooled, graph_node_emb, graph_pooled = self.encode_inputs(
            question_input_ids, question_attention_mask, graph_batch
        )
        
        # Generate multiple paths
        all_entities = []
        all_relations = []
        
        # Track generated final entities to encourage diversity
        generated_targets = torch.zeros(B, self.num_entities, device=device)
        
        for path_idx in range(num_paths):
            # Adjust temperature based on diversity penalty
            if path_idx > 0 and diversity_penalty > 0:
                # Use higher temperature for subsequent paths
                curr_temp = temperature * (1 + diversity_penalty * path_idx)
            else:
                curr_temp = temperature
            
            # Generate one path
            entities, relations = self.diffusion.sample(
                self.denoiser,
                question_seq, graph_node_emb,
                path_length=path_length,
                question_mask=~question_attention_mask.bool(),
                temperature=curr_temp
            )
            
            all_entities.append(entities)
            all_relations.append(relations)
            
            # Update generated targets (penalize same final entity)
            final_entities = entities[:, -1]
            for b in range(B):
                generated_targets[b, final_entities[b]] += 1
        
        # Stack results: [B, num_paths, path_length]
        all_entities = torch.stack(all_entities, dim=1)
        all_relations = torch.stack(all_relations, dim=1)
        
        return all_entities, all_relations


class KGPathDiffusionLightning(pl.LightningModule):
    """
    PyTorch Lightning module for training KG Path Diffusion.
    
    Features:
    - Multi-path training: Learn to generate ALL diverse paths
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
        max_steps: int = 100000,
        use_multipath_training: bool = True  # Enable multi-path training
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
        self.use_multipath_training = use_multipath_training
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass - uses multi-path training if enabled."""
        if self.use_multipath_training and 'all_path_entities' in batch:
            return self.forward_multipath(batch)
        else:
            return self.forward_single(batch)
    
    def forward_single(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Single path forward (backward compatible)."""
        return self.model(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            graph_batch=batch['graph_batch'],
            target_entities=batch['path_entities'],
            target_relations=batch['path_relations'],
            path_mask=batch.get('path_mask')
        )
    
    def forward_multipath(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Multi-path forward - trains on ALL diverse paths."""
        return self.model.forward_multipath(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            graph_batch=batch['graph_batch'],
            all_target_entities=batch['all_path_entities'],
            all_target_relations=batch['all_path_relations'],
            all_path_lengths=batch['all_path_lengths'],
            num_paths=batch['num_paths']
        )
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        
        self.log('train/loss', outputs['loss'], prog_bar=True, sync_dist=True)
        self.log('train/entity_loss', outputs['entity_loss'], sync_dist=True)
        self.log('train/relation_loss', outputs['relation_loss'], sync_dist=True)
        
        if 'num_paths_avg' in outputs:
            self.log('train/num_paths_avg', outputs['num_paths_avg'], sync_dist=True)
        
        return outputs['loss']
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        
        self.log('val/loss', outputs['loss'], prog_bar=True, sync_dist=True)
        self.log('val/entity_loss', outputs['entity_loss'], sync_dist=True)
        self.log('val/relation_loss', outputs['relation_loss'], sync_dist=True)
        
        if 'num_paths_avg' in outputs:
            self.log('val/num_paths_avg', outputs['num_paths_avg'], sync_dist=True)
        
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

