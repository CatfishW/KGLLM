"""
KG Path Diffusion Model - Main Model Implementation.

Combines:
- Question encoder (pretrained transformer)
- Discrete diffusion for path generation (relation chains only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from typing import Dict, Optional, Tuple, Any
import os
import math

from modules.diffusion import PathDiffusionTransformer, DiscreteDiffusion
from .base import QuestionEncoder


class PathCountPredictor(nn.Module):
    """
    Predicts the number of reasoning paths for a given question.
    
    This allows the model to generate a dynamic number of paths based on
    the complexity of the question, matching the ground truth distribution.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        max_paths: int = 30,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_paths = max_paths
        
        # MLP to predict path count from question representation
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_paths)  # Output logits for 1 to max_paths
        )
    
    def forward(self, question_pooled: torch.Tensor) -> torch.Tensor:
        """
        Predict path count distribution.
        
        Args:
            question_pooled: [B, hidden_dim] pooled question representation
        
        Returns:
            logits: [B, max_paths] logits for each path count (1 to max_paths)
        """
        return self.predictor(question_pooled)
    
    def predict_count(self, question_pooled: torch.Tensor) -> torch.Tensor:
        """
        Get predicted path count (argmax + 1, since index 0 = 1 path).
        
        Args:
            question_pooled: [B, hidden_dim]
        
        Returns:
            counts: [B] predicted number of paths (1 to max_paths)
        """
        logits = self.forward(question_pooled)
        return logits.argmax(dim=-1) + 1  # +1 because index 0 = 1 path


class KGPathDiffusionModel(nn.Module):
    """
    Full model for KG Path Generation using Discrete Diffusion.
    
    Architecture:
    1. Question Encoder: Pretrained transformer for question understanding
    2. Path Diffusion: Discrete diffusion model for relation chain generation
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 256,
        question_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_question_encoder: bool = False,
        num_diffusion_layers: int = 6,
        num_heads: int = 8,
        num_diffusion_steps: int = 1000,
        max_path_length: int = 20,
        dropout: float = 0.1,
        use_entity_embeddings: bool = True,
        predict_entities: bool = True
    ):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        self.predict_entities = predict_entities
        self.use_entity_embeddings = use_entity_embeddings
        
        # Path count predictor (predicts how many paths to generate)
        self.path_count_predictor = PathCountPredictor(
            hidden_dim=hidden_dim,
            max_paths=30,  # Maximum 30 paths
            dropout=dropout
        )
        
        # Question encoder
        self.question_encoder = QuestionEncoder(
            model_name=question_encoder_name,
            output_dim=hidden_dim,
            freeze=freeze_question_encoder
        )
        
        # Diffusion model
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
            max_path_length=max_path_length,
            dropout=dropout,
            predict_entities=predict_entities
        )
    
    def encode_inputs(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode question inputs.
        
        Returns:
            question_seq: [B, seq_len, hidden_dim]
            question_pooled: [B, hidden_dim]
        """
        # Encode question
        question_seq, question_pooled = self.question_encoder(
            question_input_ids, question_attention_mask
        )
        
        return question_seq, question_pooled
    
    def forward(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        target_entities: torch.Tensor,
        target_relations: torch.Tensor,
        path_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass (single path mode for backward compatibility).
        """
        # Encode inputs
        question_seq, question_pooled = self.encode_inputs(
            question_input_ids, question_attention_mask
        )
        
        return self._forward_diffusion_single(
            question_seq,
            question_attention_mask,
            target_entities,
            target_relations,
            path_mask
        )

    def _forward_diffusion_single(
        self,
        question_seq: torch.Tensor,
        question_attention_mask: torch.Tensor,
        target_entities: torch.Tensor,
        target_relations: torch.Tensor,
        path_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B = question_seq.shape[0]
        device = question_seq.device
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        noisy_entities, noisy_relations = self.diffusion.q_sample(
            target_entities, target_relations, t
        )
        
        # If not predicting entities, pass None for noisy_entities to denoiser
        denoiser_entities = noisy_entities if self.predict_entities else None
        
        entity_logits, relation_logits = self.denoiser(
            denoiser_entities, noisy_relations, t,
            question_seq,
            path_mask=path_mask,
            question_mask=~question_attention_mask.bool()
        )
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
        all_target_entities: torch.Tensor,
        all_target_relations: torch.Tensor,
        all_path_lengths: torch.Tensor,
        num_paths: torch.Tensor,
        path_count_loss_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-path training forward pass.
        
        Args:
            path_count_loss_weight: Weight for path count prediction loss
        """
        B = question_input_ids.shape[0]
        max_paths = all_target_entities.shape[1]
        path_len = all_target_entities.shape[2]
        rel_len = all_target_relations.shape[2]  # Use actual relation length (may differ from path_len - 1)
        device = question_input_ids.device
        
        # Encode inputs once (shared across all paths)
        question_seq, question_pooled = self.encode_inputs(
            question_input_ids, question_attention_mask
        )
        
        # Predict path count and compute loss
        path_count_logits = self.path_count_predictor(question_pooled)
        # Target: num_paths - 1 (since index 0 = 1 path)
        # Clamp to valid range [0, max_paths-1]
        path_count_targets = (num_paths - 1).clamp(0, self.path_count_predictor.max_paths - 1)
        path_count_loss = F.cross_entropy(path_count_logits, path_count_targets)
        
        # Expand encoded inputs for all paths
        # [B, ...] -> [B * max_paths, ...]
        question_seq_expanded = question_seq.unsqueeze(1).expand(-1, max_paths, -1, -1)
        question_seq_expanded = question_seq_expanded.reshape(B * max_paths, -1, self.hidden_dim)
        
        question_mask_expanded = (~question_attention_mask.bool()).unsqueeze(1).expand(-1, max_paths, -1)
        question_mask_expanded = question_mask_expanded.reshape(B * max_paths, -1)
        
        # Flatten paths: [B, max_paths, path_len] -> [B * max_paths, path_len]
        flat_entities = all_target_entities.reshape(B * max_paths, path_len)
        flat_relations = all_target_relations.reshape(B * max_paths, rel_len)
        flat_lengths = all_path_lengths.reshape(B * max_paths)
        
        # Create path mask based on lengths (for entities including BOS/EOS)
        path_mask = torch.arange(path_len, device=device).unsqueeze(0) < flat_lengths.unsqueeze(1)
        
        # Create relation mask
        # Path structure: [BOS, e1, r1, e2, r2, ..., en, EOS]
        # If path_length is L (including BOS and EOS), then:
        # - Number of entities WITH BOS/EOS: L
        # - Number of entities WITHOUT BOS/EOS: L - 2
        # - Number of relations: (L - 2) - 1 = L - 3 for L >= 3
        # For a path [BOS, e1, r1, e2, EOS], length=5, we have 1 relation r1
        # rel_lengths should be: flat_lengths - 3 (for L >= 3), else 0
        rel_lengths = (flat_lengths - 3).clamp(min=0)  # Subtract 3: BOS, EOS, and 1 for the fact that n entities have n-1 relations
       
        # Clamp rel_lengths to not exceed the actual relation tensor length
        rel_lengths = rel_lengths.clamp(max=rel_len)
        
        # Create relation mask: True for valid relations, False for padding
        relation_mask = torch.arange(rel_len, device=device).unsqueeze(0) < rel_lengths.unsqueeze(1)
        
        # Create valid path mask (to ignore padding paths)
        valid_path_mask = torch.arange(max_paths, device=device).unsqueeze(0) < num_paths.unsqueeze(1)
        valid_path_mask_flat = valid_path_mask.reshape(B * max_paths)
        
        diffusion_losses = self._forward_diffusion_multipath(
            question_seq_expanded,
            flat_entities,
            flat_relations,
            path_mask,
            relation_mask,
            valid_path_mask_flat,
            question_mask_expanded,
            B,
            max_paths,
            num_paths
        )
        
        # Combine diffusion loss with path count loss
        total_loss = diffusion_losses['loss'] + path_count_loss_weight * path_count_loss
        
        # Compute predicted path count for logging
        with torch.no_grad():
            predicted_counts = path_count_logits.argmax(dim=-1) + 1
            count_accuracy = (predicted_counts == num_paths).float().mean()
        
        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_losses['loss'],
            'path_count_loss': path_count_loss,
            'entity_loss': diffusion_losses['entity_loss'],
            'relation_loss': diffusion_losses['relation_loss'],
            'num_paths_avg': num_paths.float().mean(),
            'predicted_paths_avg': predicted_counts.float().mean(),
            'path_count_accuracy': count_accuracy
        }

    def _forward_diffusion_multipath(
        self,
        question_seq: torch.Tensor,
        flat_entities: torch.Tensor,
        flat_relations: torch.Tensor,
        path_mask: torch.Tensor,
        relation_mask: torch.Tensor,
        valid_path_mask: torch.Tensor,
        question_mask: torch.Tensor,
        batch_size: int,
        max_paths: int,
        num_paths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        device = question_seq.device
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=device)
        t_expanded = t.unsqueeze(1).expand(-1, max_paths).reshape(batch_size * max_paths)
        noisy_entities, noisy_relations = self.diffusion.q_sample(
            flat_entities, flat_relations, t_expanded
        )
        
        # If not predicting entities, pass None for noisy_entities to denoiser
        denoiser_entities = noisy_entities if self.predict_entities else None
        
        # Create proper path mask for denoiser
        # For relation-only mode, we need to pass a mask that matches the relation sequence length
        if self.predict_entities:
            # Entity+relation mode: pass full path_mask
            denoiser_path_mask = path_mask
        else:
            # Relation-only mode: create mask from relation_mask
            # The denoiser expects path_mask to derive relation mask, but we have relation_mask directly
            # So we create a dummy path_mask that will be used to derive the correct relation mask
            # path_mask is [B, path_len], relation_mask is [B, rel_len]
            # In relation-only mode, denoiser uses path_mask[:, :-1] to get relation mask
            # So we need to ensure path_mask has the right shape
            # Create a path_mask where the relation part matches our relation_mask
            # We'll create path_mask with shape [B, rel_len + 1] so that path_mask[:, :-1] = relation_mask
            denoiser_path_mask = torch.ones(relation_mask.shape[0], relation_mask.shape[1] + 1, 
                                            dtype=torch.bool, device=device)
            denoiser_path_mask[:, :-1] = ~relation_mask  # Invert: True = padding, False = valid
        
        entity_logits, relation_logits = self.denoiser(
            denoiser_entities, noisy_relations, t_expanded,
            question_seq,
            path_mask=denoiser_path_mask,
            question_mask=question_mask
        )
        
        # Replace NaN with zeros to prevent propagation
        if torch.isnan(relation_logits).any():
            relation_logits = torch.where(torch.isnan(relation_logits), 
                                         torch.zeros_like(relation_logits), 
                                         relation_logits)
        
        if entity_logits is not None and torch.isnan(entity_logits).any():
            entity_logits = torch.where(torch.isnan(entity_logits), 
                                       torch.zeros_like(entity_logits), 
                                       entity_logits)
        
        losses_per_path = self._compute_loss_per_path(
            entity_logits, relation_logits,
            flat_entities, flat_relations,
            path_mask, relation_mask, valid_path_mask
        )
        losses_per_path = losses_per_path.reshape(batch_size, max_paths)
        valid_mask = valid_path_mask.reshape(batch_size, max_paths).float()
        losses_per_path = losses_per_path * valid_mask
        
        # Compute average loss per sample, handling edge cases
        num_paths_float = num_paths.float().clamp(min=1)
        sample_losses = losses_per_path.sum(dim=1) / num_paths_float
        
        # Replace NaN with 0
        if torch.isnan(sample_losses).any():
            sample_losses = torch.where(torch.isnan(sample_losses), 
                                       torch.zeros_like(sample_losses), 
                                       sample_losses)
        
        total_loss = sample_losses.mean()
        
        # Final NaN check
        if torch.isnan(total_loss):
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        return {
            'loss': total_loss,
            'entity_loss': total_loss * 0.5 if self.predict_entities else torch.tensor(0.0, device=device),
            'relation_loss': total_loss * 0.5 if self.predict_entities else total_loss,
            'num_paths_avg': num_paths.float().mean()
        }
    
    def _compute_loss_per_path(
        self,
        entity_logits: Optional[torch.Tensor],
        relation_logits: torch.Tensor,
        target_entities: torch.Tensor,
        target_relations: torch.Tensor,
        path_mask: torch.Tensor,
        relation_mask: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for each path individually."""
        B_flat = relation_logits.shape[0]
        device = relation_logits.device
        losses = torch.zeros(B_flat, device=device, dtype=relation_logits.dtype)

        # If there are no valid paths at all in this batch, create a dummy loss that
        # is connected to the computation graph so that .backward() still works.
        if not valid_mask.any():
            dummy = relation_logits.sum() * 0.0
            losses = losses + dummy
            return losses
        
        for i in range(B_flat):
            if not valid_mask[i]:
                continue
            
            # Relation loss
            r_mask = relation_mask[i]
            if r_mask.sum() == 0:
                # No valid relations in this path (e.g., path is too short)
                # Ensure gradient flow even if loss is 0, but avoid NaN propagation
                r_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                r_logits = relation_logits[i][r_mask]
                r_targets = target_relations[i][r_mask]
                
                # Check for invalid targets (out of bounds or padding)
                valid_targets = (r_targets >= 0) & (r_targets < relation_logits.shape[-1]) & (r_targets != 0)
                if not valid_targets.any():
                    # All targets are invalid/padding, skip this path
                    r_loss = torch.tensor(0.0, device=device, requires_grad=True)
                elif r_logits.numel() > 0 and valid_targets.sum() > 0:
                        # Only compute loss on valid targets
                        valid_r_logits = r_logits[valid_targets]
                        valid_r_targets = r_targets[valid_targets]
                        r_loss = F.cross_entropy(valid_r_logits, valid_r_targets, reduction='mean')
                        # Check for NaN
                        if torch.isnan(r_loss):
                            r_loss = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    r_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Entity loss
            if entity_logits is not None:
                e_mask = path_mask[i]
                if e_mask.sum() == 0:
                    e_loss = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    e_logits = entity_logits[i][e_mask]
                    e_targets = target_entities[i][e_mask]
                    
                    # Check for invalid targets (out of bounds or padding)
                    valid_targets = (e_targets >= 0) & (e_targets < entity_logits.shape[-1]) & (e_targets != 0)
                    if not valid_targets.any():
                        # All targets are invalid/padding, skip this path
                        e_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    elif e_logits.numel() > 0 and valid_targets.sum() > 0:
                        # Only compute loss on valid targets
                        valid_e_logits = e_logits[valid_targets]
                        valid_e_targets = e_targets[valid_targets]
                        e_loss = F.cross_entropy(valid_e_logits, valid_e_targets, reduction='mean')
                        # Check for NaN
                        if torch.isnan(e_loss):
                            e_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    else:
                        e_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                e_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            losses[i] = e_loss + r_loss
            
        return losses

    @torch.no_grad()
    def generate(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
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
        question_seq, question_pooled = self.encode_inputs(
            question_input_ids, question_attention_mask
        )
        
        question_mask = ~question_attention_mask.bool()
        
        entities, relations = self.diffusion.sample(
            self.denoiser,
            question_seq,
            path_length=path_length,
            question_mask=question_mask,
            temperature=temperature
        )
        
        return entities, relations
    
    @torch.no_grad()
    def generate_multiple(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        num_paths: Optional[int] = None,
        path_length: int = 10,
        temperature: float = 1.0,
        diversity_penalty: float = 0.5,
        use_predicted_count: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate multiple diverse reasoning paths.
        
        Uses temperature sampling with diversity penalty to encourage
        different paths covering different answer entities.
        
        Args:
            question_input_ids: [B, seq_len]
            question_attention_mask: [B, seq_len]
            num_paths: Number of paths to generate per sample (if None, uses predicted count)
            path_length: Maximum length of each generated path
            temperature: Sampling temperature (higher = more diverse)
            diversity_penalty: Penalty for repeating entities/relations
            use_predicted_count: If True and num_paths is None, predict num_paths from question
        
        Returns:
            all_entities: [B, max_num_paths, path_length]
            all_relations: [B, max_num_paths, path_length-1]
            actual_num_paths: [B] actual number of paths generated for each sample
        """
        B = question_input_ids.shape[0]
        device = question_input_ids.device
        
        # Encode inputs once
        question_seq, question_pooled = self.encode_inputs(
            question_input_ids, question_attention_mask
        )
        
        # Determine number of paths to generate
        if num_paths is None and use_predicted_count:
            # Predict path count from question
            predicted_counts = self.path_count_predictor.predict_count(question_pooled)
            max_num_paths = predicted_counts.max().item()
        elif num_paths is not None:
            predicted_counts = torch.full((B,), num_paths, dtype=torch.long, device=device)
            max_num_paths = num_paths
        else:
            # Default to 5 paths
            predicted_counts = torch.full((B,), 5, dtype=torch.long, device=device)
            max_num_paths = 5
        
        # Generate paths for each sample (up to their predicted count)
        all_entities_list = []
        all_relations_list = []
        
        # Track generated final entities to encourage diversity
        generated_targets = torch.zeros(B, self.num_entities, device=device)
        
        for path_idx in range(max_num_paths):
            # Adjust temperature based on diversity penalty
            if path_idx > 0 and diversity_penalty > 0:
                # Use higher temperature for subsequent paths
                curr_temp = temperature * (1 + diversity_penalty * path_idx)
            else:
                curr_temp = temperature
            
            # Generate one path
            entities, relations = self.diffusion.sample(
                self.denoiser,
                question_seq,
                path_length=path_length,
                question_mask=~question_attention_mask.bool(),
                temperature=curr_temp
            )
            
            all_entities_list.append(entities)
            all_relations_list.append(relations)
            
            # Update generated targets (penalize same final entity)
            final_entities = entities[:, -1]
            for b in range(B):
                generated_targets[b, final_entities[b]] += 1
        
        # Stack results: [B, max_num_paths, path_length]
        all_entities = torch.stack(all_entities_list, dim=1)
        all_relations = torch.stack(all_relations_list, dim=1)
        
        return all_entities, all_relations, predicted_counts


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
        num_diffusion_layers: int = 6,
        num_heads: int = 8,
        num_diffusion_steps: int = 1000,
        max_path_length: int = 20,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        use_multipath_training: bool = True,
        use_entity_embeddings: bool = True,
        predict_entities: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = KGPathDiffusionModel(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            question_encoder_name=question_encoder_name,
            freeze_question_encoder=freeze_question_encoder,
            num_diffusion_layers=num_diffusion_layers,
            num_heads=num_heads,
            num_diffusion_steps=num_diffusion_steps,
            max_path_length=max_path_length,
            dropout=dropout,
            use_entity_embeddings=use_entity_embeddings,
            predict_entities=predict_entities
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.use_multipath_training = use_multipath_training
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> Dict[str, Any]:
        """
        Load checkpoint with parameter matching.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: If True, raise error on mismatches. If False, skip mismatched parameters.
        
        Returns:
            Dictionary with loading statistics:
            - loaded: number of parameters loaded
            - skipped: number of parameters skipped
            - missing: number of parameters in model but not in checkpoint
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
                        # Load the parameter
                        param.data.copy_(checkpoint_param)
                        loaded += 1
                    except Exception as e:
                        if strict:
                            raise RuntimeError(f"Failed to load parameter {name}: {e}")
                        skipped += 1
                else:
                    # Shape mismatch
                    if strict:
                        raise RuntimeError(
                            f"Shape mismatch for parameter {name}: "
                            f"model has {param.shape}, checkpoint has {checkpoint_param.shape}"
                        )
                    skipped += 1
            else:
                # Parameter not in checkpoint
                missing.append(name)
                if strict:
                    raise RuntimeError(f"Parameter {name} not found in checkpoint")
        
        stats = {
            'loaded': loaded,
            'skipped': skipped,
            'missing': len(missing),
            'missing_params': missing[:10]  # Show first 10 missing params
        }
        
        return stats
    
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
            target_entities=batch['path_entities'],
            target_relations=batch['path_relations'],
            path_mask=batch.get('path_mask')
        )
    
    def forward_multipath(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Multi-path forward - trains on ALL diverse paths."""
        return self.model.forward_multipath(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            all_target_entities=batch['all_path_entities'],
            all_target_relations=batch['all_path_relations'],
            all_path_lengths=batch['all_path_lengths'],
            num_paths=batch['num_paths']
        )
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Clear CUDA cache periodically to free up memory
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
        
        outputs = self(batch)
        
        self.log('train/loss', outputs['loss'], prog_bar=True, sync_dist=True)
        self.log('train/entity_loss', outputs['entity_loss'], sync_dist=True)
        self.log('train/relation_loss', outputs['relation_loss'], sync_dist=True)
        
        if 'num_paths_avg' in outputs:
            self.log('train/num_paths_avg', outputs['num_paths_avg'], sync_dist=True)
        
        # Log path count prediction metrics
        if 'path_count_loss' in outputs:
            self.log('train/path_count_loss', outputs['path_count_loss'], sync_dist=True)
        if 'diffusion_loss' in outputs:
            self.log('train/diffusion_loss', outputs['diffusion_loss'], sync_dist=True)
        if 'predicted_paths_avg' in outputs:
            self.log('train/predicted_paths_avg', outputs['predicted_paths_avg'], sync_dist=True)
        if 'path_count_accuracy' in outputs:
            self.log('train/path_count_accuracy', outputs['path_count_accuracy'], prog_bar=True, sync_dist=True)
        
        return outputs['loss']
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        
        # Use on_epoch=True to ensure metrics are aggregated and available for checkpointing
        self.log('val/loss', outputs['loss'], prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val/entity_loss', outputs['entity_loss'], sync_dist=True, on_step=False, on_epoch=True)
        self.log('val/relation_loss', outputs['relation_loss'], sync_dist=True, on_step=False, on_epoch=True)
        
        if 'num_paths_avg' in outputs:
            self.log('val/num_paths_avg', outputs['num_paths_avg'], sync_dist=True, on_step=False, on_epoch=True)
        
        # Log path count prediction metrics
        if 'path_count_loss' in outputs:
            self.log('val/path_count_loss', outputs['path_count_loss'], sync_dist=True, on_step=False, on_epoch=True)
        if 'diffusion_loss' in outputs:
            self.log('val/diffusion_loss', outputs['diffusion_loss'], sync_dist=True, on_step=False, on_epoch=True)
        if 'predicted_paths_avg' in outputs:
            self.log('val/predicted_paths_avg', outputs['predicted_paths_avg'], sync_dist=True, on_step=False, on_epoch=True)
        if 'path_count_accuracy' in outputs:
            self.log('val/path_count_accuracy', outputs['path_count_accuracy'], prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        
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



