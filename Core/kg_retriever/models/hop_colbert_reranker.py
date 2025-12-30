"""
HopColBERT: State-of-the-Art Reranking with Late Interaction + Hop Auxiliary Loss

Combines the best of 2025 retrieval research:
1. Late Interaction (ColBERT-style MaxSim): Fine-grained token-level matching
2. Hop-wise Auxiliary Loss: Multi-hop path supervision at each reasoning step
3. Hierarchical Path Representation: Position-aware hop encoding
4. Fast Inference: Pre-computable path embeddings, single-pass scoring

Reference architectures:
- ColBERTv2 (Santhanam et al., 2022) - Late interaction
- Contextual KGQA Rerankers (ACL 2024/2025) - Hop-aware scoring
- DCDR (Kuaishou) - Plackett-Luce listwise loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
import math

from .late_interaction import LateInteractionScorer, maxsim_batch
from .hop_auxiliary import HopAuxiliaryLoss


class HopColBERTReranker(pl.LightningModule):
    """
    HopColBERT: Late Interaction Reranker with Hop Auxiliary Loss
    
    Architecture:
    1. Shared encoder (BGE/ModernBERT) for question and paths
    2. Token projection layer for efficient MaxSim
    3. Hop position embeddings for multi-hop paths
    4. Late interaction scoring (MaxSim aggregation)
    5. Hop auxiliary heads for intermediate supervision
    
    Key Features:
    - Token-level late interaction for fine-grained matching
    - Hop-wise auxiliary losses for multi-hop supervision
    - Fast inference via pre-computed path embeddings
    - Plackett-Luce listwise loss for ranking
    """
    
    def __init__(
        self,
        encoder_name: str = "BAAI/bge-base-en-v1.5",
        hidden_dim: int = 768,
        projection_dim: int = 128,
        max_hops: int = 4,
        max_question_length: int = 128,
        max_path_length: int = 64,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_steps: int = 50000,
        freeze_encoder: bool = False,
        # Late interaction settings
        late_interaction_temp: float = 0.02,
        normalize_embeds: bool = True,
        # Hop auxiliary settings
        hop_aux_weight: float = 0.3,
        hop_loss_weights: Optional[List[float]] = None,
        hop_aux_cumulative: bool = True,
        # Primary loss settings
        primary_loss_temp: float = 1.0,
        margin_weight: float = 0.1,
        margin: float = 1.0,
    ):
        """
        Args:
            encoder_name: Pretrained encoder model name
            hidden_dim: Encoder hidden dimension
            projection_dim: Compressed dimension for MaxSim (lower = faster)
            max_hops: Maximum hops in paths
            max_question_length: Max question tokens
            max_path_length: Max path tokens
            dropout: Dropout rate
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Warmup ratio of total steps
            max_steps: Maximum training steps
            freeze_encoder: Whether to freeze the encoder
            late_interaction_temp: Temperature for MaxSim scoring
            normalize_embeds: L2 normalize before similarity
            hop_aux_weight: Weight for hop auxiliary loss
            hop_loss_weights: Per-hop loss weights
            hop_aux_cumulative: Use cumulative hop embeddings
            primary_loss_temp: Temperature for primary loss
            margin_weight: Weight for margin loss
            margin: Margin value for margin loss
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.max_hops = max_hops
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_steps = max_steps
        self.hop_aux_weight = hop_aux_weight
        self.margin_weight = margin_weight
        self.margin = margin
        self.primary_loss_temp = primary_loss_temp
        
        # === Encoder ===
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder_hidden_size = self.encoder.config.hidden_size
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # === Token Projection ===
        # Project to lower dimension for efficient MaxSim
        self.token_projection = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        
        # === Hop Position Embeddings ===
        self.hop_position_embed = nn.Embedding(max_hops, projection_dim)
        
        # === Pooled Projection (for auxiliary losses) ===
        self.pooled_projection = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # === Late Interaction Scorer ===
        self.late_interaction = LateInteractionScorer(
            hidden_dim=projection_dim,
            temperature=late_interaction_temp,
            learnable_temperature=False,
            normalize=normalize_embeds,
        )
        
        # === Hop Auxiliary Loss ===
        if hop_loss_weights is None:
            hop_loss_weights = [0.1, 0.2, 0.3, 0.4][:max_hops]
        
        self.hop_auxiliary = HopAuxiliaryLoss(
            hidden_dim=projection_dim,
            max_hops=max_hops,
            loss_weights=hop_loss_weights,
            temperature=1.0,
            cumulative=hop_aux_cumulative,
        )
        
        # === Final Scoring Head (optional, for ablation) ===
        # Can be disabled by setting use_final_scorer=False
        self.use_final_scorer = False  # Simplified: use MaxSim directly like PathRanker
        if self.use_final_scorer:
            self.final_scorer = nn.Sequential(
                nn.Linear(projection_dim * 2 + 1, projection_dim),  # +1 for MaxSim score
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, 1),
            )
        
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text and return both token-level and pooled representations.
        
        Returns:
            token_embeds: [B, T, projection_dim]
            pooled_embed: [B, projection_dim]
        """
        if self.hparams.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_states = outputs.last_hidden_state  # [B, T, hidden]
        
        # Token-level projection
        token_embeds = self.token_projection(hidden_states)  # [B, T, proj_dim]
        
        # Mean pooling for auxiliary losses
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled_embed = self.pooled_projection(pooled)  # [B, proj_dim]
        
        return token_embeds, pooled_embed
    
    def encode_paths_with_hops(
        self,
        path_input_ids: torch.Tensor,       # [B, C, T]
        path_attention_mask: torch.Tensor,  # [B, C, T]
        hop_boundaries: torch.Tensor,       # [B, C, max_hops, 2] - (start, end) for each hop
        hop_mask: torch.Tensor,             # [B, C, max_hops]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode paths and extract per-hop representations.
        
        Args:
            path_input_ids: Path tokens [B, C, T]
            path_attention_mask: Path mask [B, C, T]
            hop_boundaries: Start/end indices for each hop [B, C, max_hops, 2]
            hop_mask: Which hops are valid [B, C, max_hops]
            
        Returns:
            path_token_embeds: [B, C, T, D]
            path_pooled_embeds: [B, C, D]
            hop_embeds: [B, C, max_hops, D]
        """
        B, C, T = path_input_ids.shape
        device = path_input_ids.device
        
        # Flatten for batch encoding
        flat_input_ids = path_input_ids.view(B * C, T)
        flat_mask = path_attention_mask.view(B * C, T)
        
        # Encode all paths
        token_embeds, pooled_embeds = self.encode_text(flat_input_ids, flat_mask)
        
        # Reshape back
        token_embeds = token_embeds.view(B, C, T, -1)  # [B, C, T, D]
        pooled_embeds = pooled_embeds.view(B, C, -1)   # [B, C, D]
        
        # Extract per-hop embeddings
        D = token_embeds.shape[-1]
        hop_embeds = torch.zeros(B, C, self.max_hops, D, device=device)
        
        for h in range(self.max_hops):
            # Get start/end positions for this hop
            starts = hop_boundaries[:, :, h, 0]  # [B, C]
            ends = hop_boundaries[:, :, h, 1]    # [B, C]
            
            # For each sample and candidate, extract hop tokens and average
            for b in range(B):
                for c in range(C):
                    if hop_mask[b, c, h]:
                        s, e = starts[b, c].item(), ends[b, c].item()
                        if s < e and e <= T:
                            hop_tokens = token_embeds[b, c, s:e, :]  # [hop_len, D]
                            hop_embeds[b, c, h, :] = hop_tokens.mean(dim=0)
            
            # Add hop position embedding
            hop_pos = self.hop_position_embed(torch.tensor(h, device=device))
            hop_embeds[:, :, h, :] = hop_embeds[:, :, h, :] + hop_pos
        
        return token_embeds, pooled_embeds, hop_embeds
    
    def forward(
        self,
        question_input_ids: torch.Tensor,       # [B, Tq]
        question_attention_mask: torch.Tensor,  # [B, Tq]
        path_input_ids: torch.Tensor,           # [B, C, Tp]
        path_attention_mask: torch.Tensor,      # [B, C, Tp]
        candidate_mask: torch.Tensor,           # [B, C]
        hop_boundaries: Optional[torch.Tensor] = None,  # [B, C, max_hops, 2]
        hop_mask: Optional[torch.Tensor] = None,        # [B, C, max_hops]
        labels: Optional[torch.Tensor] = None,  # [B]
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with late interaction and hop auxiliary losses.
        
        Returns:
            logits: Final ranking scores [B, C]
            loss: Combined loss (if labels provided)
            Additional losses and metrics
        """
        B, Tq = question_input_ids.shape
        _, C, Tp = path_input_ids.shape
        device = question_input_ids.device
        
        # === Encode Question ===
        q_token_embeds, q_pooled = self.encode_text(
            question_input_ids, question_attention_mask
        )  # [B, Tq, D], [B, D]
        
        # === Encode Paths ===
        if hop_boundaries is not None and hop_mask is not None:
            p_token_embeds, p_pooled, hop_embeds = self.encode_paths_with_hops(
                path_input_ids, path_attention_mask, hop_boundaries, hop_mask
            )
        else:
            # Simple path encoding without hop extraction
            flat_path_ids = path_input_ids.view(B * C, Tp)
            flat_path_mask = path_attention_mask.view(B * C, Tp)
            p_tok_flat, p_pool_flat = self.encode_text(flat_path_ids, flat_path_mask)
            p_token_embeds = p_tok_flat.view(B, C, Tp, -1)
            p_pooled = p_pool_flat.view(B, C, -1)
            hop_embeds = None
            hop_mask = None
        
        # === Late Interaction Scores ===
        # MaxSim: for each query token, find max sim across path tokens
        maxsim_scores = self.late_interaction(
            q_token_embeds,
            p_token_embeds,
            question_attention_mask.bool(),
            path_attention_mask.bool(),
        )  # [B, C]
        
        # === Final Scoring ===
        # Simplified: Use MaxSim directly as logits (like PathRanker uses cosine sim)
        # This makes the model more comparable and stable during early training
        if self.use_final_scorer:
            # Original complex path: combine MaxSim with learned scoring
            q_pooled_expanded = q_pooled.unsqueeze(1).expand(-1, C, -1)  # [B, C, D]
            combined_features = torch.cat([
                q_pooled_expanded,
                p_pooled,
                maxsim_scores.unsqueeze(-1),
            ], dim=-1)  # [B, C, 2D+1]
            final_scores = self.final_scorer(combined_features).squeeze(-1)  # [B, C]
        else:
            # Simplified path: MaxSim scores are the final scores
            final_scores = maxsim_scores
        
        # Apply candidate mask
        logits = final_scores.masked_fill(~candidate_mask.bool(), -65504.0)
        
        result = {
            'logits': logits,
            'maxsim_scores': maxsim_scores,
            'final_scores': final_scores,
        }
        
        # === Compute Losses ===
        if labels is not None:
            # Primary loss: Plackett-Luce (cross-entropy)
            scaled_logits = logits / self.primary_loss_temp
            primary_loss = F.cross_entropy(scaled_logits, labels)
            result['primary_loss'] = primary_loss
            
            # Margin loss
            batch_idx = torch.arange(B, device=device)
            correct_scores = final_scores[batch_idx, labels]
            margin_loss = F.relu(self.margin - (correct_scores.unsqueeze(1) - final_scores))
            margin_loss = margin_loss.masked_fill(~candidate_mask.bool(), 0)
            margin_loss[batch_idx, labels] = 0
            margin_loss = margin_loss.sum(dim=1).mean()
            result['margin_loss'] = margin_loss
            
            # Hop auxiliary loss
            if hop_embeds is not None and hop_mask is not None:
                hop_aux_loss, hop_losses = self.hop_auxiliary(
                    hop_embeds, q_pooled, labels, candidate_mask, hop_mask
                )
                result['hop_aux_loss'] = hop_aux_loss
                result.update(hop_losses)
            else:
                hop_aux_loss = torch.tensor(0.0, device=device)
            
            # Combined loss
            total_loss = (
                primary_loss
                + self.margin_weight * margin_loss
                + self.hop_aux_weight * hop_aux_loss
            )
            result['loss'] = total_loss
        
        return result
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self(**batch)
        loss = outputs['loss']
        
        # Compute accuracy
        with torch.no_grad():
            preds = outputs['logits'].argmax(dim=-1)
            acc = (preds == batch['labels']).float().mean()
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/primary_loss', outputs['primary_loss'])
        self.log('train/margin_loss', outputs['margin_loss'])
        if 'hop_aux_loss' in outputs:
            self.log('train/hop_aux_loss', outputs['hop_aux_loss'])
        self.log('train/acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self(**batch)
        loss = outputs['loss']
        
        preds = outputs['logits'].argmax(dim=-1)
        acc = (preds == batch['labels']).float().mean()
        
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/acc', acc, prog_bar=True, sync_dist=True)
        
        # Log per-hop accuracies if available
        for h in range(self.max_hops):
            key = f'hop_{h+1}_acc'
            if key in outputs:
                self.log(f'val/{key}', outputs[key], sync_dist=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self(**batch)
        
        preds = outputs['logits'].argmax(dim=-1)
        acc = (preds == batch['labels']).float().mean()
        
        self.log('test/acc', acc, sync_dist=True)
        
        return {'preds': preds, 'labels': batch['labels'], 'scores': outputs['logits']}
    
    def configure_optimizers(self):
        # Use simple optimizer configuration like PathRanker
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Linear warmup then cosine decay
        warmup_steps = int(self.max_steps * self.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, self.max_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
    
    def rank_paths(
        self,
        question: str,
        candidate_paths: List[List[str]],
        tokenizer: AutoTokenizer,
        top_k: int = 10,
        batch_size: int = 32,
    ) -> List[Tuple[List[str], float]]:
        """
        Rank candidate paths for a question (inference).
        
        Args:
            question: Question text
            candidate_paths: List of paths (each path is list of relations)
            tokenizer: Tokenizer for encoding
            top_k: Number of top paths to return
            batch_size: Batch size for processing
            
        Returns:
            List of (path, score) tuples sorted by score descending
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Encode question
        q_enc = tokenizer(
            question,
            max_length=self.hparams.max_question_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        q_input_ids = q_enc['input_ids'].to(device)
        q_attention_mask = q_enc['attention_mask'].to(device)
        
        # Encode paths
        path_texts = [" -> ".join(path) for path in candidate_paths]
        p_enc = tokenizer(
            path_texts,
            max_length=self.hparams.max_path_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        p_input_ids = p_enc['input_ids'].to(device)
        p_attention_mask = p_enc['attention_mask'].to(device)
        
        C = len(candidate_paths)
        
        # Add batch dimension
        p_input_ids = p_input_ids.unsqueeze(0)  # [1, C, T]
        p_attention_mask = p_attention_mask.unsqueeze(0)
        candidate_mask = torch.ones(1, C, device=device)
        
        with torch.no_grad():
            outputs = self(
                question_input_ids=q_input_ids,
                question_attention_mask=q_attention_mask,
                path_input_ids=p_input_ids,
                path_attention_mask=p_attention_mask,
                candidate_mask=candidate_mask,
            )
            scores = outputs['logits'][0].cpu().tolist()
        
        # Sort by score
        path_scores = list(zip(candidate_paths, scores))
        path_scores.sort(key=lambda x: x[1], reverse=True)
        
        return path_scores[:top_k]


if __name__ == '__main__':
    print("Testing HopColBERTReranker...")
    
    model = HopColBERTReranker(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=384,
        projection_dim=128,
        max_hops=4,
        freeze_encoder=True,  # Faster for testing
    )
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Create dummy batch
    B, C, Tq, Tp = 2, 10, 32, 48
    max_hops = 4
    
    batch = {
        'question_input_ids': torch.randint(0, 1000, (B, Tq)),
        'question_attention_mask': torch.ones(B, Tq, dtype=torch.long),
        'path_input_ids': torch.randint(0, 1000, (B, C, Tp)),
        'path_attention_mask': torch.ones(B, C, Tp, dtype=torch.long),
        'candidate_mask': torch.ones(B, C),
        'labels': torch.randint(0, C, (B,)),
    }
    
    # Test without hop boundaries (simpler mode)
    print("\nTesting without hop boundaries...")
    outputs = model(**batch)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Test with hop boundaries
    print("\nTesting with hop boundaries...")
    hop_boundaries = torch.zeros(B, C, max_hops, 2, dtype=torch.long)
    hop_mask = torch.zeros(B, C, max_hops, dtype=torch.bool)
    
    # Create some dummy hop boundaries
    for b in range(B):
        for c in range(C):
            num_hops = min(3, max_hops)  # 3 hops per path
            step = Tp // (num_hops + 1)
            for h in range(num_hops):
                hop_boundaries[b, c, h, 0] = h * step
                hop_boundaries[b, c, h, 1] = (h + 1) * step
                hop_mask[b, c, h] = True
    
    batch['hop_boundaries'] = hop_boundaries
    batch['hop_mask'] = hop_mask
    
    outputs = model(**batch)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Primary loss: {outputs['primary_loss'].item():.4f}")
    print(f"Hop aux loss: {outputs['hop_aux_loss'].item():.4f}")
    
    print("\nHopColBERTReranker test passed!")
