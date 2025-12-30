"""
Diffusion-based Path Scorer for Reranking Candidate Paths.

Key Idea: Score candidate paths by measuring how well they can be
reconstructed from noise given a question. Paths with lower reconstruction
error are more likely to be correct.

This is a discriminative approach using diffusion principles, not generative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, List, Any, Tuple
import math


class DiffusionPathScorer(pl.LightningModule):
    """
    Scores candidate paths using diffusion-based likelihood estimation.
    
    Architecture:
    1. Encode question with frozen text encoder (BGE)
    2. Encode each candidate path as text
    3. Use cross-attention to score question-path compatibility
    4. Add diffusion-based denoising score for regularization
    
    The key insight is that good paths should be "predictable" given the question,
    which we measure through reconstruction error at various noise levels.
    """
    
    def __init__(
        self,
        encoder_name: str = "BAAI/bge-base-en-v1.5",
        hidden_dim: int = 768,
        num_diffusion_steps: int = 10,  # Fewer steps for scoring
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 20000,
        temperature: float = 0.05,
        max_question_length: int = 128,
        max_path_length: int = 64,
        diffusion_weight: float = 0.3,  # Weight of diffusion score vs similarity
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.temperature = temperature
        self.diffusion_weight = diffusion_weight
        
        # Shared text encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder_hidden_size = self.encoder.config.hidden_size
        
        # Freeze encoder initially (can unfreeze later for fine-tuning)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Question-to-path cross-attention scorer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.encoder_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Diffusion noise predictor (lightweight)
        self.noise_predictor = nn.Sequential(
            nn.Linear(self.encoder_hidden_size * 2 + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.encoder_hidden_size),
        )
        
        # Timestep embedding for diffusion
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Final scoring head
        self.score_head = nn.Sequential(
            nn.Linear(self.encoder_hidden_size * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # Hop prediction head (auxiliary task)
        self.hop_head = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 5),  # 0-4 hops
        )
        
    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb)
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text and return both sequence and pooled representations.
        
        Returns:
            sequence_output: [B, seq_len, hidden]
            pooled_output: [B, hidden]
        """
        with torch.set_grad_enabled(self.training and self.encoder.training):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden = outputs.last_hidden_state
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden * mask_expanded).sum(dim=1)
        pooled = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        return hidden, pooled
    
    def compute_similarity_score(
        self,
        question_pooled: torch.Tensor,  # [B, hidden]
        path_pooled: torch.Tensor,       # [B, num_candidates, hidden]
    ) -> torch.Tensor:
        """Compute learned scoring using concatenation + MLP."""
        B, C, H = path_pooled.shape
        
        # Expand question to match candidates
        question_expanded = question_pooled.unsqueeze(1).expand(-1, C, -1)  # [B, C, H]
        
        # Concatenate and score
        combined = torch.cat([question_expanded, path_pooled], dim=-1)  # [B, C, 2*H]
        scores = self.score_head(combined).squeeze(-1)  # [B, C]
        
        return scores
    
    def compute_cosine_similarity(
        self,
        question_pooled: torch.Tensor,
        path_pooled: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity for auxiliary loss."""
        q_norm = F.normalize(question_pooled, dim=-1)
        p_norm = F.normalize(path_pooled, dim=-1)
        return torch.bmm(p_norm, q_norm.unsqueeze(-1)).squeeze(-1)
    
    def compute_diffusion_score(
        self,
        question_pooled: torch.Tensor,  # [B, hidden]
        path_pooled: torch.Tensor,       # [B, C, hidden]
    ) -> torch.Tensor:
        """
        Compute diffusion-based reconstruction score.
        Only used during inference (adds noise during training).
        """
        B, C, H = path_pooled.shape
        device = path_pooled.device
        
        # Use fixed timestep for deterministic scoring during eval
        if not self.training:
            timesteps = torch.full((B,), self.num_diffusion_steps // 2, device=device)
        else:
            timesteps = torch.randint(1, self.num_diffusion_steps + 1, (B,), device=device)
        
        time_emb = self.get_timestep_embedding(timesteps)
        noise_level = timesteps.float() / self.num_diffusion_steps
        noise_level = noise_level.view(B, 1, 1)
        
        noise = torch.randn_like(path_pooled)
        noisy_paths = path_pooled * (1 - noise_level) + noise * noise_level
        
        question_expanded = question_pooled.unsqueeze(1).expand(-1, C, -1)
        time_expanded = time_emb.unsqueeze(1).expand(-1, C, -1)
        
        predictor_input = torch.cat([noisy_paths, question_expanded, time_expanded], dim=-1)
        predicted_noise = self.noise_predictor(predictor_input)
        
        recon_error = F.mse_loss(predicted_noise, noise, reduction='none').mean(dim=-1)
        return -recon_error
    
    def forward(
        self,
        question_input_ids: torch.Tensor,       # [B, seq_len]
        question_attention_mask: torch.Tensor,  # [B, seq_len]
        path_input_ids: torch.Tensor,           # [B, num_candidates, seq_len]
        path_attention_mask: torch.Tensor,      # [B, num_candidates, seq_len]
        candidate_mask: torch.Tensor,           # [B, num_candidates]
        labels: Optional[torch.Tensor] = None,  # [B] - index of correct path
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for path scoring.
        
        Returns:
            logits: [B, num_candidates]
            loss: scalar (if labels provided)
        """
        B, C, path_seq_len = path_input_ids.shape
        
        # Encode question
        _, question_pooled = self.encode_text(
            question_input_ids, question_attention_mask
        )  # [B, hidden]
        
        # Encode all candidate paths (reshape for batch processing)
        path_input_flat = path_input_ids.view(B * C, path_seq_len)
        path_mask_flat = path_attention_mask.view(B * C, path_seq_len)
        
        _, path_pooled_flat = self.encode_text(path_input_flat, path_mask_flat)
        path_pooled = path_pooled_flat.view(B, C, -1)  # [B, C, hidden]
        
        # Compute learned similarity score (main signal)
        sim_scores = self.compute_similarity_score(question_pooled, path_pooled)
        
        # During training, ONLY use learned scores (no noisy diffusion)
        # Diffusion is used as auxiliary regularization or at inference
        if self.training:
            # Pure learned scoring during training for stable gradients
            combined_scores = sim_scores
        else:
            # At inference, optionally blend with diffusion
            if self.diffusion_weight > 0:
                diff_scores = self.compute_diffusion_score(question_pooled, path_pooled)
                # Normalize both to similar scale
                sim_scores_norm = (sim_scores - sim_scores.mean(dim=-1, keepdim=True)) / (sim_scores.std(dim=-1, keepdim=True) + 1e-6)
                diff_scores_norm = (diff_scores - diff_scores.mean(dim=-1, keepdim=True)) / (diff_scores.std(dim=-1, keepdim=True) + 1e-6)
                combined_scores = (1 - self.diffusion_weight) * sim_scores_norm + self.diffusion_weight * diff_scores_norm
            else:
                combined_scores = sim_scores
        
        # Apply candidate mask (use fp16-safe value: max fp16 is ~65504)
        combined_scores = combined_scores.masked_fill(~candidate_mask.bool(), -65000.0)
        
        # No temperature scaling (learned head already outputs proper scale)
        logits = combined_scores
        
        result = {'logits': logits, 'sim_scores': sim_scores}
        
        # Compute loss if labels provided
        if labels is not None:
            # Cross-entropy loss for ranking
            loss = F.cross_entropy(logits, labels)
            
            # Auxiliary: hop prediction loss
            if 'hop_labels' in kwargs:
                # Get pooled representation of the ground truth path
                # path_pooled shape: [B, C, hidden]
                # We need the one at index 'labels' for each batch item
                # Using advanced indexing
                batch_indices = torch.arange(B, device=path_pooled.device)
                gt_path_pooled = path_pooled[batch_indices, labels]  # [B, hidden]
                
                hop_logits = self.hop_head(gt_path_pooled)  # [B, 5]
                hop_loss = F.cross_entropy(hop_logits, kwargs['hop_labels'])
                
                # Add to total loss (weighted)
                loss = loss + 0.1 * hop_loss
                result['hop_loss'] = hop_loss
                
            result['loss'] = loss
        
        return result
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # Extract kwargs that might be present (like hop_labels)
        kwargs = {k: v for k, v in batch.items() if k not in [
            'question_input_ids', 'question_attention_mask', 
            'path_input_ids', 'path_attention_mask', 
            'candidate_mask', 'labels'
        ]}
        
        outputs = self.forward(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            path_input_ids=batch['path_input_ids'],
            path_attention_mask=batch['path_attention_mask'],
            candidate_mask=batch['candidate_mask'],
            labels=batch['labels'],
            **kwargs
        )
        
        loss = outputs['loss']
        
        # Compute accuracy
        with torch.no_grad():
            preds = outputs['logits'].argmax(dim=-1)
            acc = (preds == batch['labels']).float().mean()
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self.forward(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            path_input_ids=batch['path_input_ids'],
            path_attention_mask=batch['path_attention_mask'],
            candidate_mask=batch['candidate_mask'],
            labels=batch['labels'],
        )
        
        loss = outputs['loss']
        
        # Compute metrics
        preds = outputs['logits'].argmax(dim=-1)
        acc = (preds == batch['labels']).float().mean()
        
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/acc', acc, prog_bar=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self.forward(
            question_input_ids=batch['question_input_ids'],
            question_attention_mask=batch['question_attention_mask'],
            path_input_ids=batch['path_input_ids'],
            path_attention_mask=batch['path_attention_mask'],
            candidate_mask=batch['candidate_mask'],
            labels=batch['labels'],
        )
        
        preds = outputs['logits'].argmax(dim=-1)
        acc = (preds == batch['labels']).float().mean()
        
        self.log('test/acc', acc, sync_dist=True)
        
        return {'preds': preds, 'labels': batch['labels'], 'logits': outputs['logits']}
    
    def configure_optimizers(self):
        # Only train non-frozen parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Linear warmup then cosine decay
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
    
    @torch.no_grad()
    def score_paths(
        self,
        question: str,
        candidate_paths: List[List[str]],
        tokenizer: AutoTokenizer,
        top_k: int = 10,
    ) -> List[Tuple[List[str], float]]:
        """
        Score and rank candidate paths for a question.
        
        Args:
            question: Question text
            candidate_paths: List of relation paths (each path is a list of relations)
            tokenizer: Tokenizer for encoding
            top_k: Number of top results to return
            
        Returns:
            List of (path, score) tuples sorted by score descending
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Encode question
        q_tokens = tokenizer(
            question,
            max_length=self.hparams.max_question_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        q_input_ids = q_tokens['input_ids'].to(device)
        q_attention_mask = q_tokens['attention_mask'].to(device)
        
        # Encode paths
        path_texts = [' -> '.join(p) for p in candidate_paths]
        p_tokens = tokenizer(
            path_texts,
            max_length=self.hparams.max_path_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        C = len(candidate_paths)
        p_input_ids = p_tokens['input_ids'].unsqueeze(0).to(device)  # [1, C, seq_len]
        p_attention_mask = p_tokens['attention_mask'].unsqueeze(0).to(device)
        candidate_mask = torch.ones(1, C, device=device)
        
        # Forward
        outputs = self.forward(
            question_input_ids=q_input_ids,
            question_attention_mask=q_attention_mask,
            path_input_ids=p_input_ids,
            path_attention_mask=p_attention_mask,
            candidate_mask=candidate_mask,
        )
        
        scores = outputs['logits'].squeeze(0).cpu().numpy()
        
        # Sort and return top-k
        indices = scores.argsort()[::-1][:top_k]
        return [(candidate_paths[i], float(scores[i])) for i in indices]


class HybridPathRanker(pl.LightningModule):
    """
    Hybrid ranker that combines Bi-Encoder scores with Diffusion scores.
    
    Uses pre-trained models and learns a fusion layer.
    """
    
    def __init__(
        self,
        bi_encoder_checkpoint: str,
        diffusion_scorer_checkpoint: Optional[str] = None,
        fusion_type: str = 'learned',  # 'learned', 'average', 'weighted'
        diffusion_weight: float = 0.3,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        from .path_ranker import PathRankerModel
        
        # Load Bi-Encoder
        self.bi_encoder = PathRankerModel.load_from_checkpoint(bi_encoder_checkpoint)
        for param in self.bi_encoder.parameters():
            param.requires_grad = False
        
        # Load Diffusion Scorer (optional)
        self.diffusion_scorer = None
        if diffusion_scorer_checkpoint:
            self.diffusion_scorer = DiffusionPathScorer.load_from_checkpoint(
                diffusion_scorer_checkpoint
            )
            for param in self.diffusion_scorer.parameters():
                param.requires_grad = False
        
        # Fusion layer
        if fusion_type == 'learned':
            self.fusion = nn.Sequential(
                nn.Linear(2, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
        else:
            self.fusion = None
        
        self.fusion_type = fusion_type
        self.diffusion_weight = diffusion_weight
        self.learning_rate = learning_rate
    
    def forward(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        path_input_ids: torch.Tensor,
        path_attention_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute fused scores from both models."""
        
        # Get Bi-Encoder scores
        bi_outputs = self.bi_encoder(
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            path_input_ids=path_input_ids,
            path_attention_mask=path_attention_mask,
            candidate_mask=candidate_mask,
        )
        bi_scores = bi_outputs['logits']  # [B, C]
        
        # Get Diffusion scores if available
        if self.diffusion_scorer is not None:
            diff_outputs = self.diffusion_scorer(
                question_input_ids=question_input_ids,
                question_attention_mask=question_attention_mask,
                path_input_ids=path_input_ids,
                path_attention_mask=path_attention_mask,
                candidate_mask=candidate_mask,
            )
            diff_scores = diff_outputs['logits']  # [B, C]
        else:
            diff_scores = torch.zeros_like(bi_scores)
        
        # Fuse scores
        if self.fusion_type == 'learned':
            stacked = torch.stack([bi_scores, diff_scores], dim=-1)  # [B, C, 2]
            fused = self.fusion(stacked).squeeze(-1)  # [B, C]
        elif self.fusion_type == 'weighted':
            fused = (1 - self.diffusion_weight) * bi_scores + self.diffusion_weight * diff_scores
        else:  # average
            fused = (bi_scores + diff_scores) / 2
        
        result = {'logits': fused, 'bi_scores': bi_scores, 'diff_scores': diff_scores}
        
        if labels is not None:
            result['loss'] = F.cross_entropy(fused, labels)
        
        return result


if __name__ == '__main__':
    # Quick test
    print("Testing DiffusionPathScorer...")
    
    model = DiffusionPathScorer(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller for testing
        hidden_dim=384,
    )
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test with dummy data
    B, C, seq_len = 2, 10, 32
    dummy_batch = {
        'question_input_ids': torch.randint(0, 1000, (B, seq_len)),
        'question_attention_mask': torch.ones(B, seq_len, dtype=torch.long),
        'path_input_ids': torch.randint(0, 1000, (B, C, seq_len)),
        'path_attention_mask': torch.ones(B, C, seq_len, dtype=torch.long),
        'candidate_mask': torch.ones(B, C),
        'labels': torch.randint(0, C, (B,)),
    }
    
    outputs = model(**dummy_batch)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
