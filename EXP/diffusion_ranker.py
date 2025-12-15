"""
Diffusion Ranker: Use diffusion models to rerank retrieved paths.

The diffusion model acts as a scoring function:
- Given a question and a candidate path, compute reconstruction likelihood
- Higher likelihood = more relevant path

This module adapts the existing DiscreteDiffusion for ranking rather than generation.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np

# Add Core to path for diffusion imports
CORE_PATH = Path(__file__).parent.parent / "Core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))


@dataclass
class DiffusionScore:
    """Score from diffusion model."""
    path_index: int
    score: float
    reconstruction_loss: float


# --- Custom Definitions to avoid Dependencies ---

try:
    from Core.modules.diffusion import PathDiffusionTransformer, DiscreteDiffusion
except ImportError:
    PathDiffusionTransformer = None
    DiscreteDiffusion = None

import torch
import torch.nn as nn
from typing import Tuple

class QuestionEncoder(nn.Module):
    """Encode questions using pretrained transformer."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", output_dim: int = 256, freeze: bool = False):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(self.hidden_size, output_dim)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.proj(outputs.last_hidden_state)
        mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sequence_output, sum_embeddings / sum_mask

class DiffusionModel(nn.Module):
    """Wrapper for diffusion components (Raw PyTorch)."""
    def __init__(self, num_entities, num_relations, hidden_dim=256, max_path_length=10, predict_entities=True):
        super().__init__()
        self.question_encoder = QuestionEncoder(output_dim=hidden_dim)
        self.denoiser = PathDiffusionTransformer(
            num_entities=num_entities, num_relations=num_relations,
            hidden_dim=hidden_dim, max_path_length=max_path_length,
            predict_entities=predict_entities, question_dim=hidden_dim
        )
        self.diffusion = DiscreteDiffusion(num_entities=num_entities, num_relations=num_relations)
        # Hack to match PL structure
        self.model = self 
        
    def forward(self, *args, **kwargs):
        # Forward to denoiser simply to match PL forward call expectation in score_path_likelihood
        # But score_path_likelihood specifically calls self.model.model(...)
        # We mapped self.model.model = self.
        # So we need to handle the call signature of denoiser
        # The PL model.forward calls _forward_diffusion_single usually, or returns loss?
        # score_path_likelihood calls self.model.model(...) directly.
        # The PL module usually wraps 'model' which is likely PathDiffusionTransformer.
        # So self.model.model should point to self.denoiser?
        # No, PL logic: self.model is KGPathDiffusionWrapper. self.model.model is whatever.
        # In KGPathDiffusionLightning, self.model is KGPathDiffusionModel.
        # In KGPathDiffusionModel, self.model is PathDiffusionTransformer??
        # Let's check KGPathDiffusionModel in Step 101.
        # KGPathDiffusionModel DOES NOT have .model. It inherits nn.Module.
        # It has self.question_encoder, self.diffusion, self.denoiser (maybe called .model?).
        # Looking at Step 101: KGPathDiffusionModel __init__: self.model = PathDiffusionTransformer(...)
        # Yes! So KGPathDiffusionModel has .model attribute which is the transformer.
        # So we need self.model attribute in our wrapper.
        pass

# Update wrapper to match hierarchy
class DiffusionModelWrapper(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim=256):
        super().__init__()
        self.question_encoder = QuestionEncoder(output_dim=hidden_dim)
        self.model = PathDiffusionTransformer(
            num_entities=num_entities, num_relations=num_relations,
            hidden_dim=hidden_dim, max_path_length=10,
            predict_entities=True, question_dim=hidden_dim
        )
        self.diffusion = DiscreteDiffusion(num_entities=num_entities, num_relations=num_relations)

    
    
class DiffusionRanker:
    """
    Diffusion-based path reranker.
    
    Uses the diffusion model's denoising capability to score paths:
    - Encode question + path
    - Run diffusion forward process (add noise)
    - Measure how well the model reconstructs the original path
    - Lower reconstruction loss = higher relevance score
    
    Two modes:
    1. "likelihood": Use diffusion loss as score (requires forward pass)
    2. "embedding": Use path embedding similarity (faster, less accurate)
    
    Example:
        >>> ranker = DiffusionRanker(enable=True, mode="likelihood")
        >>> reranked = ranker.rerank("Who is Obama's wife?", candidate_paths)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        enable: bool = True,
        mode: str = "embedding",  # "likelihood" or "embedding"
        num_diffusion_steps: int = 10,
        score_weight: float = 0.5,
        vocab_path: Optional[str] = None
    ):
        """
        Initialize diffusion ranker.
        
        Args:
            model_path: Path to trained diffusion model checkpoint
            enable: If False, acts as pass-through
            mode: "likelihood" (accurate) or "embedding" (fast)
            num_diffusion_steps: Number of diffusion steps for scoring
            score_weight: Weight for combining diffusion score with retrieval score
            vocab_path: Path to vocabulary file for relation encoding
        """
        self.enabled = enable
        self.mode = mode
        self.num_diffusion_steps = num_diffusion_steps
        self.score_weight = score_weight
        self.model = None
        self.vocab = None
        
        if not enable:
            return
        
        # Try to load model
        if model_path:
            self._load_model(model_path)
        
        if vocab_path:
            self._load_vocab(vocab_path)
    
    def _load_model(self, model_path: str):
        """Load diffusion model from checkpoint."""
        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            print(f"Diffusion model not found at {model_path}")
            return

        try:
            # Try loading as PL module first (if package exists and it's a PL ckpt)
            try:
                import pytorch_lightning as pl
                from kg_path_diffusion import KGPathDiffusionLightning
                self.model = KGPathDiffusionLightning.load_from_checkpoint(
                    str(checkpoint_path),
                    strict=False
                )
                print(f"Loaded diffusion model (PL) from {model_path}")
            except (ImportError, Exception):
                # Fallback to loading raw state dict
                import torch
                state = torch.load(checkpoint_path, map_location='cpu')
                
                # Check for config
                if 'config' in state:
                    config = state['config']
                    num_entities = config['num_entities']
                    num_relations = config['num_relations']
                else:
                    # Try to infer or use defaults
                    # If we loaded vocab, we can use that
                    if hasattr(self, 'vocab') and self.vocab:
                        rel2idx = self.vocab.get('relation2idx', {})
                        ent2idx = self.vocab.get('entity2idx', {})
                        num_entities = len(ent2idx) + 1
                        num_relations = len(rel2idx) + 1
                    else:
                        # Fallback defaults (risky)
                        num_entities = 2000
                        num_relations = 100
                
                # Instantiate custom wrapper
                self.model = DiffusionModelWrapper(num_entities, num_relations)
                
                # Load weights
                # Keys might need adjustment if they start with 'model.'
                # State dict from train_diffusion.py has top-level keys matching wrapper
                # (question_encoder, denoiser->model, diffusion)
                # But my wrapper in this file uses .model for denoiser to match existing logic.
                # My train_diffusion.py used .denoiser.
                # So I need to map keys.
                
                raw_state = state['state_dict']
                new_state = {}
                for k, v in raw_state.items():
                    # If trained with train_diffusion.py, keys are:
                    # question_encoder.*
                    # denoiser.*
                    # diffusion.*
                    # We want:
                    # question_encoder.*
                    # model.* (since DiffusionModelWrapper uses .model for denoiser)
                    # diffusion.*
                    if k.startswith('denoiser.'):
                        new_state[k.replace('denoiser.', 'model.')] = v
                    else:
                        new_state[k] = v
                        
                self.model.load_state_dict(new_state, strict=False)
                print(f"Loaded diffusion model (Raw) from {model_path}")
            
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
        except Exception as e:
            print(f"Failed to load diffusion model: {e}")
            self.model = None
            import traceback
            traceback.print_exc()
    
    def _load_vocab(self, vocab_path: str):
        """Load relation vocabulary."""
        try:
            import json
            
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
            
            self.relation2idx = self.vocab.get('relation2idx', {})
            print(f"Loaded vocabulary with {len(self.relation2idx)} relations")
            
        except Exception as e:
            print(f"Failed to load vocabulary: {e}")
    
    def _encode_relations(self, relations: List[str]) -> List[int]:
        """Convert relation strings to indices."""
        if not self.relation2idx:
            return list(range(len(relations)))
        
        unk_idx = self.relation2idx.get('<UNK>', 1)
        return [self.relation2idx.get(r, unk_idx) for r in relations]
    
    def score_path_likelihood(
        self,
        question: str,
        relations: List[str]
    ) -> float:
        """
        Score a single path using diffusion likelihood.
        
        Args:
            question: Question string
            relations: List of relation strings
            
        Returns:
            Score (higher = more relevant)
        """
        if self.model is None:
            return 0.0
        
        try:
            import torch
            from transformers import AutoTokenizer
            
            # Tokenize question
            tokenizer = AutoTokenizer.from_pretrained(
                self.model.model.question_encoder.encoder.config._name_or_path
            )
            
            inputs = tokenizer(
                question,
                return_tensors='pt',
                padding='max_length',
                max_length=64,
                truncation=True
            )
            
            # Encode relations
            relation_ids = self._encode_relations(relations)
            relation_tensor = torch.tensor([relation_ids], dtype=torch.long)
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            relation_tensor = relation_tensor.to(device)
            
            # Compute diffusion loss (lower = better reconstruction = more relevant)
            with torch.no_grad():
                # Use a few diffusion steps to estimate likelihood
                total_loss = 0.0
                for _ in range(self.num_diffusion_steps):
                    outputs = self.model.model(
                        question_input_ids=inputs['input_ids'],
                        question_attention_mask=inputs['attention_mask'],
                        target_entities=torch.zeros_like(relation_tensor),  # Dummy
                        target_relations=relation_tensor
                    )
                    loss_val = outputs['relation_loss'].item()
                    import math
                    if math.isnan(loss_val):
                        loss_val = 1e9 # Penalty for NaN
                    total_loss += loss_val
                
                avg_loss = total_loss / self.num_diffusion_steps
            
            # Convert loss to score (negative loss so lower loss = higher score)
            score = -avg_loss
            return score
            
        except Exception as e:
            print(f"Diffusion scoring failed: {e}")
            return -1e9 # Low score on error -> 203:             return 0.0
    
    def score_path_embedding(
        self,
        question: str,
        relations: List[str]
    ) -> float:
        """
        Score path using embedding similarity (fast mode).
        
        Uses the question encoder to embed both question and path,
        then computes cosine similarity.
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            
            # Convert relations to natural language
            path_text = " -> ".join(relations)
            
            # Embed both
            embeddings = self._embedding_model.encode(
                [question, path_text],
                normalize_embeddings=True
            )
            
            # Cosine similarity
            score = np.dot(embeddings[0], embeddings[1])
            return float(score)
            
        except Exception as e:
            print(f"Embedding scoring failed: {e}")
            return 0.0
    
    def score_paths(
        self,
        question: str,
        paths: List['RetrievedPath']
    ) -> List[DiffusionScore]:
        """
        Score multiple paths.
        
        Args:
            question: Question string
            paths: List of RetrievedPath objects
            
        Returns:
            List of DiffusionScore objects
        """
        if not self.enabled:
            return [
                DiffusionScore(i, paths[i].score, 0.0)
                for i in range(len(paths))
            ]
        
        scores = []
        for i, path in enumerate(paths):
            if self.mode == "likelihood" and self.model is not None:
                score = self.score_path_likelihood(question, path.relations)
            else:
                score = self.score_path_embedding(question, path.relations)
            
            scores.append(DiffusionScore(
                path_index=i,
                score=score,
                reconstruction_loss=-score  # Store for debugging
            ))
        
        return scores
    
    def rerank(
        self,
        question: str,
        paths: List['RetrievedPath'],
        top_k: Optional[int] = None
    ) -> List['RetrievedPath']:
        """
        Rerank paths using diffusion scores.
        
        Combines retrieval score with diffusion score using weighted average.
        
        Args:
            question: Question string
            paths: List of RetrievedPath objects
            top_k: Return only top-k paths (None = return all)
            
        Returns:
            Reranked list of paths
        """
        if not self.enabled or not paths:
            return paths[:top_k] if top_k else paths
        
        # Get diffusion scores
        diff_scores = self.score_paths(question, paths)
        
        # Normalize scores
        if diff_scores:
            min_score = min(s.score for s in diff_scores)
            max_score = max(s.score for s in diff_scores)
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            normalized = [
                (s.score - min_score) / score_range
                for s in diff_scores
            ]
        else:
            normalized = [0.0] * len(paths)
        
        # Combine scores
        combined = []
        for i, path in enumerate(paths):
            # Normalize retrieval score to 0-1 range
            retrieval_score = path.score
            
            # Weighted combination
            combined_score = (
                (1 - self.score_weight) * retrieval_score +
                self.score_weight * normalized[i]
            )
            
            combined.append((path, combined_score, i))
        
        # Sort by combined score (descending)
        combined.sort(key=lambda x: -x[1])
        
        # Update ranks and scores
        reranked = []
        for new_rank, (path, combined_score, orig_rank) in enumerate(combined, 1):
            # Create new path with updated rank/score
            from .retriever import RetrievedPath
            new_path = RetrievedPath(
                rank=new_rank,
                score=combined_score,
                relation_chain=path.relation_chain,
                relations=path.relations,
                natural_language=path.natural_language,
                example_questions=path.example_questions,
                example_entities=path.example_entities,
                frequency=path.frequency
            )
            reranked.append(new_path)
        
        return reranked[:top_k] if top_k else reranked
    
    def is_available(self) -> bool:
        """Check if diffusion ranker is ready to use."""
        if not self.enabled:
            return False
        if self.mode == "likelihood":
            return self.model is not None
        return True  # Embedding mode always available


class EnsembleRanker:
    """
    Ensemble of multiple ranking methods.
    
    Combines retrieval score, diffusion score, and other signals.
    """
    
    def __init__(
        self,
        diffusion_ranker: Optional[DiffusionRanker] = None,
        retrieval_weight: float = 0.4,
        diffusion_weight: float = 0.3,
        frequency_weight: float = 0.3
    ):
        self.diffusion_ranker = diffusion_ranker
        self.retrieval_weight = retrieval_weight
        self.diffusion_weight = diffusion_weight
        self.frequency_weight = frequency_weight
    
    def rerank(
        self,
        question: str,
        paths: List['RetrievedPath'],
        top_k: Optional[int] = None
    ) -> List['RetrievedPath']:
        """Rerank using ensemble of signals."""
        if not paths:
            return paths
        
        # Get diffusion scores
        if self.diffusion_ranker and self.diffusion_ranker.enabled:
            diff_scores = self.diffusion_ranker.score_paths(question, paths)
            diff_scores = [s.score for s in diff_scores]
        else:
            diff_scores = [0.0] * len(paths)
        
        # Normalize all scores
        def normalize(scores):
            if not scores:
                return scores
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0
            return [(s - min_s) / range_s for s in scores]
        
        retrieval_scores = normalize([p.score for p in paths])
        diff_scores = normalize(diff_scores)
        freq_scores = normalize([p.frequency for p in paths])
        
        # Combine
        combined = []
        for i, path in enumerate(paths):
            score = (
                self.retrieval_weight * retrieval_scores[i] +
                self.diffusion_weight * diff_scores[i] +
                self.frequency_weight * freq_scores[i]
            )
            combined.append((path, score))
        
        # Sort
        combined.sort(key=lambda x: -x[1])
        
        # Update ranks
        from .retriever import RetrievedPath
        reranked = []
        for new_rank, (path, score) in enumerate(combined, 1):
            new_path = RetrievedPath(
                rank=new_rank,
                score=score,
                relation_chain=path.relation_chain,
                relations=path.relations,
                natural_language=path.natural_language,
                example_questions=path.example_questions,
                example_entities=path.example_entities,
                frequency=path.frequency
            )
            reranked.append(new_path)
        
        return reranked[:top_k] if top_k else reranked
