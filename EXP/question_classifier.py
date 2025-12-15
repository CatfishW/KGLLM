"""
Question Classifier: Classify questions by reasoning complexity.

Supports four question types:
- LITERAL: Direct attribute lookup (dates, numbers, names)
- DEFINITIONAL: Type/category questions
- ONE_HOP: Single relation reasoning
- TWO_HOP: Multi-hop reasoning (2+ relations)

This module provides both:
1. Rule-based classifier (fast, no training required)
2. Neural classifier (trainable, more accurate)

Both modes can be toggled for experimentation.
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class QuestionType(Enum):
    """Question complexity types."""
    LITERAL = "literal"           # Direct attribute (birthday, count, etc.)
    DEFINITIONAL = "definitional" # Type/category questions
    ONE_HOP = "one_hop"           # Single relation reasoning
    TWO_HOP = "two_hop"           # Multi-hop reasoning (2+ hops)
    
    @property
    def hop_range(self) -> Tuple[int, int]:
        """Expected path hop range for this question type."""
        ranges = {
            QuestionType.LITERAL: (1, 1),
            QuestionType.DEFINITIONAL: (1, 1),
            QuestionType.ONE_HOP: (1, 2),
            QuestionType.TWO_HOP: (2, 5),
        }
        return ranges[self]
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        descriptions = {
            QuestionType.LITERAL: "Direct attribute lookup",
            QuestionType.DEFINITIONAL: "Type or category question",
            QuestionType.ONE_HOP: "Single relation reasoning",
            QuestionType.TWO_HOP: "Multi-hop reasoning",
        }
        return descriptions[self]


@dataclass
class ClassificationResult:
    """Result from question classification."""
    question_type: QuestionType
    confidence: float
    min_hops: int
    max_hops: int
    reasoning: str  # Explanation for the classification
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'question_type': self.question_type.value,
            'confidence': self.confidence,
            'min_hops': self.min_hops,
            'max_hops': self.max_hops,
            'reasoning': self.reasoning
        }


# === Rule-based patterns ===

LITERAL_PATTERNS = [
    # Date/time patterns
    r'\bwhen (was|did|is)\b',
    r'\bwhat (year|date|day|time)\b',
    r'\bhow (old|long|many|much)\b',
    r'\bbirthday\b',
    r'\bborn\b',
    r'\bdied\b',
    r'\bfounded\b',
    r'\bestablished\b',
    # Numeric patterns
    r'\bnumber of\b',
    r'\bpopulation\b',
    r'\bheight\b',
    r'\bweight\b',
    r'\bage\b',
]

DEFINITIONAL_PATTERNS = [
    r'^what is a\b',
    r'^what are\b',
    r'\bdefine\b',
    r'\bdefinition of\b',
    r'\bmeaning of\b',
    r'\btype of\b',
    r'\bkind of\b',
    r'\bcategory\b',
    r'^is .+ a\b',
]

TWO_HOP_PATTERNS = [
    # Possessive chains
    r"'s .+'s",
    r"'s .+ 's",
    # Multi-relation indicators
    r'\b(and|or) (what|who|where|when)\b',
    r'\bthrough\b',
    r'\bvia\b',
    # Complex relations
    r'\bsister.+brother\b',
    r'\bbrother.+wife\b',
    r'\bwife.+mother\b',
    r'\bfather.+brother\b',
    r'\bparent.+child\b',
    # Indirect questions
    r'\brelated to\b',
    r'\bconnected to\b',
    r'\blinked to\b',
]

ONE_HOP_RELATION_KEYWORDS = [
    # Direct relations
    'wife', 'husband', 'spouse', 'married',
    'father', 'mother', 'parent', 'child', 'children',
    'brother', 'sister', 'sibling',
    'capital', 'country', 'city', 'state', 'continent',
    'director', 'actor', 'actress', 'starring',
    'author', 'wrote', 'written by',
    'plays', 'played', 'play',
    'born', 'died', 'nationality',
    'language', 'currency', 'religion',
    'founder', 'ceo', 'president', 'leader',
]


class RuleBasedClassifier:
    """
    Rule-based question classifier using pattern matching.
    
    Fast and interpretable, good baseline for comparison.
    """
    
    def __init__(self):
        # Compile regex patterns
        self.literal_patterns = [re.compile(p, re.IGNORECASE) for p in LITERAL_PATTERNS]
        self.definitional_patterns = [re.compile(p, re.IGNORECASE) for p in DEFINITIONAL_PATTERNS]
        self.two_hop_patterns = [re.compile(p, re.IGNORECASE) for p in TWO_HOP_PATTERNS]
    
    def classify(self, question: str) -> ClassificationResult:
        """Classify a question using rule-based patterns."""
        question_lower = question.lower().strip()
        
        # Check definitional first (most specific patterns)
        for pattern in self.definitional_patterns:
            if pattern.search(question_lower):
                return ClassificationResult(
                    question_type=QuestionType.DEFINITIONAL,
                    confidence=0.85,
                    min_hops=1,
                    max_hops=1,
                    reasoning=f"Matched definitional pattern: {pattern.pattern}"
                )
        
        # Check literal patterns
        for pattern in self.literal_patterns:
            if pattern.search(question_lower):
                return ClassificationResult(
                    question_type=QuestionType.LITERAL,
                    confidence=0.80,
                    min_hops=1,
                    max_hops=1,
                    reasoning=f"Matched literal pattern: {pattern.pattern}"
                )
        
        # Check two-hop patterns
        for pattern in self.two_hop_patterns:
            if pattern.search(question_lower):
                return ClassificationResult(
                    question_type=QuestionType.TWO_HOP,
                    confidence=0.75,
                    min_hops=2,
                    max_hops=5,
                    reasoning=f"Matched two-hop pattern: {pattern.pattern}"
                )
        
        # Count relation keywords as indicator
        keyword_count = sum(
            1 for kw in ONE_HOP_RELATION_KEYWORDS 
            if kw in question_lower
        )
        
        # Default to one-hop with confidence based on keyword matches
        if keyword_count >= 1:
            return ClassificationResult(
                question_type=QuestionType.ONE_HOP,
                confidence=min(0.7, 0.5 + keyword_count * 0.1),
                min_hops=1,
                max_hops=2,
                reasoning=f"Found {keyword_count} relation keyword(s)"
            )
        
        # Fallback: assume one-hop (most common in WebQSP)
        return ClassificationResult(
            question_type=QuestionType.ONE_HOP,
            confidence=0.5,
            min_hops=1,
            max_hops=3,
            reasoning="Default classification (no strong patterns matched)"
        )


class QuestionClassifierModel(nn.Module):
    """Neural classifier architecture (must match training script)."""
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class NeuralClassifier:
    """
    Neural question classifier using sentence transformers.
    
    More accurate but requires training data.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.label_encoder = {
            0: QuestionType.LITERAL,
            1: QuestionType.DEFINITIONAL,
            2: QuestionType.ONE_HOP,
            3: QuestionType.TWO_HOP,
        }
        
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load trained classifier model."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            import json
            
            model_dir = Path(model_path)
            
            # Load config if available to get dimensions
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                hidden_dim = config.get('hidden_dim', 256)
                dropout = config.get('dropout', 0.3)
                embedding_model = config.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2")
            else:
                hidden_dim = 256
                dropout = 0.3
                embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            
            # Load embedding model
            self.encoder = SentenceTransformer(embedding_model)
            
            # Initialize model architecture
            self.classifier = QuestionClassifierModel(
                input_dim=self.encoder.get_sentence_embedding_dimension(),
                hidden_dim=hidden_dim,
                num_classes=4,
                dropout=dropout
            )
            
            # Load weights
            weights_path = model_dir / "classifier.pt"
            if weights_path.exists():
                # Load state dict
                state_dict = torch.load(weights_path, map_location='cpu')
                self.classifier.load_state_dict(state_dict)
                self.classifier.eval()
                self.model = True
                print(f"Loaded neural classifier from {model_path}")
            else:
                print(f"Classifier weights not found at {weights_path}, using fallback")
                self.model = None
                
        except Exception as e:
            print(f"Failed to load neural classifier: {e}")
            self.model = None
    
    def classify(self, question: str) -> ClassificationResult:
        """Classify using neural model."""
        if self.model is None:
            # Fallback to rule-based
            return RuleBasedClassifier().classify(question)
        
        try:
            import torch
            
            # Encode question
            embedding = self.encoder.encode(question, convert_to_tensor=True)
            
            # Classify
            with torch.no_grad():
                logits = self.classifier(embedding.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred_class].item()
            
            question_type = self.label_encoder[pred_class]
            min_hops, max_hops = question_type.hop_range
            
            return ClassificationResult(
                question_type=question_type,
                confidence=confidence,
                min_hops=min_hops,
                max_hops=max_hops,
                reasoning=f"Neural classifier prediction (confidence: {confidence:.2f})"
            )
            
        except Exception as e:
            print(f"Neural classification failed: {e}")
            return RuleBasedClassifier().classify(question)


class QuestionClassifier:
    """
    Main question classifier with mode selection.
    
    Modes:
    - "rule": Fast rule-based classification
    - "neural": Neural network classification (requires trained model)
    - "hybrid": Use neural with rule-based fallback
    
    Example:
        >>> classifier = QuestionClassifier(mode="rule")
        >>> result = classifier.classify("Who is Obama's wife?")
        >>> print(result.question_type)  # ONE_HOP
    """
    
    def __init__(
        self,
        mode: str = "rule",
        model_path: Optional[str] = None,
        enable: bool = True
    ):
        """
        Initialize classifier.
        
        Args:
            mode: "rule", "neural", or "hybrid"
            model_path: Path to trained neural model (optional)
            enable: If False, always returns ONE_HOP (pass-through mode)
        """
        self.mode = mode
        self.enabled = enable
        
        if not enable:
            return
        
        if mode == "rule":
            self.classifier = RuleBasedClassifier()
        elif mode == "neural":
            self.classifier = NeuralClassifier(model_path)
        elif mode == "hybrid":
            self.rule_classifier = RuleBasedClassifier()
            self.neural_classifier = NeuralClassifier(model_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def classify(self, question: str) -> ClassificationResult:
        """
        Classify a question.
        
        Args:
            question: Natural language question
            
        Returns:
            ClassificationResult with type, confidence, and hop range
        """
        if not self.enabled:
            return ClassificationResult(
                question_type=QuestionType.ONE_HOP,
                confidence=1.0,
                min_hops=1,
                max_hops=5,
                reasoning="Classifier disabled (pass-through mode)"
            )
        
        if self.mode == "hybrid":
            # Try neural first, fall back to rules
            neural_result = self.neural_classifier.classify(question)
            if neural_result.confidence >= 0.7:
                return neural_result
            return self.rule_classifier.classify(question)
        
        return self.classifier.classify(question)
    
    def batch_classify(self, questions: List[str]) -> List[ClassificationResult]:
        """Classify multiple questions."""
        return [self.classify(q) for q in questions]
    
    def get_hop_range(self, question_type: QuestionType) -> Tuple[int, int]:
        """Get expected path hop range for a question type."""
        return question_type.hop_range


def infer_question_type_from_paths(
    question: str, 
    paths: List[Dict[str, Any]]
) -> QuestionType:
    """
    Infer question type from ground truth paths (for training data).
    
    Args:
        question: Question string
        paths: List of path dictionaries with 'relations' key
        
    Returns:
        Inferred QuestionType
    """
    if not paths:
        return QuestionType.ONE_HOP
    
    # Get path lengths
    lengths = [len(p.get('relations', [])) for p in paths if p.get('relations')]
    if not lengths:
        return QuestionType.ONE_HOP
    
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    
    q_lower = question.lower()
    
    # Definitional patterns
    if any(kw in q_lower for kw in ['what is a', 'what are', 'define', 'type of']):
        return QuestionType.DEFINITIONAL
    
    # Literal patterns
    if any(kw in q_lower for kw in ['when was', 'how many', 'what year', 'birthday', 'born']):
        return QuestionType.LITERAL
    
    # By path length
    if avg_len >= 2 or max_len >= 3:
        return QuestionType.TWO_HOP
    
    return QuestionType.ONE_HOP
