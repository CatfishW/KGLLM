"""
Subgraph Filter: Filter retrieved paths based on question classification.

Uses the QuestionClassifier to predict question complexity and filters
candidate paths to match the expected hop count.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from .question_classifier import QuestionClassifier, QuestionType, ClassificationResult
from .retriever import RetrievedPath


@dataclass
class FilterResult:
    """Result from subgraph filtering."""
    original_count: int
    filtered_count: int
    question_type: QuestionType
    classification_confidence: float
    applied_filter: bool
    reason: str


class SubgraphFilter:
    """
    Filter retrieved paths based on question type classification.
    
    When enabled, analyzes the question to predict its complexity
    and filters out paths that don't match the expected hop count.
    
    Example:
        >>> classifier = QuestionClassifier(mode="rule")
        >>> filter = SubgraphFilter(classifier, enable=True)
        >>> 
        >>> # For a simple question, filter out long paths
        >>> paths = filter.filter_paths("Who is Obama's wife?", all_paths)
    """
    
    def __init__(
        self,
        classifier: QuestionClassifier,
        enable: bool = True,
        min_results: int = 3,
        strict: bool = False
    ):
        """
        Initialize the subgraph filter.
        
        Args:
            classifier: QuestionClassifier instance
            enable: If False, acts as pass-through (no filtering)
            min_results: Minimum results to keep (fallback if too aggressive)
            strict: If True, always apply filter even if few results remain
        """
        self.classifier = classifier
        self.enabled = enable
        self.min_results = min_results
        self.strict = strict
    
    def filter_paths(
        self,
        question: str,
        paths: List[RetrievedPath],
        strict: Optional[bool] = None
    ) -> Tuple[List[RetrievedPath], FilterResult]:
        """
        Filter paths based on question type.
        
        Args:
            question: Natural language question
            paths: List of retrieved paths
            strict: Override default strict mode
            
        Returns:
            Tuple of (filtered_paths, filter_result)
        """
        strict = strict if strict is not None else self.strict
        
        # Pass-through if disabled
        if not self.enabled:
            return paths, FilterResult(
                original_count=len(paths),
                filtered_count=len(paths),
                question_type=QuestionType.ONE_HOP,
                classification_confidence=1.0,
                applied_filter=False,
                reason="Filter disabled (pass-through mode)"
            )
        
        # Classify question
        classification = self.classifier.classify(question)
        min_hops = classification.min_hops
        max_hops = classification.max_hops
        
        # Filter by hop count
        filtered = [
            p for p in paths
            if min_hops <= len(p.relations) <= max_hops
        ]
        
        # Check if filter was too aggressive
        if len(filtered) < self.min_results and not strict:
            # Relax constraints
            return paths, FilterResult(
                original_count=len(paths),
                filtered_count=len(paths),
                question_type=classification.question_type,
                classification_confidence=classification.confidence,
                applied_filter=False,
                reason=f"Filter too aggressive ({len(filtered)} < {self.min_results}), kept all paths"
            )
        
        # Apply filter
        return filtered, FilterResult(
            original_count=len(paths),
            filtered_count=len(filtered),
            question_type=classification.question_type,
            classification_confidence=classification.confidence,
            applied_filter=True,
            reason=f"Filtered to {min_hops}-{max_hops} hop paths ({classification.reasoning})"
        )
    
    def filter_by_type(
        self,
        paths: List[RetrievedPath],
        question_type: QuestionType
    ) -> List[RetrievedPath]:
        """
        Filter paths by a specific question type.
        
        Args:
            paths: List of paths
            question_type: Target question type
            
        Returns:
            Filtered paths
        """
        min_hops, max_hops = question_type.hop_range
        return [
            p for p in paths
            if min_hops <= len(p.relations) <= max_hops
        ]
    
    def analyze_paths(
        self,
        paths: List[RetrievedPath]
    ) -> dict:
        """
        Analyze path distribution by hop count.
        
        Args:
            paths: List of paths
            
        Returns:
            Dictionary with hop count statistics
        """
        if not paths:
            return {'total': 0, 'by_hops': {}}
        
        by_hops = {}
        for p in paths:
            hop_count = len(p.relations)
            if hop_count not in by_hops:
                by_hops[hop_count] = 0
            by_hops[hop_count] += 1
        
        return {
            'total': len(paths),
            'by_hops': by_hops,
            'min_hops': min(by_hops.keys()),
            'max_hops': max(by_hops.keys()),
            'avg_hops': sum(k * v for k, v in by_hops.items()) / len(paths)
        }


class MultiTypeFilter:
    """
    Filter that can combine multiple question types.
    
    Useful when classification is uncertain and you want to
    include paths from multiple complexity levels.
    """
    
    def __init__(self, classifier: QuestionClassifier, enable: bool = True):
        self.classifier = classifier
        self.enabled = enable
    
    def filter_with_confidence(
        self,
        question: str,
        paths: List[RetrievedPath],
        confidence_threshold: float = 0.7
    ) -> Tuple[List[RetrievedPath], ClassificationResult]:
        """
        Filter paths, expanding range if classification confidence is low.
        
        Args:
            question: Question string
            paths: Candidate paths
            confidence_threshold: Below this, expand hop range
            
        Returns:
            Filtered paths and classification result
        """
        if not self.enabled:
            classification = ClassificationResult(
                question_type=QuestionType.ONE_HOP,
                confidence=1.0,
                min_hops=1,
                max_hops=5,
                reasoning="Filter disabled"
            )
            return paths, classification
        
        classification = self.classifier.classify(question)
        
        # If confident, use strict filtering
        if classification.confidence >= confidence_threshold:
            min_hops = classification.min_hops
            max_hops = classification.max_hops
        else:
            # Expand range for low confidence
            min_hops = max(1, classification.min_hops - 1)
            max_hops = classification.max_hops + 1
            classification.reasoning += f" (expanded range due to low confidence: {classification.confidence:.2f})"
        
        filtered = [
            p for p in paths
            if min_hops <= len(p.relations) <= max_hops
        ]
        
        # Fallback if empty
        if not filtered:
            return paths, classification
        
        return filtered, classification
