"""
Fast Question Classifier Inference Module

Provides optimized inference for question type classification (one-hop vs multi-hop).
Supports both PyTorch and ONNX runtime backends.
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from transformers import AutoTokenizer


LABEL_NAMES = ['one_hop', 'multi_hop']


class QuestionClassifierInference:
    """
    Fast inference for question type classification.
    
    Uses ONNX runtime if available, else falls back to PyTorch.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        use_onnx: bool = True,
        device: str = 'cuda',
        max_length: int = 128,
    ):
        """
        Initialize classifier for inference.
        
        Args:
            model_path: Path to ONNX model or PyTorch checkpoint
            tokenizer_path: Path to tokenizer (auto-detected if None)
            use_onnx: Whether to use ONNX runtime
            device: Device for PyTorch inference ('cuda' or 'cpu')
            max_length: Maximum token sequence length
        """
        self.max_length = max_length
        self.device = device
        self.use_onnx = use_onnx and HAS_ONNX
        
        # Load tokenizer
        if tokenizer_path is None:
            # Try to find tokenizer directory next to model
            model_dir = Path(model_path).parent
            tokenizer_dir = model_dir / 'tokenizer'
            if tokenizer_dir.exists():
                tokenizer_path = str(tokenizer_dir)
            else:
                tokenizer_path = 'huawei-noah/TinyBERT_General_6L_768D'
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load model
        if self.use_onnx and model_path.endswith('.onnx'):
            self._load_onnx(model_path)
        else:
            self._load_pytorch(model_path)
    
    def _load_onnx(self, model_path: str):
        """Load ONNX model."""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.is_onnx = True
        print(f"Loaded ONNX model from {model_path}")
    
    def _load_pytorch(self, model_path: str):
        """Load PyTorch model."""
        from models.question_classifier import QuestionTypeClassifier
        self.model = QuestionTypeClassifier.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.to(self.device)
        self.is_onnx = False
        print(f"Loaded PyTorch model from {model_path}")
    
    def predict(self, question: str) -> Tuple[str, float]:
        """
        Predict question type for a single question.
        
        Returns:
            (class_name, confidence)
        """
        results = self.predict_batch([question])
        return results[0]
    
    def predict_batch(
        self,
        questions: List[str],
        batch_size: int = 32,
    ) -> List[Tuple[str, float]]:
        """
        Batch prediction for efficiency.
        
        Returns:
            List of (class_name, confidence) tuples
        """
        results = []
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            
            # Tokenize
            encoding = self.tokenizer(
                batch_questions,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt' if not self.is_onnx else 'np'
            )
            
            if self.is_onnx:
                # ONNX inference
                outputs = self.session.run(
                    None,
                    {
                        'input_ids': encoding['input_ids'].astype(np.int64),
                        'attention_mask': encoding['attention_mask'].astype(np.int64),
                    }
                )
                logits = outputs[0]
                probs = self._softmax(logits)
            else:
                # PyTorch inference
                with torch.no_grad():
                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    logits = self.model(input_ids, attention_mask)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Extract predictions
            preds = probs.argmax(axis=-1)
            for j in range(len(batch_questions)):
                pred = preds[j]
                conf = probs[j, pred]
                results.append((LABEL_NAMES[pred], float(conf)))
        
        return results
    
    def _softmax(self, x):
        """Numpy softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)


def benchmark_inference(
    model_path: str,
    num_samples: int = 1000,
    batch_size: int = 32,
):
    """
    Benchmark inference speed.
    
    Args:
        model_path: Path to model
        num_samples: Number of samples to test
        batch_size: Batch size for inference
    """
    import time
    
    clf = QuestionClassifierInference(model_path)
    
    # Generate test questions
    test_questions = [
        f"What is the capital of country {i}?"
        for i in range(num_samples)
    ]
    
    # Warmup
    _ = clf.predict_batch(test_questions[:10])
    
    # Benchmark
    start = time.time()
    _ = clf.predict_batch(test_questions, batch_size=batch_size)
    elapsed = time.time() - start
    
    print(f"\nInference Benchmark:")
    print(f"  Samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Time per sample: {1000 * elapsed / num_samples:.2f}ms")
    print(f"  Throughput: {num_samples / elapsed:.1f} samples/s")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--question', type=str, help='Question to classify')
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_inference(args.model)
    elif args.question:
        clf = QuestionClassifierInference(args.model)
        label, conf = clf.predict(args.question)
        print(f"Question: {args.question}")
        print(f"Prediction: {label} (confidence: {conf:.3f})")
    else:
        # Demo
        clf = QuestionClassifierInference(args.model)
        
        test_questions = [
            "What country is Paris in?",
            "Who is the director of the movie that won the Oscar in 2020?",
            "What is the capital of France?",
            "What language do people speak in the country where the Eiffel Tower is located?",
        ]
        
        print("Demo predictions:")
        for q in test_questions:
            label, conf = clf.predict(q)
            print(f"  {label} ({conf:.2f}): {q}")
