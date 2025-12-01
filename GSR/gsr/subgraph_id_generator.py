"""
Subgraph ID Generator using T5 for GSR-style relation pattern generation.

Generates subgraph IDs (relation patterns) from questions using a T5 model.
"""

from typing import List, Dict, Optional, Tuple
import json

# Lazy imports for torch and transformers (only when needed)
try:
    import torch
    import torch.nn as nn
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    T5ForConditionalGeneration = None
    T5Tokenizer = None


class SubgraphIDGenerator(nn.Module):
    """
    T5-based generator for subgraph IDs (relation patterns).
    
    Input: Question text
    Output: Subgraph ID (relation pattern string)
    """
    
    def __init__(
        self,
        model_name: str = "t5-small",
        max_length: int = 128,
        num_beams: int = 5
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and transformers are required for SubgraphIDGenerator. "
                            "Install with: pip install torch transformers")
        
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.num_beams = num_beams
        
        # Load T5 model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Add special tokens for subgraph IDs if needed
        # (T5 tokenizer should handle relation names as regular text)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            input_ids: [B, seq_len] Question token IDs
            attention_mask: [B, seq_len] Attention mask
            labels: [B, target_len] Target subgraph ID token IDs
        
        Returns:
            Dictionary with loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }
    
    @torch.no_grad()
    def generate(
        self,
        questions: List[str],
        num_beams: Optional[int] = None,
        max_length: Optional[int] = None,
        top_k: int = 1,
        batch_size: int = 8
    ) -> List[List[str]]:
        """
        Generate subgraph IDs for questions.
        
        Args:
            questions: List of question strings
            num_beams: Number of beams for beam search
            max_length: Maximum generation length
            top_k: Number of top predictions to return per question
            batch_size: Batch size for processing (to avoid OOM)
        
        Returns:
            List of lists of subgraph IDs (one list per question, with top_k predictions)
        """
        num_beams = num_beams or self.num_beams
        max_length = max_length or self.max_length
        
        # Ensure num_return_sequences doesn't exceed num_beams
        num_return = min(top_k, num_beams)
        
        all_generated_ids = []
        
        # Process in batches to avoid OOM
        for batch_start in range(0, len(questions), batch_size):
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]
            
            # Format questions for T5
            input_texts = [f"Question: {q}" for q in batch_questions]
            
            # Tokenize
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return,
                early_stopping=True,
                do_sample=False
            )
            
            # Decode
            batch_generated_ids = []
            for i in range(len(batch_questions)):
                question_predictions = []
                for j in range(num_return):
                    idx = i * num_return + j
                    if idx < len(outputs):
                        pred_text = self.tokenizer.decode(
                            outputs[idx],
                            skip_special_tokens=True
                        )
                        question_predictions.append(pred_text.strip())
                batch_generated_ids.append(question_predictions)
            
            all_generated_ids.extend(batch_generated_ids)
            
            # Clear cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return all_generated_ids
    
    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, model_directory: str, **kwargs):
        """Load model and tokenizer."""
        instance = cls(**kwargs)
        instance.model = T5ForConditionalGeneration.from_pretrained(model_directory)
        instance.tokenizer = T5Tokenizer.from_pretrained(model_directory)
        return instance


def prepare_gsr_training_data(
    data_path: str,
    subgraph_index_path: str,
    output_path: str
):
    """
    Prepare training data for GSR model.
    
    Format: (question, subgraph_id) pairs
    
    Args:
        data_path: Path to dataset with questions and paths
        subgraph_index_path: Path to subgraph index
        output_path: Path to save training data
    """
    from pathlib import Path
    import pandas as pd
    import json
    from .subgraph_index import SubgraphIndex
    
    # Load subgraph index
    index = SubgraphIndex.load(subgraph_index_path)
    
    # Load dataset
    path = Path(data_path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
        samples = df.to_dict('records')
        for sample in samples:
            for key in ['graph', 'paths', 'answer', 'q_entity', 'a_entity']:
                if key in sample and isinstance(sample[key], str):
                    try:
                        sample[key] = json.loads(sample[key])
                    except:
                        pass
    elif path.suffix in ['.jsonl', '.json']:
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            if path.suffix == '.jsonl':
                for line in f:
                    samples.append(json.loads(line))
            else:
                samples = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Create training samples
    training_samples = []
    
    for sample in samples:
        question = sample.get('question', '')
        paths = sample.get('paths', [])
        
        if not question or not paths:
            continue
        
        # Get subgraph ID for each path
        for path in paths:
            if isinstance(path, dict):
                relations = path.get('relations', [])
                if relations:
                    # Find matching pattern in index
                    subgraph_id = index._create_subgraph_id([str(r) for r in relations])
                    
                    # Verify pattern exists in index
                    if subgraph_id in index.patterns:
                        training_samples.append({
                            'question': question,
                            'subgraph_id': subgraph_id,
                            'relations': [str(r) for r in relations]
                        })
    
    # Save training data
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path_obj.suffix == '.jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    elif output_path_obj.suffix == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_samples, f, ensure_ascii=False, indent=2)
    else:
        # Default to jsonl
        with open(output_path + '.jsonl', 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Prepared {len(training_samples)} training samples")
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare GSR training data')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--subgraph_index_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    args = parser.parse_args()
    
    prepare_gsr_training_data(
        data_path=args.data_path,
        subgraph_index_path=args.subgraph_index_path,
        output_path=args.output_path
    )

