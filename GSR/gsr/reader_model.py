"""
Reader Model for GSR: Generates answers from question + retrieved subgraph.

This is a simplified reader that can be extended with more sophisticated LLMs.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import json


class SimpleReader(nn.Module):
    """
    Simple reader model that extracts answers from retrieved subgraphs.
    
    For production use, this should be replaced with a fine-tuned LLM
    (e.g., using unsloth/QLoRA as mentioned in GSR paper).
    """
    
    def __init__(self):
        super().__init__()
        # In a real implementation, this would be a fine-tuned LLM
        # For now, this is a placeholder that extracts answer entities
    
    def forward(
        self,
        question: str,
        subgraph_triples: List[tuple],
        answer_entities: List[str] = None
    ) -> Dict:
        """
        Generate answer from question and retrieved subgraph.
        
        Args:
            question: Question text
            subgraph_triples: List of (subject, relation, object) triples
            answer_entities: Optional list of answer entity IDs
        
        Returns:
            Dictionary with answer and metadata
        """
        # Simple extraction: find entities connected to question entities
        # In production, this would use a fine-tuned LLM
        
        if answer_entities:
            # If answer entities are provided, return them
            return {
                'answer_entities': answer_entities,
                'answer_text': ', '.join(answer_entities),
                'confidence': 1.0
            }
        
        # Otherwise, extract from triples
        # This is a simplified version - real implementation would use LLM
        if subgraph_triples:
            # Get object entities from triples
            answer_entities = [triple[2] for triple in subgraph_triples]
            return {
                'answer_entities': answer_entities,
                'answer_text': ', '.join(answer_entities),
                'confidence': 0.8
            }
        
        return {
            'answer_entities': [],
            'answer_text': '',
            'confidence': 0.0
        }


def prepare_reader_data(
    gsr_predictions_path: str,
    subgraph_index_path: str,
    original_data_path: str,
    output_path: str
):
    """
    Prepare training data for reader model.
    
    Format: (question, retrieved_subgraph, answer)
    
    Args:
        gsr_predictions_path: Path to GSR predictions (subgraph IDs)
        subgraph_index_path: Path to subgraph index
        original_data_path: Path to original dataset with answers
        output_path: Path to save reader training data
    """
    from pathlib import Path
    from gsr.subgraph_index import SubgraphIndex
    import pandas as pd
    
    # Load subgraph index
    index = SubgraphIndex.load(subgraph_index_path)
    
    # Load GSR predictions
    predictions = []
    with open(gsr_predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    # Load original data
    path = Path(original_data_path)
    original_data = {}
    
    if path.suffix == '.parquet':
        df = pd.read_parquet(original_data_path)
        for _, row in df.iterrows():
            original_data[row['id']] = row.to_dict()
    elif path.suffix in ['.jsonl', '.json']:
        with open(original_data_path, 'r', encoding='utf-8') as f:
            if path.suffix == '.jsonl':
                for line in f:
                    data = json.loads(line)
                    original_data[data.get('id', '')] = data
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        original_data[item.get('id', '')] = item
                else:
                    original_data[data.get('id', '')] = data
    
    # Create reader training samples
    reader_samples = []
    
    for pred in predictions:
        question_id = pred.get('id', '')
        question = pred.get('question', '')
        top_prediction = pred.get('predictions', [{}])[0] if pred.get('predictions') else {}
        
        subgraph_id = top_prediction.get('subgraph_id', '')
        pattern = index.get_pattern(subgraph_id)
        
        if pattern and question_id in original_data:
            original = original_data[question_id]
            answer = original.get('answer', original.get('a_entity', []))
            
            reader_samples.append({
                'question': question,
                'subgraph_id': subgraph_id,
                'subgraph_triples': pattern.example_triples,
                'relations': pattern.relations,
                'answer': answer
            })
    
    # Save reader data
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path_obj.suffix == '.jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in reader_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reader_samples, f, ensure_ascii=False, indent=2)
    
    print(f"Prepared {len(reader_samples)} reader training samples")
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare reader training data')
    parser.add_argument('--gsr_predictions_path', type=str, required=True)
    parser.add_argument('--subgraph_index_path', type=str, required=True)
    parser.add_argument('--original_data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    args = parser.parse_args()
    
    prepare_reader_data(
        gsr_predictions_path=args.gsr_predictions_path,
        subgraph_index_path=args.subgraph_index_path,
        original_data_path=args.original_data_path,
        output_path=args.output_path
    )

