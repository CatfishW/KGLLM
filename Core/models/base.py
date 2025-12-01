"""
Base components shared across all models.

Contains:
- QuestionEncoder: Shared question encoding component
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Tuple


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

