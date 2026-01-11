"""
Question Type Classifier Model

Fast 3-class classifier using TinyBERT for KGQA question type classification:
- one_hop: Single relation path questions
- multi_hop: Multi-relation path questions  
- numeric: Count/quantity questions

Optimized for fast inference with ONNX export support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import pytorch_lightning as pl
except ImportError:
    class _LightningModule:
        def save_hyperparameters(self, *args, **kwargs):
            pass
        def log(self, *args, **kwargs):
            pass
    class _PL:
        LightningModule = _LightningModule
    pl = _PL()

from data.question_classifier_dataset import LABEL_NAMES, LABEL_MAP, NUM_CLASSES


class QuestionTypeClassifier(pl.LightningModule):
    """
    Fast question type classifier using TinyBERT.
    
    Classifies questions into: one_hop, multi_hop, numeric
    """
    
    def __init__(
        self,
        encoder_name: str = 'huawei-noah/TinyBERT_General_6L_768D',
        num_classes: int = 3,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        class_weights: Optional[torch.Tensor] = None,
        freeze_encoder_layers: int = 0,  # Number of encoder layers to freeze
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Load encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim = self.encoder.config.hidden_size
        
        # Optionally freeze early layers for faster training
        if freeze_encoder_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layer[:freeze_encoder_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Class weights for imbalanced data
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        # Metrics tracking
        self.train_correct = 0
        self.train_total = 0
        self.val_correct = 0
        self.val_total = 0
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [B, L] token ids
            attention_mask: [B, L] attention mask
            
        Returns:
            logits: [B, num_classes]
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, D]
        
        # Classify
        logits = self.classifier(cls_output)  # [B, num_classes]
        
        return logits
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss with optional class weights."""
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        logits = self(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)
        
        # Compute accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        self.train_correct += correct
        self.train_total += labels.size(0)
        
        # Log
        self.log('train_loss', loss, prog_bar=True)
        
        if batch_idx % 100 == 0 and self.train_total > 0:
            acc = self.train_correct / self.train_total
            self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        if self.train_total > 0:
            acc = self.train_correct / self.train_total
            self.log('train_epoch_acc', acc)
        self.train_correct = 0
        self.train_total = 0
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        logits = self(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)
        
        # Compute accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        self.val_correct += correct
        self.val_total += labels.size(0)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        if self.val_total > 0:
            acc = self.val_correct / self.val_total
            self.log('val_acc', acc, prog_bar=True)
            print(f"\nValidation Accuracy: {acc:.4f} ({self.val_correct}/{self.val_total})")
        self.val_correct = 0
        self.val_total = 0
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        logits = self(input_ids, attention_mask)
        preds = logits.argmax(dim=-1)
        
        return {'preds': preds, 'labels': labels}
    
    def predict(self, question: str, tokenizer: AutoTokenizer) -> Tuple[str, float]:
        """
        Predict class for a single question.
        
        Returns:
            (class_name, confidence)
        """
        self.eval()
        with torch.no_grad():
            encoding = tokenizer(
                question,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            device = next(self.parameters()).device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            logits = self(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1).item()
            conf = probs[0, pred].item()
            
            return LABEL_NAMES[pred], conf
    
    def predict_batch(
        self,
        questions: List[str],
        tokenizer: AutoTokenizer,
        batch_size: int = 32,
    ) -> List[Tuple[str, float]]:
        """Batch prediction for efficiency."""
        self.eval()
        results = []
        
        with torch.no_grad():
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i+batch_size]
                encoding = tokenizer(
                    batch_questions,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                device = next(self.parameters()).device
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                logits = self(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)
                
                for j in range(len(batch_questions)):
                    pred = preds[j].item()
                    conf = probs[j, pred].item()
                    results.append((LABEL_NAMES[pred], conf))
        
        return results
    
    def configure_optimizers(self):
        """Configure optimizer with warmup and linear decay."""
        # Separate parameters for different learning rates
        encoder_params = list(self.encoder.parameters())
        classifier_params = list(self.classifier.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self.learning_rate},
            {'params': classifier_params, 'lr': self.learning_rate * 5},  # Higher LR for head
        ], weight_decay=self.weight_decay)
        
        # Linear warmup then linear decay
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            else:
                progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
                return max(0.0, 1.0 - progress)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
    
    def export_onnx(self, save_path: str, tokenizer: AutoTokenizer):
        """
        Export model to ONNX format for fast inference.
        
        Args:
            save_path: Path to save ONNX model
            tokenizer: Tokenizer for dummy input
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Create dummy input
        dummy_text = "What country is Paris in?"
        encoding = tokenizer(
            dummy_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        dummy_input_ids = encoding['input_ids'].to(device)
        dummy_attention_mask = encoding['attention_mask'].to(device)
        
        # Export
        torch.onnx.export(
            self,
            (dummy_input_ids, dummy_attention_mask),
            save_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
            },
            opset_version=14,
        )
        print(f"Exported ONNX model to {save_path}")


if __name__ == '__main__':
    # Quick test
    print("Testing QuestionTypeClassifier...")
    
    model = QuestionTypeClassifier(
        encoder_name='huawei-noah/TinyBERT_General_6L_768D',
    )
    
    tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_6L_768D')
    
    # Test prediction
    test_questions = [
        "What country is Paris located in?",
        "What country sharing borders with Spain does the city belong to?",
        "How many people live in New York?",
    ]
    
    for q in test_questions:
        label, conf = model.predict(q, tokenizer)
        print(f"  Q: {q[:50]}... -> {label} ({conf:.3f})")
    
    print("\nModel test passed!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
