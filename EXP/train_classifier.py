"""
Train Question Classifier: Train a neural classifier for question types.

This script:
1. Extracts training data from WebQSP parquet files
2. Infers question types from ground truth paths
3. Trains a sentence-transformer based classifier
4. Saves the trained model for use in the RAG pipeline
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Question types
QUESTION_TYPES = ['literal', 'definitional', 'one_hop', 'two_hop']
TYPE_TO_IDX = {t: i for i, t in enumerate(QUESTION_TYPES)}
IDX_TO_TYPE = {i: t for i, t in enumerate(QUESTION_TYPES)}


def infer_question_type(question: str, paths: List[Dict]) -> str:
    """
    Infer question type from ground truth paths.
    
    Uses both question patterns and path characteristics.
    """
    q_lower = question.lower()
    
    # Get path statistics
    if paths:
        lengths = [len(p.get('relations', [])) for p in paths if p.get('relations')]
        avg_len = sum(lengths) / len(lengths) if lengths else 1
        max_len = max(lengths) if lengths else 1
    else:
        avg_len = 1
        max_len = 1
    
    # DEFINITIONAL patterns (type/category questions)
    definitional_patterns = [
        'what is a ', 'what are ', 'what is the meaning',
        'what kind of', 'what type of', 'define ',
        'what does ', 'what do '
    ]
    if any(q_lower.startswith(p) or p in q_lower for p in definitional_patterns):
        # Check if path suggests single-hop definitional
        if max_len <= 1:
            return 'definitional'
    
    # LITERAL patterns (direct attribute lookup)
    literal_patterns = [
        'when was', 'when did', 'when is',
        'what year', 'what date', 'what day',
        'how old', 'how many', 'how much', 'how long',
        'how tall', 'how big',
        'birthday', 'birth date', 'born',
        'population', 'number of', 'count',
        'age of', 'height of', 'weight of'
    ]
    if any(p in q_lower for p in literal_patterns):
        return 'literal'
    
    # Use path length as primary indicator for hop classification
    if max_len >= 3:
        return 'two_hop'
    elif avg_len >= 2:
        return 'two_hop'
    else:
        return 'one_hop'


def extract_training_data(parquet_path: Path) -> List[Tuple[str, str]]:
    """
    Extract (question, label) pairs from parquet file.
    """
    df = pd.read_parquet(parquet_path)
    
    data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {parquet_path.name}"):
        question = row['question']
        paths_str = row['paths']
        
        # Parse paths
        if isinstance(paths_str, str):
            try:
                paths = json.loads(paths_str)
            except:
                paths = []
        else:
            paths = paths_str if paths_str else []
        
        # Infer type
        qtype = infer_question_type(question, paths)
        data.append((question, qtype))
    
    return data


class QuestionDataset(Dataset):
    """Dataset for question classification."""
    
    def __init__(self, questions: List[str], labels: List[int], embeddings: np.ndarray):
        self.questions = questions
        self.labels = labels
        self.embeddings = embeddings
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            'embedding': torch.tensor(self.embeddings[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class QuestionClassifierModel(nn.Module):
    """Neural classifier for question types."""
    
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


def train_classifier(
    data_dir: str = "Data/webqsp_final",
    output_dir: str = "EXP/models/classifier",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_dim: int = 256,
    dropout: float = 0.3
):
    """Train the question classifier."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract training data
    print("\n" + "="*60)
    print("Step 1: Extracting training data")
    print("="*60)
    
    train_data = extract_training_data(Path(data_dir) / "train.parquet")
    val_data = extract_training_data(Path(data_dir) / "val.parquet")
    
    # Combine and analyze
    all_data = train_data + val_data
    questions = [q for q, _ in all_data]
    labels = [TYPE_TO_IDX[l] for _, l in all_data]
    
    # Print distribution
    label_counts = Counter([l for _, l in all_data])
    print(f"\nTotal samples: {len(all_data)}")
    print("Label distribution:")
    for qtype, count in sorted(label_counts.items()):
        print(f"  {qtype}: {count} ({count/len(all_data)*100:.1f}%)")
    
    # Step 2: Create embeddings
    print("\n" + "="*60)
    print("Step 2: Creating embeddings")
    print("="*60)
    
    encoder = SentenceTransformer(embedding_model)
    print(f"Using model: {embedding_model}")
    
    embeddings = encoder.encode(
        questions,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    # Step 3: Split data
    print("\n" + "="*60)
    print("Step 3: Splitting data")
    print("="*60)
    
    X_train, X_test, y_train, y_test, q_train, q_test = train_test_split(
        embeddings, labels, questions,
        test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Create datasets
    train_dataset = QuestionDataset(q_train, y_train, X_train)
    test_dataset = QuestionDataset(q_test, y_test, X_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Step 4: Train model
    print("\n" + "="*60)
    print("Step 4: Training model")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = QuestionClassifierModel(
        input_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        num_classes=len(QUESTION_TYPES),
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Class weights for imbalanced data
    class_counts = [sum(1 for l in y_train if l == i) for i in range(len(QUESTION_TYPES))]
    class_weights = torch.tensor([len(y_train) / (len(QUESTION_TYPES) * c + 1) for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    best_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            embeddings_batch = batch['embedding'].to(device)
            labels_batch = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels_batch).sum().item()
            total += labels_batch.size(0)
        
        train_acc = correct / total
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                embeddings_batch = batch['embedding'].to(device)
                labels_batch = batch['label'].to(device)
                
                outputs = model(embeddings_batch)
                pred = outputs.argmax(dim=1)
                test_correct += (pred == labels_batch).sum().item()
                test_total += labels_batch.size(0)
        
        test_acc = test_correct / test_total
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Step 5: Final evaluation
    print("\n" + "="*60)
    print("Step 5: Final evaluation")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            embeddings_batch = batch['embedding'].to(device)
            labels_batch = batch['label'].to(device)
            
            outputs = model(embeddings_batch)
            pred = outputs.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=QUESTION_TYPES))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Step 6: Save model
    print("\n" + "="*60)
    print("Step 6: Saving model")
    print("="*60)
    
    # Save PyTorch model
    torch.save(model.state_dict(), output_path / "classifier.pt")
    
    # Save model config
    config = {
        'input_dim': embeddings.shape[1],
        'hidden_dim': hidden_dim,
        'num_classes': len(QUESTION_TYPES),
        'dropout': dropout,
        'embedding_model': embedding_model,
        'type_to_idx': TYPE_TO_IDX,
        'idx_to_type': IDX_TO_TYPE,
        'best_accuracy': best_acc
    }
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved to {output_path}")
    print(f"Best test accuracy: {best_acc:.4f}")
    
    return model, best_acc


def main():
    parser = argparse.ArgumentParser(description="Train Question Classifier")
    parser.add_argument("--data_dir", type=str, default="Data/webqsp_final",
                       help="Directory containing parquet files")
    parser.add_argument("--output_dir", type=str, default="EXP/models/classifier",
                       help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    
    args = parser.parse_args()
    
    train_classifier(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )


if __name__ == "__main__":
    main()
