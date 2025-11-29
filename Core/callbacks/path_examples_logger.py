"""
Callback to log structured examples of predicted and ground truth relation paths after each epoch.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class PathExamplesLogger(Callback):
    """
    Logs structured examples of predicted and ground truth relation paths after each epoch.
    
    For each example, logs:
    - Question ID
    - Question text
    - Ground truth relation paths (decoded from indices)
    - Predicted relation paths (decoded from indices)
    """
    
    def __init__(
        self,
        num_examples: int = 10,
        log_dir: Optional[str] = None,
        log_to_file: bool = True,
        log_to_tensorboard: bool = True,
        log_every_n_epochs: int = 1
    ):
        """
        Args:
            num_examples: Number of examples to log per epoch
            log_dir: Directory to save log files (defaults to trainer's log_dir)
            log_to_file: Whether to save examples to JSON file
            log_to_tensorboard: Whether to log examples to TensorBoard
            log_every_n_epochs: Log examples every N epochs (default: 1, every epoch)
        """
        super().__init__()
        self.num_examples = num_examples
        self.log_dir = log_dir
        self.log_to_file = log_to_file
        self.log_to_tensorboard = log_to_tensorboard
        self.log_every_n_epochs = log_every_n_epochs
        
        self.current_epoch = 0
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate and log examples after validation epoch ends."""
        self.current_epoch = trainer.current_epoch
        
        # Skip if not logging this epoch
        if self.current_epoch % self.log_every_n_epochs != 0:
            return
        
        # Get vocabulary from datamodule
        if not hasattr(trainer, 'datamodule') or trainer.datamodule is None:
            return
        if not hasattr(trainer.datamodule, 'vocab') or trainer.datamodule.vocab is None:
            return
        
        vocab = trainer.datamodule.vocab
        
        # Get validation dataloader
        val_dataloader = trainer.datamodule.val_dataloader()
        if val_dataloader is None:
            return
        
        # Collect a small subset of samples (only what we need)
        examples = []
        samples_collected = 0
        
        # Iterate through validation dataloader to collect samples
        with torch.no_grad():
            pl_module.eval()
            for batch_idx, batch in enumerate(val_dataloader):
                if samples_collected >= self.num_examples:
                    break
                
                # Move batch to device
                device = next(pl_module.parameters()).device
                question_input_ids = batch['question_input_ids'].to(device)
                question_attention_mask = batch['question_attention_mask'].to(device)
                
                batch_size = question_input_ids.shape[0]
                num_needed = min(batch_size, self.num_examples - samples_collected)
                
                # Only process the samples we need
                question_input_ids_subset = question_input_ids[:num_needed]
                question_attention_mask_subset = question_attention_mask[:num_needed]
                
                # Generate predictions for this subset only
                try:
                    num_paths_to_generate = 3
                    pred_entities, pred_relations = pl_module.model.generate_multiple(
                        question_input_ids=question_input_ids_subset,
                        question_attention_mask=question_attention_mask_subset,
                        num_paths=num_paths_to_generate,
                        path_length=pl_module.model.max_path_length,
                        temperature=1.0
                    )
                except Exception as e:
                    print(f"Warning: Failed to generate predictions in callback: {e}")
                    continue
                
                # Process each sample
                for i in range(num_needed):
                    if samples_collected >= self.num_examples:
                        break
                    
                    sample_id = batch.get('ids', [None] * batch_size)[i] if 'ids' in batch else f"val_batch_{batch_idx}_sample_{i}"
                    
                    # Decode ground truth paths
                    try:
                        if 'all_path_relations' in batch and 'all_path_lengths' in batch and 'num_paths' in batch:
                            # Move to CPU for decoding
                            gt_relations = batch['all_path_relations'][i].cpu()
                            gt_lengths = batch['all_path_lengths'][i].cpu()
                            gt_num_paths = batch['num_paths'][i].cpu()
                            gt_paths = self._decode_paths(
                                gt_relations,
                                gt_lengths,
                                gt_num_paths,
                                vocab
                            )
                        elif 'path_relations' in batch:
                            # Fallback to single path format
                            single_path = batch['path_relations'][i].cpu().unsqueeze(0)
                            path_length = batch.get('path_lengths', [single_path.shape[1]])[i] if 'path_lengths' in batch else single_path.shape[1]
                            gt_paths = self._decode_paths(
                                single_path,
                                torch.tensor([path_length]),
                                torch.tensor(1),
                                vocab
                            )
                        else:
                            gt_paths = []
                    except Exception as e:
                        print(f"Warning: Failed to decode ground truth paths: {e}")
                        gt_paths = []
                    
                    # Decode predicted paths
                    try:
                        pred_relations_cpu = pred_relations[i].cpu()
                        pred_paths = self._decode_paths(
                            pred_relations_cpu,
                            None,
                            num_paths_to_generate,
                            vocab
                        )
                    except Exception as e:
                        print(f"Warning: Failed to decode predicted paths: {e}")
                        pred_paths = []
                    
                    # Get question text
                    try:
                        question_text = self._decode_question(
                            question_input_ids_subset[i].cpu(),
                            question_attention_mask_subset[i].cpu(),
                            trainer.datamodule
                        )
                    except Exception as e:
                        question_text = f"Question (decode error: {e})"
                    
                    example = {
                        'id': sample_id,
                        'question': question_text,
                        'ground_truth_paths': gt_paths,
                        'predicted_paths': pred_paths
                    }
                    
                    examples.append(example)
                    samples_collected += 1
        
        # Log the examples
        if examples:
            self._log_examples(trainer, examples)
    
    def _log_examples(self, trainer: pl.Trainer, examples: List[Dict[str, Any]]):
        """Log examples to file and TensorBoard."""
        # Determine log directory
        log_dir = self.log_dir
        if log_dir is None:
            if hasattr(trainer.logger, 'log_dir'):
                log_dir = trainer.logger.log_dir
            elif hasattr(trainer.logger, 'save_dir'):
                log_dir = os.path.join(trainer.logger.save_dir, trainer.logger.name)
            else:
                log_dir = trainer.default_root_dir
        
        # Create examples directory
        examples_dir = Path(log_dir) / 'examples'
        examples_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON file
        if self.log_to_file:
            epoch_file = examples_dir / f'epoch_{self.current_epoch:03d}_examples.json'
            with open(epoch_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'epoch': self.current_epoch,
                    'num_examples': len(examples),
                    'examples': examples
                }, f, indent=2, ensure_ascii=False)
            
            # Also save latest examples
            latest_file = examples_dir / 'latest_examples.json'
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'epoch': self.current_epoch,
                    'num_examples': len(examples),
                    'examples': examples
                }, f, indent=2, ensure_ascii=False)
        
        # Log to TensorBoard
        if self.log_to_tensorboard and hasattr(trainer.logger, 'experiment'):
            # Format examples as text for TensorBoard
            for idx, example in enumerate(examples[:5]):  # Log first 5 to TensorBoard
                example_text = self._format_example_for_tensorboard(example)
                trainer.logger.experiment.add_text(
                    f'examples/epoch_{self.current_epoch}/example_{idx}',
                    example_text,
                    global_step=trainer.global_step
                )
    
    def _decode_paths(
        self,
        relations_tensor: torch.Tensor,
        path_lengths: Optional[torch.Tensor],
        num_paths: torch.Tensor,
        vocab: Any
    ) -> List[List[str]]:
        """
        Decode relation indices to relation strings.
        
        Args:
            relations_tensor: [max_paths, max_rel_len] tensor of relation indices
            path_lengths: [max_paths] tensor of actual path lengths (optional)
            num_paths: Scalar tensor indicating number of valid paths
            vocab: Vocabulary object with idx2relation mapping
        
        Returns:
            List of paths, where each path is a list of relation strings
        """
        paths = []
        max_paths = relations_tensor.shape[0]
        num_valid_paths = int(num_paths.item()) if isinstance(num_paths, torch.Tensor) else num_paths
        
        pad_idx = vocab.relation2idx.get("<PAD>", 0)
        unk_idx = vocab.relation2idx.get("<UNK>", 1)
        mask_idx = vocab.relation2idx.get("<MASK>", 2)
        
        for path_idx in range(min(num_valid_paths, max_paths)):
            path_relations = relations_tensor[path_idx]
            
            # Determine actual length
            if path_lengths is not None:
                # For ground truth: path_lengths includes BOS/EOS, so relations = path_length - 3
                # But we have relation tensor, so we need to find where padding starts
                actual_length = int(path_lengths[path_idx].item())
                # Relations are between entities, so if path has L entities (including BOS/EOS),
                # we have L-3 relations (excluding BOS, first entity, and EOS)
                rel_length = max(0, actual_length - 3)
                # Clamp to actual tensor size
                rel_length = min(rel_length, path_relations.shape[0])
            else:
                # For predictions: find first padding token
                rel_length = 0
                for j in range(path_relations.shape[0]):
                    rel_id = path_relations[j].item()
                    if rel_id == pad_idx:
                        break
                    rel_length = j + 1
            
            # Decode relations
            decoded_path = []
            for rel_idx in range(rel_length):
                rel_id = path_relations[rel_idx].item()
                # Skip padding, UNK, and MASK tokens
                if rel_id not in (pad_idx, unk_idx, mask_idx) and rel_id >= 0:
                    rel_str = vocab.idx2relation.get(rel_id, f"<UNK_{rel_id}>")
                    if rel_str not in ("<PAD>", "<UNK>", "<MASK>"):
                        decoded_path.append(rel_str)
            
            # Always add path, even if empty (to show structure)
            paths.append(decoded_path)
        
        return paths
    
    def _decode_question(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        datamodule: Any
    ) -> str:
        """Decode question from token IDs."""
        if hasattr(datamodule, 'train_dataset') and hasattr(datamodule.train_dataset, 'tokenizer'):
            tokenizer = datamodule.train_dataset.tokenizer
            # Get actual length from attention mask
            actual_length = attention_mask.sum().item()
            question_ids = input_ids[:actual_length].cpu().tolist()
            question_text = tokenizer.decode(question_ids, skip_special_tokens=True)
            return question_text
        else:
            return f"Question (token IDs: {input_ids[:attention_mask.sum()].cpu().tolist()})"
    
    def _format_example_for_tensorboard(self, example: Dict[str, Any]) -> str:
        """Format example as text for TensorBoard."""
        lines = [
            f"ID: {example['id']}",
            f"Question: {example['question']}",
            "",
            "Ground Truth Paths:"
        ]
        
        for i, path in enumerate(example['ground_truth_paths']):
            path_str = " -> ".join(path) if path else "[empty]"
            lines.append(f"  Path {i+1}: {path_str}")
        
        lines.append("")
        lines.append("Predicted Paths:")
        
        for i, path in enumerate(example['predicted_paths']):
            path_str = " -> ".join(path) if path else "[empty]"
            lines.append(f"  Path {i+1}: {path_str}")
        
        return "\n".join(lines)

