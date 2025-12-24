"""
Diffusion Path Generator for the KGQA system.

Wrapper for the Core diffusion model to generate reasoning paths from questions.
"""
import os
import sys
import torch
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

# Add Core to path
CORE_PATH = Path(__file__).parent.parent / "Core"
sys.path.insert(0, str(CORE_PATH))

from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion generator."""
    checkpoint_path: str = "/data/Yanlai/KGLLM/Core/diffusion_100M/checkpoints/last.ckpt"
    vocab_path: str = "/data/Yanlai/KGLLM/Core/diffusion_100M/vocab.json"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_paths: int = 5
    path_length: int = 6  # Max relations in path
    temperature: float = 1.0
    use_predicted_hops: bool = True


class DiffusionPathGenerator:
    """
    Wrapper for generating reasoning paths using the trained diffusion model.
    
    Features:
    - Load trained checkpoint with vocab
    - Generate multiple diverse paths per question
    - Convert paths to string format for reranking
    - Hop count prediction for optimal path length
    """
    
    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.config = config or DiffusionConfig()
        self.model = None
        self.vocab = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self):
        """Load the diffusion model and vocabulary."""
        if self._loaded:
            return
        
        # Import Core modules
        from data.dataset import EntityRelationVocab
        from kg_path_diffusion import KGPathDiffusionLightning
        from transformers import AutoTokenizer
        
        print(f"[DiffusionGenerator] Loading vocabulary from {self.config.vocab_path}")
        self.vocab = EntityRelationVocab.load(self.config.vocab_path)
        
        print(f"[DiffusionGenerator] Loading model from {self.config.checkpoint_path}")
        self.model = KGPathDiffusionLightning.load_from_checkpoint(
            self.config.checkpoint_path,
            num_entities=self.vocab.num_entities,
            num_relations=self.vocab.num_relations,
            map_location=self.config.device,
            strict=False
        )
        self.model.eval()
        self.model.to(self.config.device)
        
        # Load tokenizer for question encoding
        tokenizer_name = self.model.hparams.get(
            'question_encoder_name', 
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        print(f"[DiffusionGenerator] Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self._loaded = True
        print(f"[DiffusionGenerator] Model loaded on {self.config.device}")
        print(f"[DiffusionGenerator] Vocabulary: {self.vocab.num_entities} entities, {self.vocab.num_relations} relations")
    
    def _encode_question(self, question: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and encode a question."""
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        return (
            inputs["input_ids"].to(self.config.device),
            inputs["attention_mask"].to(self.config.device)
        )
    
    def _decode_path(
        self, 
        entities: torch.Tensor, 
        relations: torch.Tensor
    ) -> Dict[str, Any]:
        """Decode generated indices back to entity/relation names."""
        entity_names = []
        relation_names = []
        
        # Decode entities (skip if all zeros - relation-only mode)
        for idx in entities.tolist():
            if idx == 0:
                continue
            name = self.vocab.idx2entity.get(idx, "<UNK>")
            if name not in ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]:
                entity_names.append(name)
        
        # Decode relations
        for idx in relations.tolist():
            name = self.vocab.idx2relation.get(idx, "<UNK>")
            if name not in ["<PAD>", "<MASK>", "<UNK>"]:
                relation_names.append(name)
        
        # Build path string
        if not entity_names:
            path_string = " -> ".join(relation_names) if relation_names else ""
        else:
            path_parts = []
            for i, entity in enumerate(entity_names):
                path_parts.append(f"({entity})")
                if i < len(relation_names):
                    path_parts.append(f" --[{relation_names[i]}]--> ")
            path_string = "".join(path_parts)
        
        return {
            "entities": entity_names,
            "relations": relation_names,
            "path_string": path_string,
            "relation_chain": " -> ".join(relation_names) if relation_names else ""
        }
    
    @torch.no_grad()
    def generate(
        self,
        question: str,
        num_paths: Optional[int] = None,
        path_length: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[str]:
        """
        Generate reasoning paths for a question.
        
        Args:
            question: The input question
            num_paths: Number of paths to generate (default from config)
            path_length: Max path length in relations (default from config)
            temperature: Sampling temperature (default from config)
            
        Returns:
            List of relation chain strings (e.g., ["rel1 -> rel2", "rel3 -> rel4 -> rel5"])
        """
        self.load()
        
        num_paths = num_paths or self.config.num_paths
        path_length = path_length or self.config.path_length
        temperature = temperature or self.config.temperature
        
        # Encode question
        input_ids, attention_mask = self._encode_question(question)
        
        # Generate multiple paths
        all_entities, all_relations = self.model.model.generate_multiple(
            input_ids,
            attention_mask,
            num_paths=num_paths,
            path_length=path_length + 1,  # +1 for entities
            temperature=temperature
        )
        
        # Decode paths
        paths = []
        for path_idx in range(num_paths):
            decoded = self._decode_path(
                all_entities[0, path_idx],
                all_relations[0, path_idx]
            )
            if decoded["relation_chain"]:
                paths.append(decoded["relation_chain"])
        
        return paths
    
    @torch.no_grad()
    def generate_batch(
        self,
        questions: List[str],
        num_paths: Optional[int] = None,
        path_length: Optional[int] = None,
        temperature: Optional[float] = None,
        show_progress: bool = True
    ) -> List[List[str]]:
        """
        Generate reasoning paths for multiple questions.
        
        Args:
            questions: List of input questions
            num_paths: Number of paths per question
            path_length: Max path length
            temperature: Sampling temperature
            show_progress: Show progress bar
            
        Returns:
            List of lists of relation chain strings
        """
        self.load()
        
        from tqdm import tqdm
        
        num_paths = num_paths or self.config.num_paths
        path_length = path_length or self.config.path_length
        temperature = temperature or self.config.temperature
        
        all_results = []
        iterator = tqdm(questions, desc="Generating paths") if show_progress else questions
        
        for question in iterator:
            paths = self.generate(question, num_paths, path_length, temperature)
            all_results.append(paths)
        
        return all_results
    
    @torch.no_grad()
    def generate_with_details(
        self,
        question: str,
        num_paths: Optional[int] = None,
        path_length: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate paths with full details (entities and relations).
        
        Returns:
            List of dicts with 'entities', 'relations', 'path_string', 'relation_chain'
        """
        self.load()
        
        num_paths = num_paths or self.config.num_paths
        path_length = path_length or self.config.path_length
        temperature = temperature or self.config.temperature
        
        # Encode question
        input_ids, attention_mask = self._encode_question(question)
        
        # Generate multiple paths
        all_entities, all_relations = self.model.model.generate_multiple(
            input_ids,
            attention_mask,
            num_paths=num_paths,
            path_length=path_length + 1,
            temperature=temperature
        )
        
        # Decode paths
        results = []
        for path_idx in range(num_paths):
            decoded = self._decode_path(
                all_entities[0, path_idx],
                all_relations[0, path_idx]
            )
            results.append(decoded)
        
        return results
    
    def unload(self):
        """Unload model to free GPU memory."""
        import gc
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        print("[DiffusionGenerator] Model unloaded")


# Singleton instance for convenience
_default_generator: Optional[DiffusionPathGenerator] = None


def get_diffusion_generator(config: Optional[DiffusionConfig] = None) -> DiffusionPathGenerator:
    """Get or create default diffusion generator."""
    global _default_generator
    if _default_generator is None or config is not None:
        _default_generator = DiffusionPathGenerator(config)
    return _default_generator


if __name__ == "__main__":
    # Quick test
    generator = DiffusionPathGenerator()
    
    question = "What language is spoken in Jamaica?"
    print(f"Question: {question}")
    
    paths = generator.generate(question, num_paths=5)
    print(f"Generated {len(paths)} paths:")
    for i, path in enumerate(paths, 1):
        print(f"  {i}. {path}")
    
    # Test with details
    print("\nWith details:")
    details = generator.generate_with_details(question, num_paths=3)
    for d in details:
        print(f"  Relations: {d['relations']}")
        print(f"  Chain: {d['relation_chain']}")
