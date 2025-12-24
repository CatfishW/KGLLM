"""
Qwen3-Reranker integration for high-accuracy reranking.
Implements batched inference with Flash Attention for speed.
"""
import torch
from torch import nn
from typing import List, Tuple, Optional
from tqdm import tqdm
import gc

from .config import RAGConfig


class Qwen3Reranker:
    """
    State-of-the-art reranker using Qwen3-Reranker-0.6B.
    
    Features:
    - Batched inference for throughput
    - Flash Attention 2 support for speed
    - Custom instruction prompts for KGQA task
    - GPU memory optimization
    """
    
    # Instruction prompt optimized for KG path retrieval
    DEFAULT_INSTRUCTION = (
        "Given a question about entities and their relationships in a knowledge graph, "
        "determine if the provided relation path could help answer the question. "
        "The path shows a sequence of relations connecting entities."
    )
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = config.device
        self.max_length = config.reranker_max_length
        self.batch_size = config.reranker_batch_size
        
        # Token IDs for yes/no
        self.token_true_id = None
        self.token_false_id = None
        
        # Prefix/suffix tokens
        self.prefix_tokens = None
        self.suffix_tokens = None
        
        self._loaded = False
    
    def load(self):
        """Load the model and tokenizer."""
        if self._loaded:
            return
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"Loading {self.config.reranker_model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.reranker_model_name,
            padding_side="left"
        )
        
        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.reranker_dtype, torch.float16)
        
        # Load model with optimizations
        model_kwargs = {
            "torch_dtype": dtype,
        }
        
        if self.config.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Using Flash Attention 2")
            except Exception as e:
                print(f"Flash Attention not available: {e}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.reranker_model_name,
            **model_kwargs
        ).to(self.device).eval()
        
        # Get token IDs
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        
        # Build prefix/suffix
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        
        self._loaded = True
        print(f"Reranker loaded on {self.device}")
    
    def _format_input(
        self, 
        query: str, 
        document: str, 
        instruction: Optional[str] = None
    ) -> str:
        """Format a query-document pair for the reranker."""
        if instruction is None:
            instruction = self.DEFAULT_INSTRUCTION
        
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    
    def _process_inputs(self, pairs: List[str]) -> dict:
        """Tokenize and prepare inputs for the model."""
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        # Add prefix and suffix tokens
        for i, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + input_ids + self.suffix_tokens
        
        # Pad
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        
        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        return inputs
    
    @torch.no_grad()
    def _compute_scores(self, inputs: dict) -> List[float]:
        """Compute relevance scores from model logits."""
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]  # Last token logits
        
        true_logits = logits[:, self.token_true_id]
        false_logits = logits[:, self.token_false_id]
        
        # Stack and softmax
        stacked = torch.stack([false_logits, true_logits], dim=1)
        probs = torch.nn.functional.softmax(stacked, dim=1)
        
        # Return probability of "yes"
        scores = probs[:, 1].cpu().tolist()
        return scores
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> Tuple[List[int], List[float], List[str]]:
        """
        Rerank documents for a query.
        
        Args:
            query: The input question
            documents: List of candidate documents (paths)
            top_k: Number of top results to return
            instruction: Custom instruction (optional)
        
        Returns:
            indices: Reranked indices (relative to input documents)
            scores: Relevance scores
            reranked_docs: Reranked documents
        """
        self.load()
        
        if top_k is None:
            top_k = self.config.top_k_rerank
        
        if not documents:
            return [], [], []
        
        # Format all pairs
        pairs = [self._format_input(query, doc, instruction) for doc in documents]
        
        # Process in batches
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            inputs = self._process_inputs(batch_pairs)
            batch_scores = self._compute_scores(inputs)
            all_scores.extend(batch_scores)
            
            # Clear CUDA cache periodically
            if i % (self.batch_size * 10) == 0:
                torch.cuda.empty_cache()
        
        # Sort by score
        scored_docs = list(enumerate(all_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k
        top_results = scored_docs[:top_k]
        indices = [idx for idx, _ in top_results]
        scores = [score for _, score in top_results]
        reranked_docs = [documents[idx] for idx in indices]
        
        return indices, scores, reranked_docs
    
    def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        top_k: Optional[int] = None,
        instruction: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[Tuple[List[int], List[float], List[str]]]:
        """
        Rerank documents for multiple queries.
        
        Args:
            queries: List of questions
            documents_list: List of candidate document lists (one per query)
            top_k: Number of top results per query
            instruction: Custom instruction
            show_progress: Show progress bar
        
        Returns:
            List of (indices, scores, reranked_docs) tuples
        """
        self.load()
        
        if top_k is None:
            top_k = self.config.top_k_rerank
        
        results = []
        iterator = zip(queries, documents_list)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Reranking")
        
        for query, documents in iterator:
            result = self.rerank(query, documents, top_k, instruction)
            results.append(result)
        
        return results
    
    def unload(self):
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        print("Reranker unloaded")
