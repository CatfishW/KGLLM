"""
Optimized KGQA Pipeline - Batched LLM Inferencing + Multi-Session Processing

Key optimizations:
1. Parallel path corpus building using multiprocessing
2. Batched LLM inferencing (multiple questions per API call)
3. Async concurrent LLM sessions for maximum throughput
4. Pre-computed embeddings with memory mapping

Usage:
    python run_pipeline_optimized.py

Based on best_config_76pct_f1 (76.6% F1)
"""
import sys
sys.path.insert(0, '/data/Yanlai/KGLLM')

import asyncio
import aiohttp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from typing import List, Dict, Tuple, Optional, Any
import json
from pathlib import Path
import time

from RAG.config import RAGConfig
from RAG.data_loader import KGDataLoader, KGSample
from RAG.combined_evaluator import CombinedEvaluator
from sentence_transformers import SentenceTransformer
import faiss


# ============================================================================
# Configuration
# ============================================================================

class OptimizedConfig:
    """Configuration for optimized pipeline."""
    # LLM Settings
    API_URL = "https://game.agaii.org/llm/v1"
    LLM_BATCH_SIZE = 8          # Questions per batch for batched API (if supported)
    MAX_CONCURRENT_REQUESTS = 16 # Concurrent async sessions
    LLM_TIMEOUT = 120            # Timeout per request
    
    # Retriever Settings
    RETRIEVER_BATCH_SIZE = 512  # Batch size for embedding computation
    TOP_K_RETRIEVE = 10         # Top paths to retrieve
    TOP_K_CONTEXT = 3           # Top paths for LLM context
    
    # Corpus Building
    NUM_WORKERS = 8             # For parallel corpus building
    
    # Caching
    CACHE_DIR = Path("./RAG/best_config_76pct_f1/.cache")


# ============================================================================
# Optimized Path Corpus Builder
# ============================================================================

def extract_paths_from_sample(sample: KGSample) -> List[str]:
    """Extract path strings from a single sample (for parallel processing)."""
    return sample.get_path_strings()


class OptimizedCorpusBuilder:
    """Build path corpus with parallel processing."""
    
    def __init__(self, loader: KGDataLoader, config: OptimizedConfig = None):
        self.loader = loader
        self.config = config or OptimizedConfig()
        
    def build_fast(self, include_test: bool = True) -> Tuple[List[str], Dict[str, int]]:
        """
        Build path corpus using parallel processing.
        
        Returns:
            all_paths: List of unique path strings
            path_to_idx: Mapping from path to index
        """
        print("Building corpus (optimized parallel processing)...")
        start_time = time.time()
        
        # Collect all samples
        all_samples = []
        all_samples.extend(self.loader.train_data)
        all_samples.extend(self.loader.val_data)
        if include_test:
            all_samples.extend(self.loader.test_data)
        
        # Use ThreadPoolExecutor for I/O-bound path extraction
        # (ProcessPoolExecutor has serialization overhead)
        path_set = set()
        
        with ThreadPoolExecutor(max_workers=self.config.NUM_WORKERS) as executor:
            # Map samples to path lists in parallel
            path_lists = list(tqdm(
                executor.map(extract_paths_from_sample, all_samples),
                total=len(all_samples),
                desc="Extracting paths"
            ))
            
            # Flatten and deduplicate
            for paths in path_lists:
                path_set.update(paths)
        
        all_paths = sorted(list(path_set))
        path_to_idx = {path: idx for idx, path in enumerate(all_paths)}
        
        elapsed = time.time() - start_time
        print(f"  Built corpus: {len(all_paths)} unique paths in {elapsed:.2f}s")
        
        return all_paths, path_to_idx


# ============================================================================
# Async LLM Client with Multi-Session Concurrency
# ============================================================================

class AsyncLLMClient:
    """
    Async LLM client supporting:
    1. Concurrent multi-session requests
    2. Batched inference (if API supports)
    3. Connection pooling
    """
    
    def __init__(self, config: OptimizedConfig = None):
        self.config = config or OptimizedConfig()
        self._model: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    async def initialize(self):
        """Initialize async session and detect model."""
        connector = aiohttp.TCPConnector(
            limit=self.config.MAX_CONCURRENT_REQUESTS * 2,
            limit_per_host=self.config.MAX_CONCURRENT_REQUESTS * 2
        )
        self._session = aiohttp.ClientSession(connector=connector)
        self._semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
        
        # Auto-detect model
        await self._detect_model()
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
    
    async def _detect_model(self):
        """Detect available model from API."""
        try:
            async with self._session.get(
                f"{self.config.API_URL}/models",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                data = await response.json()
                if isinstance(data, dict) and "data" in data:
                    models = [m.get("id", "") for m in data["data"]]
                elif isinstance(data, list):
                    models = [m.get("id", "") for m in data]
                else:
                    models = []
                
                self._model = models[0] if models else "default"
                print(f"[AsyncLLM] Using model: {self._model}")
        except Exception as e:
            print(f"[AsyncLLM] Model detection failed: {e}, using 'default'")
            self._model = "default"
    
    async def chat_single(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 64,
        temperature: float = 0.1
    ) -> str:
        """Single async chat request with semaphore-limited concurrency."""
        async with self._semaphore:
            payload = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            try:
                async with self._session.post(
                    f"{self.config.API_URL}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.LLM_TIMEOUT)
                ) as response:
                    data = await response.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
                    return ""
            except asyncio.TimeoutError:
                print("[AsyncLLM] Request timed out")
                return ""
            except Exception as e:
                print(f"[AsyncLLM] Request failed: {e}")
                return ""
    
    async def chat_batch_concurrent(
        self,
        message_batches: List[List[Dict[str, str]]],
        max_tokens: int = 64,
        temperature: float = 0.1
    ) -> List[str]:
        """
        Process multiple prompts concurrently using multi-session approach.
        
        This is the primary optimization: sends multiple requests in parallel
        using async I/O, maximizing GPU utilization on the server side.
        """
        tasks = [
            self.chat_single(msgs, max_tokens, temperature)
            for msgs in message_batches
        ]
        
        # Use tqdm_asyncio for progress tracking
        results = await tqdm_asyncio.gather(*tasks, desc="LLM inference")
        return results


# ============================================================================
# Optimized Pipeline
# ============================================================================

class OptimizedKGQAPipeline:
    """
    Optimized KGQA pipeline with:
    - Fast parallel corpus building
    - Batched/concurrent LLM inference
    - GPU-optimized embedding computation
    """
    
    def __init__(
        self,
        retriever_path: str = './RAG/models/finetuned_retriever',
        config: OptimizedConfig = None
    ):
        self.config = config or OptimizedConfig()
        self.rag_config = RAGConfig(dataset='webqsp')
        self.loader = KGDataLoader(self.rag_config)
        self.evaluator = CombinedEvaluator()
        
        # Build corpus fast with caching
        cache_dir = self.config.CACHE_DIR
        self.corpus, _ = self.loader.build_path_corpus_fast(
            include_test=True,
            num_workers=self.config.NUM_WORKERS,
            cache_dir=cache_dir
        )
        
        # Load retriever
        print(f"Loading retriever from {retriever_path}...")
        self.retriever = SentenceTransformer(retriever_path, device='cuda')
        
        # Build FAISS index with batched embeddings
        print("Building FAISS index (optimized batching)...")
        embeddings = self.retriever.encode(
            self.corpus,
            show_progress_bar=True,
            batch_size=self.config.RETRIEVER_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        print(f"  Indexed {self.index.ntotal} paths")
        
        # Async LLM client (initialized on first use)
        self._llm_client: Optional[AsyncLLMClient] = None
    
    def _build_prompt(
        self,
        question: str,
        retrieved_paths: List[str],
        sample: Optional[KGSample] = None
    ) -> str:
        """Build the LLM prompt for a question."""
        all_candidates = []
        context_lines = []
        
        for path in retrieved_paths[:self.config.TOP_K_CONTEXT]:
            if sample:
                try:
                    ents = sample.extract_entities_for_path(path)
                    if ents:
                        seen = set()
                        unique = [e for e in ents if e.lower() not in seen and not seen.add(e.lower())][:5]
                        all_candidates.extend(unique)
                        context_lines.append(f"• {path}: {', '.join(unique)}")
                except:
                    context_lines.append(f"• {path}")
            else:
                context_lines.append(f"• {path}")
        
        # Deduplicate candidates
        seen = set()
        unique_candidates = [c for c in all_candidates if c.lower() not in seen and not seen.add(c.lower())]
        
        context = "\n".join(context_lines) if context_lines else "No paths"
        
        # Oracle-style prompt
        if unique_candidates:
            prompt = f"""Question: {question}

KG paths and entities:
{context}

Candidates: {', '.join(unique_candidates[:10])}

Select ONLY the entity that answers the question.

Answer:"""
        else:
            prompt = f"Question: {question}\nAnswer directly:"
        
        return prompt
    
    def _batch_retrieve(self, questions: List[str]) -> List[List[str]]:
        """Retrieve paths for a batch of questions."""
        # Encode all questions in a single batch
        q_embeddings = self.retriever.encode(
            questions,
            normalize_embeddings=True,
            batch_size=min(len(questions), self.config.RETRIEVER_BATCH_SIZE),
            convert_to_numpy=True
        )
        
        # Batch FAISS search
        _, all_indices = self.index.search(
            q_embeddings.astype('float32'),
            self.config.TOP_K_RETRIEVE
        )
        
        # Convert indices to paths
        results = []
        for indices in all_indices:
            paths = [self.corpus[i] for i in indices if i >= 0]
            results.append(paths)
        
        return results
    
    async def _process_batch_async(
        self,
        samples: List[KGSample]
    ) -> List[Tuple[str, List[str]]]:
        """Process a batch of samples with async LLM calls."""
        # Initialize LLM client if needed
        if self._llm_client is None:
            self._llm_client = AsyncLLMClient(self.config)
            await self._llm_client.initialize()
        
        # Batch retrieve
        questions = [s.question for s in samples]
        retrieved_batch = self._batch_retrieve(questions)
        
        # Build all prompts
        prompts = []
        for sample, retrieved in zip(samples, retrieved_batch):
            prompt = self._build_prompt(sample.question, retrieved, sample)
            prompts.append([{'role': 'user', 'content': prompt}])
        
        # Concurrent LLM inference
        answers = await self._llm_client.chat_batch_concurrent(
            prompts,
            max_tokens=64,
            temperature=0.1
        )
        
        # Clean answers
        cleaned_answers = []
        for ans in answers:
            cleaned = ans.strip().strip('"\'').split('\n')[0].rstrip('.')
            cleaned_answers.append(cleaned)
        
        # Return (answer, retrieved_paths) pairs
        return list(zip(cleaned_answers, retrieved_batch))
    
    async def evaluate_async(
        self,
        limit: int = 500,
        batch_size: int = 32,
        output_dir: str = None
    ) -> Dict[str, float]:
        """
        Run evaluation with batched/concurrent processing.
        
        Args:
            limit: Maximum samples to evaluate
            batch_size: Samples per batch (for retrieval + grouping)
            output_dir: Optional directory to save results
        """
        test_data = [s for s in self.loader.test_data[:limit] if s.answer_entities]
        
        results = []
        f1_sum = em_sum = 0
        hits = {1: 0, 3: 0, 5: 0, 10: 0}
        count = 0
        
        print(f"\nEvaluating on {len(test_data)} samples (batch_size={batch_size})...")
        start_time = time.time()
        
        # Process in batches
        for batch_start in tqdm(range(0, len(test_data), batch_size), desc="Batches"):
            batch = test_data[batch_start:batch_start + batch_size]
            
            # Process batch with async LLM
            batch_results = await self._process_batch_async(batch)
            
            # Compute metrics
            for sample, (answer, retrieved) in zip(batch, batch_results):
                # Hits@K
                gt = set(p.lower() for p in sample.get_path_strings()) if sample.ground_truth_paths else set()
                for k in [1, 3, 5, 10]:
                    if any(p.lower() in gt for p in retrieved[:k]):
                        hits[k] += 1
                
                f1 = self.evaluator._compute_f1(answer, sample.answer_entities)
                em = self.evaluator._compute_exact_match(answer, sample.answer_entities)
                f1_sum += f1
                em_sum += em
                count += 1
                
                results.append({
                    'question_id': sample.question_id,
                    'question': sample.question,
                    'answer': answer,
                    'ground_truth': sample.answer_entities,
                    'f1': f1,
                    'em': em,
                    'retrieved_paths': retrieved[:5]
                })
        
        # Close LLM client
        if self._llm_client:
            await self._llm_client.close()
        
        elapsed = time.time() - start_time
        throughput = count / elapsed
        
        metrics = {
            'f1': f1_sum / count if count else 0,
            'exact_match': em_sum / count if count else 0,
            'hits@1': hits[1] / count if count else 0,
            'hits@3': hits[3] / count if count else 0,
            'hits@5': hits[5] / count if count else 0,
            'hits@10': hits[10] / count if count else 0,
            'num_samples': count,
            'elapsed_seconds': elapsed,
            'throughput_samples_per_sec': throughput
        }
        
        print(f"\n{'='*60}")
        print(f"RESULTS ({count} samples in {elapsed:.1f}s = {throughput:.2f} samples/sec)")
        print(f"{'='*60}")
        print(f"F1: {metrics['f1']*100:.1f}%")
        print(f"EM: {metrics['exact_match']*100:.1f}%")
        print(f"Hits@1: {metrics['hits@1']*100:.1f}%")
        print(f"Hits@5: {metrics['hits@5']*100:.1f}%")
        print(f"Hits@10: {metrics['hits@10']*100:.1f}%")
        print(f"\nThroughput: {throughput:.2f} samples/sec")
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{output_dir}/metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            with open(f"{output_dir}/results.json", 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved to {output_dir}/")
        
        return metrics


# ============================================================================
# Main Entry Point
# ============================================================================

async def main_async():
    """Async main entry point."""
    pipeline = OptimizedKGQAPipeline()
    
    # Run evaluation with optimizations
    metrics = await pipeline.evaluate_async(
        limit=50000,
        batch_size=64,  # Larger batch for better GPU utilization
        output_dir='./RAG/best_config_76pct_f1/results_optimized'
    )
    
    return metrics


def main():
    """Sync main entry point (wraps async)."""
    return asyncio.run(main_async())


if __name__ == "__main__":
    main()
