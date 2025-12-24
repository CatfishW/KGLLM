"""
Advanced LLM Client with True Batch Inference Support

Provides multiple inference strategies:
1. Single request (baseline)
2. Multi-session concurrent (async I/O)
3. True batched inference (if API supports /v1/completions with multiple prompts)
"""
import asyncio
import aiohttp
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class BatchLLMConfig:
    """Configuration for batch LLM client."""
    api_url: str = "https://game.agaii.org/llm/v1"
    max_concurrent: int = 16
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0


class BatchLLMClient:
    """
    Advanced LLM client supporting multiple inference modes.
    
    Mode 1: Concurrent Async Sessions
        - Uses asyncio + aiohttp for parallel HTTP requests
        - Good for APIs that don't support native batching
        - Throughput scales with max_concurrent setting
        
    Mode 2: True Batched Inference (vLLM-style)
        - Single request with multiple prompts
        - Maximum efficiency if server supports it
        - Falls back to concurrent if not supported
    """
    
    def __init__(self, config: BatchLLMConfig = None):
        self.config = config or BatchLLMConfig()
        self._model: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._supports_batch: Optional[bool] = None
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def initialize(self):
        """Initialize async session and detect capabilities."""
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent * 2,
            limit_per_host=self.config.max_concurrent * 2,
            keepalive_timeout=300
        )
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Detect model and batch support
        await self._detect_capabilities()
    
    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
    
    async def _detect_capabilities(self):
        """Detect model and batch inference support."""
        try:
            async with self._session.get(f"{self.config.api_url}/models") as resp:
                data = await resp.json()
                if isinstance(data, dict) and "data" in data:
                    models = [m.get("id", "") for m in data["data"]]
                else:
                    models = [data.get("id", "default")] if isinstance(data, dict) else []
                
                self._model = models[0] if models else "default"
                print(f"[BatchLLM] Detected model: {self._model}")
        except Exception as e:
            print(f"[BatchLLM] Model detection failed: {e}")
            self._model = "default"
        
        # Check if true batch is supported by trying /completions endpoint
        # vLLM and some other servers support multiple prompts in one call
        self._supports_batch = await self._check_batch_support()
        print(f"[BatchLLM] True batch support: {self._supports_batch}")
    
    async def _check_batch_support(self) -> bool:
        """Check if the API supports true batched inference."""
        try:
            # Try a minimal batch request
            payload = {
                "model": self._model,
                "prompt": ["Hello", "World"],  # Multiple prompts
                "max_tokens": 1,
                "temperature": 0
            }
            async with self._session.post(
                f"{self.config.api_url}/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Check if we got multiple choices back
                    choices = data.get("choices", [])
                    return len(choices) >= 2
                return False
        except:
            return False
    
    async def _single_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 64,
        temperature: float = 0.1,
        retry_count: int = 0
    ) -> str:
        """Single chat completion with retry logic."""
        async with self._semaphore:
            payload = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            try:
                async with self._session.post(
                    f"{self.config.api_url}/chat/completions",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            return choices[0].get("message", {}).get("content", "").strip()
                    elif resp.status in (429, 503) and retry_count < self.config.max_retries:
                        # Rate limited or overloaded - retry with backoff
                        await asyncio.sleep(self.config.retry_delay * (2 ** retry_count))
                        return await self._single_chat(messages, max_tokens, temperature, retry_count + 1)
                    return ""
            except asyncio.TimeoutError:
                if retry_count < self.config.max_retries:
                    return await self._single_chat(messages, max_tokens, temperature, retry_count + 1)
                return ""
            except Exception as e:
                print(f"[BatchLLM] Error: {e}")
                return ""
    
    async def _true_batch_completion(
        self,
        prompts: List[str],
        max_tokens: int = 64,
        temperature: float = 0.1
    ) -> List[str]:
        """True batched completion using /completions endpoint."""
        payload = {
            "model": self._model,
            "prompt": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            async with self._session.post(
                f"{self.config.api_url}/completions",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    choices = data.get("choices", [])
                    # Sort by index and extract text
                    sorted_choices = sorted(choices, key=lambda c: c.get("index", 0))
                    return [c.get("text", "").strip() for c in sorted_choices]
                return [""] * len(prompts)
        except Exception as e:
            print(f"[BatchLLM] Batch error: {e}")
            return [""] * len(prompts)
    
    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 64,
        temperature: float = 0.1,
        use_true_batch: bool = True
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Uses true batch inference if supported, otherwise falls back to concurrent.
        
        Args:
            prompts: List of prompt strings
            max_tokens: Max tokens per response
            temperature: Sampling temperature
            use_true_batch: Try true batch first (if supported)
            
        Returns:
            List of response strings
        """
        if not prompts:
            return []
        
        # Try true batch if supported and requested
        if use_true_batch and self._supports_batch:
            return await self._true_batch_completion(prompts, max_tokens, temperature)
        
        # Fallback to concurrent sessions
        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        tasks = [self._single_chat(msgs, max_tokens, temperature) for msgs in messages_list]
        return await asyncio.gather(*tasks)
    
    async def chat_batch_concurrent(
        self,
        message_batches: List[List[Dict[str, str]]],
        max_tokens: int = 64,
        temperature: float = 0.1
    ) -> List[str]:
        """
        Process multiple chat message sequences concurrently.
        
        This is the main method for concurrent LLM inference.
        """
        tasks = [
            self._single_chat(msgs, max_tokens, temperature)
            for msgs in message_batches
        ]
        return await asyncio.gather(*tasks)


# ============================================================================
# Synchronous Batch Client (wrapper for non-async code)
# ============================================================================

class SyncBatchLLMClient:
    """
    Synchronous wrapper for BatchLLMClient.
    
    Useful when integrating with non-async code but still wanting
    batch/concurrent inference benefits.
    """
    
    def __init__(self, config: BatchLLMConfig = None):
        self.config = config or BatchLLMConfig()
        self._async_client: Optional[BatchLLMClient] = None
    
    def _ensure_initialized(self):
        """Ensure async client is initialized."""
        if self._async_client is None:
            self._async_client = BatchLLMClient(self.config)
            asyncio.run(self._async_client.initialize())
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 64,
        temperature: float = 0.1
    ) -> List[str]:
        """Synchronous batch generation."""
        self._ensure_initialized()
        return asyncio.run(
            self._async_client.generate_batch(prompts, max_tokens, temperature)
        )
    
    def chat_batch(
        self,
        message_batches: List[List[Dict[str, str]]],
        max_tokens: int = 64,
        temperature: float = 0.1
    ) -> List[str]:
        """Synchronous batch chat."""
        self._ensure_initialized()
        return asyncio.run(
            self._async_client.chat_batch_concurrent(message_batches, max_tokens, temperature)
        )
    
    def close(self):
        """Close the client."""
        if self._async_client:
            asyncio.run(self._async_client.close())


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    async def test():
        config = BatchLLMConfig(max_concurrent=8)
        
        async with BatchLLMClient(config) as client:
            # Test concurrent chat
            prompts = [
                "What is 2+2?",
                "Who is the president of France?",
                "What color is the sky?"
            ]
            
            print("\nTesting concurrent generation...")
            start = time.time()
            results = await client.generate_batch(prompts)
            elapsed = time.time() - start
            
            for prompt, result in zip(prompts, results):
                print(f"Q: {prompt}")
                print(f"A: {result}\n")
            
            print(f"Time: {elapsed:.2f}s for {len(prompts)} prompts")
            print(f"Throughput: {len(prompts)/elapsed:.2f} prompts/sec")
    
    asyncio.run(test())
