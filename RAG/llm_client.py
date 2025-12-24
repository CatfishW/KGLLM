"""
LLM Client for the KGQA system.

OpenAI-compatible client with auto model detection.
Uses the API at https://game.agaii.org/llm/v1
"""
import os
import json
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    api_url: str = "https://game.agaii.org/llm/v1"
    model: str = ""  # Auto-detect if empty
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: int = 60


class LLMClient:
    """
    OpenAI-compatible LLM client with auto model detection.
    
    Features:
    - Auto-detects available models from /models endpoint
    - Chat completion for answer generation
    - Streaming support
    - Error handling with retries
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._available_models: Optional[List[str]] = None
        self._selected_model: Optional[str] = None
    
    @property
    def api_url(self) -> str:
        return self.config.api_url.rstrip("/")
    
    @property
    def available_models(self) -> List[str]:
        """Fetch available models from the API."""
        if self._available_models is None:
            self._available_models = self._fetch_models()
        return self._available_models
    
    @property
    def model(self) -> str:
        """Get the model to use (auto-detect if not specified)."""
        if self._selected_model:
            return self._selected_model
        
        if self.config.model:
            self._selected_model = self.config.model
        else:
            # Auto-detect: use the first available model
            models = self.available_models
            if models:
                self._selected_model = models[0]
                print(f"[LLMClient] Auto-detected model: {self._selected_model}")
            else:
                raise RuntimeError("No models available at the API")
        
        return self._selected_model
    
    def _fetch_models(self) -> List[str]:
        """Fetch available models from /models endpoint."""
        try:
            response = requests.get(
                f"{self.api_url}/models",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Handle OpenAI-style response
            if isinstance(data, dict) and "data" in data:
                models = [m.get("id", m.get("model", "")) for m in data["data"]]
            elif isinstance(data, list):
                models = [m.get("id", m.get("model", "")) for m in data]
            else:
                models = []
            
            return [m for m in models if m]  # Filter empty strings
        except Exception as e:
            print(f"[LLMClient] Failed to fetch models: {e}")
            return []
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override config temperature
            max_tokens: Override config max_tokens
            
        Returns:
            Assistant's response content
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract response content
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                return content.strip()
            
            return ""
        except requests.exceptions.Timeout:
            print(f"[LLMClient] Request timed out")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"[LLMClient] Request failed: {e}")
            return ""
        except Exception as e:
            print(f"[LLMClient] Unexpected error: {e}")
            return ""
    
    def generate_answer(
        self,
        question: str,
        paths: List[str],
        topic_entity: Optional[str] = None,
        max_tokens: int = 128
    ) -> str:
        """
        Generate an answer using the question and retrieved/generated paths.
        
        Uses KGQA-optimized prompts that emphasize:
        1. Extracting specific entity names (not paraphrasing)
        2. Short, direct answers matching benchmark format
        3. Using relation paths to infer answer entities
        """
        # Format paths clearly
        paths_str = "\n".join([f"• {p}" for p in paths[:5]])  # Use fewer, top paths
        
        # Refined system prompt for KGQA - simpler, more effective
        system_prompt = """You are a Knowledge Graph Question Answering expert. Extract the EXACT answer entity from the given paths.

RULES:
1. Give SHORT answers - typically 1-5 words (entity names only)
2. ONLY use information from the paths - DO NOT use your own knowledge  
3. Multiple answers should be comma-separated
4. Never explain - just give the entity name(s)

EXAMPLES:
- Q: "What language is spoken in Jamaica?" + Path: "location.country.languages_spoken" → "English"
- Q: "Where was X born?" + Path: "people.person.place_of_birth" → "Mobile"
- Q: "Who founded X?" + Path: "organization.founders" → "John Smith"

IMPORTANT: Only answer with what can be inferred from the paths."""
        
        # Build user prompt with topic entity integration
        topic_line = f"Topic Entity: {topic_entity}\n" if topic_entity else ""
        
        user_prompt = f"""Question: {question}
{topic_line}
Knowledge Graph Paths:
{paths_str}

Based on these paths, what is the answer? Give ONLY the specific entity name(s).

Answer:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.chat(messages, max_tokens=max_tokens, temperature=0.1)
        
        # Post-process: clean up common issues
        response = self._clean_answer(response)
        
        return response
    
    def _extract_answer_type_hints(self, paths: List[str]) -> str:
        """
        Extract expected answer type hints from path endings.
        
        Analyzes the last relation in each path to determine what type
        of entity the answer should be (place, language, person, etc.)
        """
        # Mapping from relation keywords to answer types
        type_mapping = {
            'place_of_birth': 'place/city',
            'place_of_death': 'place/city',
            'location': 'place/location',
            'country': 'country',
            'capital': 'city',
            'language': 'language',
            'languages_spoken': 'language',
            'official_language': 'language',
            'religion': 'religion',
            'nationality': 'nationality/country',
            'form_of_government': 'government type',
            'championships': 'championship/year',
            'sport': 'sport',
            'president': 'person name',
            'vice_president': 'person name',
            'office_holder': 'person name',
            'founder': 'person/organization',
            'inventor': 'invention',
            'inventions': 'invention',
            'team': 'team name',
            'mascot': 'mascot',
            'spouse': 'person name',
            'parent': 'person name',
            'child': 'person name',
            'currency': 'currency',
            'airport': 'airport name',
            'school': 'school/university',
        }
        
        detected_types = set()
        
        for path in paths[:5]:  # Look at top 5 paths
            # Get the last relation
            relations = path.split('->')
            last_rel = relations[-1].strip().lower()
            
            # Check against mapping
            for keyword, answer_type in type_mapping.items():
                if keyword in last_rel:
                    detected_types.add(answer_type)
                    break
        
        if detected_types:
            return ', '.join(detected_types)
        return ""
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up LLM answer for better matching."""
        if not answer:
            return ""
        
        # Remove common prefixes the model might add
        prefixes_to_remove = [
            "The answer is ",
            "Answer: ",
            "Based on the paths, ",
            "The specific entity is ",
            "According to the knowledge graph, ",
        ]
        
        answer = answer.strip()
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Remove quotes
        answer = answer.strip('"\'')
        
        # Remove trailing periods
        answer = answer.rstrip('.')
        
        return answer.strip()
    
    def generate_answer_with_entities(
        self,
        question: str,
        paths: List[str],
        candidate_entities: List[str],
        topic_entity: Optional[str] = None,
        max_tokens: int = 256
    ) -> str:
        """
        Generate an answer with candidate answer entities.
        
        Args:
            question: The user's question
            paths: List of relation paths
            candidate_entities: Entities found at the end of the paths
            topic_entity: Optional topic entity
            max_tokens: Maximum tokens for response
            
        Returns:
            Generated answer string
        """
        paths_str = "\n".join([f"• {p}" for p in paths[:5]])
        entities_str = ", ".join(candidate_entities[:20])
        
        # Hybrid LLM+KG prompt - SOTA approach combining KG grounding with LLM reasoning
        system_prompt = """You are a KGQA expert combining Knowledge Graph facts with reasoning.

PROCESS:
1. First, analyze what the question is asking for (type of entity needed)
2. Check if any candidate entities from the KG match what's needed
3. If a candidate matches well, choose it
4. If NO candidate matches well, use your own knowledge to answer

RULES:
- Prefer KG candidates when they make sense for the question
- Give SHORT answers (1-5 words)
- Multiple answers separated by commas
- If KG candidates don't match, you MAY use your knowledge"""
        
        topic_line = f"Topic Entity: {topic_entity}\n" if topic_entity else ""
        
        user_prompt = f"""Question: {question}
{topic_line}
Knowledge Graph Paths:
{paths_str}

Candidate Entities from KG: {entities_str}

Step 1 - Question asks for: [what type of entity?]
Step 2 - Best matching candidate: [or "none match well"]
Step 3 - Final answer:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.chat(messages, max_tokens=max_tokens, temperature=0.1)
        
        # Extract final answer
        if "Final answer:" in response.lower():
            answer = response.lower().split("final answer:")[-1].strip()
        elif "step 3" in response.lower():
            lines = response.split("\n")
            for line in lines:
                if "step 3" in line.lower():
                    answer = line.split(":")[-1].strip() if ":" in line else ""
                    break
            else:
                answer = response.strip()
        else:
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            answer = lines[-1] if lines else response
        
        # Clean answer
        answer = answer.strip('."\'')
        # Remove step markers if present
        for marker in ['step 1', 'step 2', 'step 3']:
            if marker in answer.lower():
                answer = answer.split(':')[-1].strip()
        
        return answer


# Singleton instance for convenience
_default_client: Optional[LLMClient] = None


def get_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """Get or create default LLM client."""
    global _default_client
    if _default_client is None or config is not None:
        _default_client = LLMClient(config)
    return _default_client


if __name__ == "__main__":
    # Quick test
    client = LLMClient()
    print(f"Available models: {client.available_models}")
    print(f"Using model: {client.model}")
    
    # Test answer generation
    answer = client.generate_answer(
        question="What language is spoken in Jamaica?",
        paths=["location.country -> language.official_language"],
        topic_entity="Jamaica"
    )
    print(f"Answer: {answer}")
