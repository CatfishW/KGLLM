#!/usr/bin/env python3
"""
Anti-GPTZero Paper Humanizer Script - FREE VERSION
====================================================
This script processes LaTeX papers section-by-section to bypass AI detection
tools like GPTZero, Turnitin, and Originality.ai.

FREE API/METHOD OPTIONS:
1. Ollama (LOCAL) - Run local LLM like Llama3, Mistral - COMPLETELY FREE
2. Selenium Web Scraper - Scrapes free humanizer websites
3. Apify - $5 free credits/month
4. Rule-based transformations - No API needed

Usage:
    # Using local Ollama (recommended - completely free)
    python humanize_paper.py lam_main_latest.tex --api ollama --model llama3
    
    # Using web scraping (free, but slower)
    python humanize_paper.py lam_main_latest.tex --api webscrape
    
    # Using rule-based (no API, basic transformations)
    python humanize_paper.py lam_main_latest.tex --api rulebased
"""

import re
import os
import sys
import json
import time
import random
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('humanize_paper.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a LaTeX section with its content."""
    name: str
    level: str
    content: str
    start_pos: int
    end_pos: int
    raw_header: str


@dataclass
class HumanizationResult:
    """Result from a humanization API call."""
    success: bool
    original_text: str
    humanized_text: str
    error_message: Optional[str] = None
    api_response: Optional[Dict] = None


class HumanizerAPI(ABC):
    """Abstract base class for humanizer API providers."""
    
    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self.session = requests.Session()
    
    @abstractmethod
    def humanize(self, text: str) -> HumanizationResult:
        """Humanize the given text."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name."""
        pass


class OllamaLocalAPI(HumanizerAPI):
    """
    Local Ollama API - COMPLETELY FREE
    Requires: ollama installed locally with a model like llama3, mistral, or phi3
    Install: https://ollama.ai/download
    Pull model: ollama pull llama3
    """
    
    BASE_URL = "http://localhost:11434/api/generate"
    
    def __init__(self, model: str = "llama3", timeout: int = 180):
        super().__init__(timeout)
        self.model = model
    
    def get_provider_name(self) -> str:
        return f"Ollama Local ({self.model})"
    
    def humanize(self, text: str) -> HumanizationResult:
        """Humanize text using local Ollama model."""
        
        prompt = f"""You are an expert academic writing editor. Your task is to rewrite the following academic text to sound more naturally human-written while:
1. Preserving all technical accuracy and meaning
2. Keeping the same formal academic tone
3. Maintaining all citations in their exact format (e.g., \\cite{{...}})
4. Preserving all mathematical notation and LaTeX commands
5. Varying sentence structure and length naturally
6. Using more varied vocabulary and phrasing
7. Adding natural transitions between ideas

IMPORTANT: Do NOT change any LaTeX commands, citations, equations, or technical terms.
Only rewrite the prose text to sound more human.

Original text:
{text}

Rewritten text (preserve all LaTeX formatting):"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": len(text) * 2  # Allow for expansion
            }
        }
        
        try:
            response = self.session.post(
                self.BASE_URL,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            humanized = data.get("response", "").strip()
            
            if humanized:
                return HumanizationResult(
                    success=True,
                    original_text=text,
                    humanized_text=humanized,
                    api_response=data
                )
            else:
                return HumanizationResult(
                    success=False,
                    original_text=text,
                    humanized_text="",
                    error_message=f"Empty response from Ollama",
                    api_response=data
                )
                
        except requests.exceptions.ConnectionError:
            return HumanizationResult(
                success=False,
                original_text=text,
                humanized_text="",
                error_message="Ollama not running. Start with: ollama serve"
            )
        except requests.exceptions.RequestException as e:
            return HumanizationResult(
                success=False,
                original_text=text,
                humanized_text="",
                error_message=str(e)
            )


class RuleBasedHumanizer(HumanizerAPI):
    """
    Rule-based text humanization - NO API NEEDED
    Uses NLP techniques to vary text without external APIs.
    Less effective but completely free and offline.
    """
    
    # Synonym replacements for common academic words
    SYNONYMS = {
        "utilize": ["use", "employ", "leverage", "apply"],
        "demonstrate": ["show", "illustrate", "reveal", "indicate"],
        "significant": ["notable", "substantial", "considerable", "meaningful"],
        "implement": ["execute", "carry out", "put into practice", "realize"],
        "achieve": ["attain", "accomplish", "reach", "obtain"],
        "propose": ["suggest", "put forward", "present", "introduce"],
        "evaluate": ["assess", "examine", "analyze", "appraise"],
        "improve": ["enhance", "boost", "strengthen", "advance"],
        "reduce": ["decrease", "diminish", "lower", "minimize"],
        "increase": ["raise", "elevate", "boost", "amplify"],
        "however": ["nevertheless", "nonetheless", "yet", "still"],
        "therefore": ["thus", "hence", "consequently", "as a result"],
        "additionally": ["moreover", "furthermore", "in addition", "also"],
        "specifically": ["particularly", "especially", "notably", "in particular"],
        "primarily": ["mainly", "chiefly", "principally", "predominantly"],
        "existing": ["current", "present", "established", "conventional"],
        "novel": ["new", "innovative", "original", "unique"],
        "efficient": ["effective", "streamlined", "optimized", "productive"],
        "fundamental": ["basic", "essential", "core", "underlying"],
        "comprehensive": ["thorough", "complete", "extensive", "detailed"],
    }
    
    # Sentence starters for variety
    TRANSITIONS = [
        "To this end,", "In this regard,", "Building on this,",
        "With this in mind,", "In light of this,", "Given this context,",
        "From this perspective,", "Along these lines,", "In this vein,",
    ]
    
    def __init__(self, timeout: int = 120):
        super().__init__(timeout)
        random.seed(42)  # For reproducibility
    
    def get_provider_name(self) -> str:
        return "Rule-based (No API)"
    
    def _replace_synonyms(self, text: str) -> str:
        """Replace words with synonyms randomly."""
        words = text.split()
        result = []
        for word in words:
            word_lower = word.lower().strip('.,;:!?')
            if word_lower in self.SYNONYMS and random.random() > 0.5:
                # Preserve case
                replacement = random.choice(self.SYNONYMS[word_lower])
                if word[0].isupper():
                    replacement = replacement.capitalize()
                # Preserve trailing punctuation
                trailing = ''
                for c in reversed(word):
                    if c in '.,;:!?':
                        trailing = c + trailing
                    else:
                        break
                result.append(replacement + trailing)
            else:
                result.append(word)
        return ' '.join(result)
    
    def _vary_sentence_structure(self, text: str) -> str:
        """Add variety to sentence structures."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Skip very short sentences or those starting with LaTeX commands
            if len(sentence) < 20 or sentence.startswith('\\'):
                result.append(sentence)
                continue
            
            # Occasionally add transition phrases (10% chance)
            if random.random() > 0.9 and not sentence.startswith(tuple(self.TRANSITIONS)):
                transition = random.choice(self.TRANSITIONS)
                # Lowercase the first letter of the sentence after transition
                sentence = transition + " " + sentence[0].lower() + sentence[1:]
            
            result.append(sentence)
        
        return ' '.join(result)
    
    def _add_hedging_language(self, text: str) -> str:
        """Add academic hedging language to make text more human."""
        # Replace absolute statements with hedged versions
        hedging_patterns = [
            (r'\bis\b', lambda m: random.choice(['is', 'appears to be', 'seems to be']) if random.random() > 0.8 else 'is'),
            (r'\bshows\b', lambda m: random.choice(['shows', 'suggests', 'indicates']) if random.random() > 0.7 else 'shows'),
            (r'\bproves\b', lambda m: random.choice(['proves', 'demonstrates', 'provides evidence']) if random.random() > 0.7 else 'proves'),
        ]
        
        for pattern, replacement in hedging_patterns:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
        
        return text
    
    def humanize(self, text: str) -> HumanizationResult:
        """Apply rule-based humanization."""
        try:
            # Apply transformations
            humanized = text
            humanized = self._replace_synonyms(humanized)
            humanized = self._vary_sentence_structure(humanized)
            humanized = self._add_hedging_language(humanized)
            
            return HumanizationResult(
                success=True,
                original_text=text,
                humanized_text=humanized
            )
        except Exception as e:
            return HumanizationResult(
                success=False,
                original_text=text,
                humanized_text="",
                error_message=str(e)
            )


class HuggingFaceHumanizer(HumanizerAPI):
    """
    HuggingFace Transformers - COMPLETELY FREE, runs locally
    
    Recommended models:
    - tuner007/pegasus_paraphrase: Best for paraphrasing (~2GB, fast)
    - humarin/chatgpt_paraphraser_on_T5_base: T5-based paraphraser (~900MB)
    
    Install: pip install transformers torch sentencepiece
    """
    
    MODELS = {
        "pegasus": "tuner007/pegasus_paraphrase",
        "t5": "humarin/chatgpt_paraphraser_on_T5_base",
    }
    
    def __init__(self, model_name: str = "pegasus", timeout: int = 300, force_cpu: bool = False):
        super().__init__(timeout)
        self.model_name = model_name
        self.model_id = self.MODELS.get(model_name, model_name)
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.force_cpu = force_cpu
        self.device = "cpu"  # Will be updated on load
    
    def get_provider_name(self) -> str:
        return f"HuggingFace ({self.model_name})"
    
    def _load_model(self) -> bool:
        """Lazy load the model on first use."""
        if self._loaded:
            return True
            
        try:
            import torch
            
            logger.info(f"Loading HuggingFace model: {self.model_id}")
            logger.info("This may take a few minutes on first run (downloading model)...")
            
            # Use specific tokenizer/model classes for PEGASUS
            if "pegasus" in self.model_id.lower():
                from transformers import PegasusTokenizer, PegasusForConditionalGeneration
                self.tokenizer = PegasusTokenizer.from_pretrained(self.model_id)
                self.model = PegasusForConditionalGeneration.from_pretrained(self.model_id)
            else:
                # For T5 and other models
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            
            # Use GPU if available and not forced to CPU
            if torch.cuda.is_available() and not self.force_cpu:
                self.model = self.model.cuda()
                self.device = "cuda"
                logger.info("Using GPU for inference")
            else:
                self.device = "cpu"
                logger.info("Using CPU for inference" + (" (forced)" if self.force_cpu else " (no GPU available)"))
            
            self._loaded = True
            logger.info("Model loaded successfully!")
            return True
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.error("Install with: pip install transformers torch sentencepiece")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _paraphrase_chunk(self, text: str, num_beams: int = 5, num_return_sequences: int = 1) -> str:
        """Paraphrase a single chunk of text."""
        import torch
        
        # For PEGASUS model
        if "pegasus" in self.model_id.lower():
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
        else:
            # For T5 model - add prefix
            inputs = self.tokenizer(
                f"paraphrase: {text}",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
        
        # Move inputs to correct device
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        paraphrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrased
    
    def humanize(self, text: str) -> HumanizationResult:
        """Humanize text using HuggingFace model."""
        if not self._load_model():
            return HumanizationResult(
                success=False,
                original_text=text,
                humanized_text="",
                error_message="Failed to load HuggingFace model. Install: pip install transformers torch sentencepiece"
            )
        
        try:
            # Split into sentences for better handling of long text
            sentences = re.split(r'(?<=[.!?])\s+', text)
            humanized_sentences = []
            
            # Process in chunks (batch sentences for efficiency)
            chunk_size = 3  # Process 3 sentences at a time
            for i in range(0, len(sentences), chunk_size):
                chunk = ' '.join(sentences[i:i+chunk_size])
                
                # Skip very short chunks or chunks that are mostly placeholders
                if len(chunk.strip()) < 50 or chunk.count('__') > 5:
                    humanized_sentences.append(chunk)
                    continue
                
                try:
                    paraphrased = self._paraphrase_chunk(chunk)
                    humanized_sentences.append(paraphrased)
                except Exception as e:
                    logger.warning(f"Failed to paraphrase chunk, keeping original: {e}")
                    humanized_sentences.append(chunk)
            
            humanized = ' '.join(humanized_sentences)
            
            return HumanizationResult(
                success=True,
                original_text=text,
                humanized_text=humanized
            )
            
        except Exception as e:
            return HumanizationResult(
                success=False,
                original_text=text,
                humanized_text="",
                error_message=str(e)
            )


class WebScraperHumanizer(HumanizerAPI):
    """
    Web scraper for free humanizer websites.
    Uses Selenium to interact with free online humanizers.
    Requires: pip install selenium webdriver-manager
    """
    
    SITES = {
        "humanizeai": {
            "url": "https://humanizeai.pro/",
            "input_selector": "textarea",
            "button_selector": "button[type='submit'], button.humanize-btn",
            "output_selector": ".output-text, textarea.output"
        },
        "zerogpt": {
            "url": "https://zerogpt.com/humanize-ai-text",
            "input_selector": "textarea#input-text",
            "button_selector": "button#humanize-btn",
            "output_selector": "#output-text"
        }
    }
    
    def __init__(self, site: str = "humanizeai", timeout: int = 60):
        super().__init__(timeout)
        self.site = site
        self.driver = None
        
    def get_provider_name(self) -> str:
        return f"Web Scraper ({self.site})"
    
    def _init_driver(self):
        """Initialize Selenium WebDriver."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager
            
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            return True
        except ImportError:
            logger.error("Selenium not installed. Run: pip install selenium webdriver-manager")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            return False
    
    def humanize(self, text: str) -> HumanizationResult:
        """Humanize text using web scraping."""
        if not self.driver and not self._init_driver():
            return HumanizationResult(
                success=False,
                original_text=text,
                humanized_text="",
                error_message="Failed to initialize Selenium. Install with: pip install selenium webdriver-manager"
            )
        
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            site_config = self.SITES.get(self.site, self.SITES["humanizeai"])
            
            # Navigate to the site
            self.driver.get(site_config["url"])
            time.sleep(3)  # Wait for page to load
            
            # Find and fill the input textarea
            input_elem = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, site_config["input_selector"]))
            )
            input_elem.clear()
            input_elem.send_keys(text)
            
            # Click the humanize button
            button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, site_config["button_selector"]))
            )
            button.click()
            
            # Wait for result
            time.sleep(10)  # Wait for processing
            
            # Get the output
            output_elem = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, site_config["output_selector"]))
            )
            humanized = output_elem.text or output_elem.get_attribute("value")
            
            if humanized and humanized.strip():
                return HumanizationResult(
                    success=True,
                    original_text=text,
                    humanized_text=humanized.strip()
                )
            else:
                return HumanizationResult(
                    success=False,
                    original_text=text,
                    humanized_text="",
                    error_message="No output received from website"
                )
                
        except Exception as e:
            return HumanizationResult(
                success=False,
                original_text=text,
                humanized_text="",
                error_message=f"Web scraping error: {str(e)}"
            )
    
    def __del__(self):
        """Clean up WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass


class ApifyHumanizer(HumanizerAPI):
    """
    Apify humanizer - $5 FREE credits per month
    Sign up at: https://apify.com/ (free tier available)
    Get API token from: Apify Console -> Settings -> Integrations
    """
    
    ACTOR_ID = "curious_coder/humanize-ai-text"
    BASE_URL = "https://api.apify.com/v2/acts"
    
    def __init__(self, api_token: str = None, timeout: int = 120):
        super().__init__(timeout)
        self.api_token = api_token or os.environ.get('APIFY_TOKEN')
    
    def get_provider_name(self) -> str:
        return "Apify (Free $5/month)"
    
    def humanize(self, text: str) -> HumanizationResult:
        """Humanize text using Apify."""
        if not self.api_token:
            return HumanizationResult(
                success=False,
                original_text=text,
                humanized_text="",
                error_message="Apify token required. Set APIFY_TOKEN env var or get free account at apify.com"
            )
        
        url = f"{self.BASE_URL}/{self.ACTOR_ID}/runs?token={self.api_token}"
        
        payload = {
            "text": text,
            "style": "academic"
        }
        
        try:
            # Start the actor run
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            run_data = response.json()
            run_id = run_data.get("data", {}).get("id")
            
            if not run_id:
                return HumanizationResult(
                    success=False,
                    original_text=text,
                    humanized_text="",
                    error_message="Failed to start Apify actor"
                )
            
            # Poll for completion
            run_url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={self.api_token}"
            for _ in range(30):
                time.sleep(5)
                run_response = self.session.get(run_url, timeout=self.timeout)
                run_info = run_response.json()
                status = run_info.get("data", {}).get("status")
                
                if status == "SUCCEEDED":
                    # Get the dataset
                    dataset_id = run_info.get("data", {}).get("defaultDatasetId")
                    dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={self.api_token}"
                    dataset_response = self.session.get(dataset_url, timeout=self.timeout)
                    items = dataset_response.json()
                    
                    if items and len(items) > 0:
                        humanized = items[0].get("humanizedText", items[0].get("text", ""))
                        if humanized:
                            return HumanizationResult(
                                success=True,
                                original_text=text,
                                humanized_text=humanized
                            )
                    break
                elif status in ["FAILED", "ABORTED", "TIMED-OUT"]:
                    break
            
            return HumanizationResult(
                success=False,
                original_text=text,
                humanized_text="",
                error_message=f"Apify run failed or timed out"
            )
                
        except requests.exceptions.RequestException as e:
            return HumanizationResult(
                success=False,
                original_text=text,
                humanized_text="",
                error_message=str(e)
            )


class LaTeXParser:
    """Parser for LaTeX documents to extract sections."""
    
    PROTECTED_ENVIRONMENTS = [
        'equation', 'align', 'gather', 'multline', 'eqnarray',
        'figure', 'table', 'tabular', 'algorithm', 'algorithmic',
        'lstlisting', 'verbatim', 'minted', 'tikzpicture',
        'proof', 'theorem', 'lemma', 'corollary', 'definition',
    ]
    
    PROTECTED_COMMANDS = [
        r'\\cite\{[^}]*\}',
        r'\\ref\{[^}]*\}',
        r'\\label\{[^}]*\}',
        r'\\texttt\{[^}]*\}',
        r'\\model\{\}',
        r'\\[a-zA-Z]+\{[^}]*\}',
        r'\$[^$]+\$',
        r'\$\$[^$]+\$\$',
        r'~',
    ]
    
    def __init__(self, content: str):
        self.content = content
        self.sections: List[Section] = []
    
    def extract_sections(self) -> List[Section]:
        """Extract all sections from the LaTeX document."""
        section_pattern = r'\\(section|subsection|subsubsection)\{([^}]+)\}'
        matches = list(re.finditer(section_pattern, self.content))
        
        for i, match in enumerate(matches):
            level = match.group(1)
            name = match.group(2)
            start_pos = match.end()
            
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_match = re.search(r'\\end\{document\}', self.content[start_pos:])
                if end_match:
                    end_pos = start_pos + end_match.start()
                else:
                    end_pos = len(self.content)
            
            section_content = self.content[start_pos:end_pos].strip()
            
            self.sections.append(Section(
                name=name,
                level=level,
                content=section_content,
                start_pos=match.start(),
                end_pos=end_pos,
                raw_header=match.group(0)
            ))
        
        return self.sections
    
    def extract_text_for_humanization(self, content: str) -> Tuple[str, Dict[str, str]]:
        """Extract plain text suitable for humanization."""
        placeholder_map = {}
        text = content
        
        for env in self.PROTECTED_ENVIRONMENTS:
            pattern = rf'\\begin\{{{env}\}}.*?\\end\{{{env}\}}'
            for i, match in enumerate(re.finditer(pattern, text, re.DOTALL)):
                placeholder = f"__ENV_{env.upper()}_{i}__"
                placeholder_map[placeholder] = match.group(0)
                text = text.replace(match.group(0), placeholder, 1)
        
        for i, pattern in enumerate(self.PROTECTED_COMMANDS):
            for j, match in enumerate(re.finditer(pattern, text)):
                placeholder = f"__CMD_{i}_{j}__"
                placeholder_map[placeholder] = match.group(0)
                text = text.replace(match.group(0), placeholder, 1)
        
        return text, placeholder_map
    
    def restore_placeholders(self, text: str, placeholder_map: Dict[str, str]) -> str:
        """Restore placeholders with original LaTeX content."""
        for placeholder, original in placeholder_map.items():
            text = text.replace(placeholder, original)
        return text


class PaperHumanizer:
    """Main class to humanize academic papers section by section."""
    
    SUPPORTED_APIS = {
        'huggingface': HuggingFaceHumanizer,  # RECOMMENDED - best quality, free
        'ollama': OllamaLocalAPI,
        'rulebased': RuleBasedHumanizer,
        'webscrape': WebScraperHumanizer,
        'apify': ApifyHumanizer,
    }
    
    SKIP_SECTIONS = [
        'Acknowledgments', 'Acknowledgement', 'References', 
        'Bibliography', 'Appendix', 'Notation'
    ]
    
    def __init__(self, api_provider: str, 
                 model: str = "pegasus",
                 api_key: str = None,
                 delay_between_requests: float = 2.0,
                 max_retries: int = 3,
                 force_cpu: bool = False):
        
        if api_provider.lower() not in self.SUPPORTED_APIS:
            raise ValueError(f"Unsupported API: {api_provider}. "
                           f"Supported: {list(self.SUPPORTED_APIS.keys())}")
        
        self.api_provider = api_provider.lower()
        
        # Initialize the appropriate API
        if self.api_provider == 'huggingface':
            self.api = HuggingFaceHumanizer(model_name=model, force_cpu=force_cpu)
        elif self.api_provider == 'ollama':
            self.api = OllamaLocalAPI(model=model)
        elif self.api_provider == 'apify':
            self.api = ApifyHumanizer(api_token=api_key)
        elif self.api_provider == 'webscrape':
            self.api = WebScraperHumanizer()
        else:
            self.api = RuleBasedHumanizer()
        
        self.delay = delay_between_requests
        self.max_retries = max_retries
        self.parser = None
        self.results: Dict[str, HumanizationResult] = {}
    
    def should_skip_section(self, section: Section) -> bool:
        """Check if a section should be skipped."""
        for skip_pattern in self.SKIP_SECTIONS:
            if skip_pattern.lower() in section.name.lower():
                return True
        return False
    
    def humanize_section(self, section: Section) -> HumanizationResult:
        """Humanize a single section with retries."""
        if self.should_skip_section(section):
            logger.info(f"Skipping section: {section.name}")
            return HumanizationResult(
                success=True,
                original_text=section.content,
                humanized_text=section.content,
                error_message="Skipped - technical/reference section"
            )
        
        text_to_humanize, placeholder_map = self.parser.extract_text_for_humanization(section.content)
        
        clean_text = re.sub(r'__\w+__', '', text_to_humanize).strip()
        if len(clean_text) < 100:
            logger.info(f"Section '{section.name}' too short, keeping original")
            return HumanizationResult(
                success=True,
                original_text=section.content,
                humanized_text=section.content,
                error_message="Section too short"
            )
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Humanizing section: {section.name} (attempt {attempt + 1}/{self.max_retries})")
                result = self.api.humanize(text_to_humanize)
                
                if result.success:
                    humanized_content = self.parser.restore_placeholders(
                        result.humanized_text, 
                        placeholder_map
                    )
                    result.humanized_text = humanized_content
                    return result
                else:
                    logger.warning(f"Attempt {attempt + 1} failed: {result.error_message}")
                    time.sleep(self.delay * (attempt + 1))
                    
            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1}: {e}")
                time.sleep(self.delay * (attempt + 1))
        
        logger.error(f"All retries failed for section: {section.name}")
        return HumanizationResult(
            success=False,
            original_text=section.content,
            humanized_text=section.content,
            error_message="All retries exhausted"
        )
    
    def humanize_paper(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Humanize the entire paper section by section."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.parser = LaTeXParser(content)
        sections = self.parser.extract_sections()
        
        logger.info(f"Found {len(sections)} sections in the paper")
        logger.info(f"Using API provider: {self.api.get_provider_name()}")
        
        total_sections = len(sections)
        humanized_content = content
        
        for i, section in enumerate(sections, 1):
            logger.info(f"\nProcessing section {i}/{total_sections}: {section.name}")
            
            result = self.humanize_section(section)
            self.results[section.name] = result
            
            if result.success and result.humanized_text != section.content:
                original_section = section.raw_header + section.content
                new_section = section.raw_header + result.humanized_text
                humanized_content = humanized_content.replace(original_section, new_section, 1)
                logger.info(f"✓ Section '{section.name}' humanized successfully")
            else:
                logger.warning(f"✗ Section '{section.name}' kept original: {result.error_message}")
            
            if i < total_sections:
                time.sleep(self.delay)
        
        if output_path is None:
            output_path = str(input_path.stem) + "_humanized.tex"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(humanized_content)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Humanization complete! Output saved to: {output_path}")
        logger.info(f"Sections processed: {total_sections}")
        logger.info(f"Successful: {sum(1 for r in self.results.values() if r.success)}")
        logger.info(f"Failed: {sum(1 for r in self.results.values() if not r.success)}")
        
        return output_path
    
    def generate_report(self) -> str:
        """Generate a summary report."""
        report = []
        report.append("=" * 60)
        report.append("HUMANIZATION REPORT")
        report.append("=" * 60)
        report.append(f"API Provider: {self.api.get_provider_name()}")
        report.append("")
        
        for section_name, result in self.results.items():
            status = "✓ SUCCESS" if result.success else "✗ FAILED"
            report.append(f"{status}: {section_name}")
            if result.error_message and "Skipped" in result.error_message:
                report.append(f"    └── {result.error_message}")
            elif not result.success:
                report.append(f"    └── Error: {result.error_message}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Humanize academic papers to bypass AI detection (GPTZero) - FREE VERSION',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FREE API Options (ALL COMPLETELY FREE):
  huggingface - RECOMMENDED: Local PEGASUS/T5 paraphrase models via HuggingFace
  ollama      - Local LLM (Llama3, Mistral) - requires Ollama installation
  rulebased   - Rule-based transformations - No API needed, basic but works
  webscrape   - Scrapes free humanizer websites - Slower but free
  apify       - $5 free credits per month - Sign up at apify.com

Examples:
  # RECOMMENDED: Using HuggingFace PEGASUS (best quality, free)
  python humanize_paper.py lam_main_latest.tex --api huggingface --model pegasus
  
  # Using T5 paraphraser (smaller model, ~900MB)
  python humanize_paper.py lam_main_latest.tex --api huggingface --model t5
  
  # Using rule-based (no downloads needed, instant)
  python humanize_paper.py lam_main_latest.tex --api rulebased

HuggingFace Models (--model option):
  pegasus  - tuner007/pegasus_paraphrase (~2GB) - Best quality
  t5       - humarin/chatgpt_paraphraser_on_T5_base (~900MB) - Smaller, faster

First Run: Model will be downloaded automatically (~2GB for pegasus)
        """
    )
    
    parser.add_argument('input', help='Path to the input LaTeX file')
    parser.add_argument('--api', '-a', 
                        choices=['huggingface', 'ollama', 'rulebased', 'webscrape', 'apify'],
                        default='huggingface',
                        help='API provider to use (default: huggingface)')
    parser.add_argument('--model', '-m',
                        default='pegasus',
                        help='Model: pegasus (best), t5 (smaller), or Ollama model name')
    parser.add_argument('--api-key', '-k', 
                        help='API key for Apify (or set APIFY_TOKEN env var)')
    parser.add_argument('--output', '-o', 
                        help='Output file path (default: input_humanized.tex)')
    parser.add_argument('--delay', '-d', 
                        type=float, default=2.0,
                        help='Delay between requests in seconds (default: 2.0)')
    parser.add_argument('--max-retries', '-r',
                        type=int, default=3,
                        help='Maximum retries per section (default: 3)')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='Force CPU mode (slower but more stable)')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        humanizer = PaperHumanizer(
            api_provider=args.api,
            model=args.model,
            api_key=args.api_key,
            delay_between_requests=args.delay,
            max_retries=args.max_retries,
            force_cpu=args.cpu
        )
        
        output_path = humanizer.humanize_paper(args.input, args.output)
        
        report = humanizer.generate_report()
        print(report)
        
        report_path = Path(output_path).stem + "_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
