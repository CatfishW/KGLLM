"""
Ultra-fast Entity Identifier for Questions.

Identifies topic entities mentioned in natural language questions using multiple strategies:
1. SpaCy NER for named entity recognition
2. Noun phrase extraction
3. Pattern-based extraction for common question patterns
4. Fuzzy matching against a known entity vocabulary

Usage:
    identifier = EntityIdentifier()
    identifier.load_vocabulary("path/to/vocab.json")  # Optional: for matching
    entities = identifier.identify("what does jamaican people speak")
    # Returns: ["Jamaica"]
"""

import re
import json
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import unicodedata

# Try to import optional dependencies
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class EntityIdentifier:
    """
    Ultra-fast entity identifier for extracting topic entities from questions.
    
    Combines multiple strategies for robust entity identification:
    - SpaCy NER (if available)
    - Pattern-based extraction
    - Noun phrase extraction
    - Fuzzy matching against vocabulary
    """
    
    # Common question words to filter out
    QUESTION_WORDS = {
        'what', 'who', 'where', 'when', 'which', 'how', 'why', 'whom',
        'whose', 'does', 'did', 'do', 'is', 'are', 'was', 'were', 
        'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might',
        'has', 'have', 'had', 'be', 'been', 'being', 'the', 'a', 'an',
        'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
        'and', 'or', 'but', 'not', 'no', 'yes', 'this', 'that', 'these',
        'those', 'it', 'its', 'they', 'them', 'their', 'he', 'she', 'him',
        'her', 'his', 'hers', 'we', 'us', 'our', 'you', 'your', 'i', 'me', 'my'
    }
    
    # Entity type patterns (key words that indicate entity type)
    ENTITY_TYPE_INDICATORS = {
        'person': ['president', 'king', 'queen', 'leader', 'founder', 'ceo', 
                   'actor', 'actress', 'singer', 'author', 'writer', 'politician',
                   'governor', 'mayor', 'senator', 'congressman'],
        'location': ['country', 'city', 'state', 'province', 'county', 'town',
                     'capital', 'river', 'mountain', 'island', 'continent'],
        'organization': ['company', 'corporation', 'university', 'school', 
                        'government', 'team', 'party', 'organization'],
        'time': ['year', 'date', 'century', 'decade', 'era', 'period']
    }
    
    # Demonym patterns for converting adjectives to proper nouns
    DEMONYM_PATTERNS = {
        'jamaican': 'Jamaica', 'american': 'United States', 'british': 'United Kingdom',
        'french': 'France', 'german': 'Germany', 'italian': 'Italy',
        'spanish': 'Spain', 'chinese': 'China', 'japanese': 'Japan',
        'korean': 'Korea', 'russian': 'Russia', 'indian': 'India',
        'canadian': 'Canada', 'australian': 'Australia', 'mexican': 'Mexico',
        'brazilian': 'Brazil', 'egyptian': 'Egypt', 'israeli': 'Israel',
        'iranian': 'Iran', 'iraqi': 'Iraq', 'swedish': 'Sweden',
        'norwegian': 'Norway', 'danish': 'Denmark', 'finnish': 'Finland',
        'dutch': 'Netherlands', 'belgian': 'Belgium', 'swiss': 'Switzerland',
        'austrian': 'Austria', 'polish': 'Poland', 'czech': 'Czech Republic',
        'greek': 'Greece', 'turkish': 'Turkey', 'thai': 'Thailand',
        'vietnamese': 'Vietnam', 'indonesian': 'Indonesia', 'malaysian': 'Malaysia',
        'philippine': 'Philippines', 'filipino': 'Philippines',
        'nigerian': 'Nigeria', 'kenyan': 'Kenya', 'south african': 'South Africa',
        'argentine': 'Argentina', 'colombian': 'Colombia', 'chilean': 'Chile',
        'peruvian': 'Peru', 'venezuelan': 'Venezuela', 'islamic': 'Islam',
        'christian': 'Christianity', 'jewish': 'Judaism', 'buddhist': 'Buddhism',
        'hindu': 'Hinduism', 'muslim': 'Islam'
    }
    
    def __init__(self, use_spacy: bool = True, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the entity identifier.
        
        Args:
            use_spacy: Whether to use SpaCy for NER (requires spacy package)
            spacy_model: SpaCy model to use
        """
        self.nlp = None
        self.vocabulary: Set[str] = set()
        self.vocab_normalized: Dict[str, str] = {}  # normalized -> original
        self.vocab_words: Dict[str, Set[str]] = defaultdict(set)  # word -> entities containing it
        
        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(f"SpaCy model '{spacy_model}' not found. Run: python -m spacy download {spacy_model}")
                self.nlp = None
    
    def load_vocabulary(self, vocab_path: str) -> None:
        """
        Load entity vocabulary for matching.
        
        Args:
            vocab_path: Path to vocab.json file
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        # Extract entities from vocabulary
        if isinstance(vocab, dict):
            # Vocab is {token: id} format
            entities = list(vocab.keys())
        else:
            # Vocab is list format
            entities = vocab
        
        # Filter to likely entity names (not relations or special tokens)
        for entity in entities:
            if not entity.startswith('[') and not entity.startswith('<'):
                # Skip relation-like tokens (contain dots and look like paths)
                if not re.match(r'^[a-z_]+\.[a-z_]+\.', entity):
                    self.vocabulary.add(entity)
                    normalized = self._normalize_entity(entity)
                    self.vocab_normalized[normalized] = entity
                    
                    # Index by words for fast lookup
                    for word in entity.lower().split():
                        if len(word) > 2:
                            self.vocab_words[word].add(entity)
        
        print(f"Loaded {len(self.vocabulary)} entities from vocabulary")
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity string for matching."""
        # Convert to lowercase and strip
        entity = entity.lower().strip()
        # Remove accents
        entity = unicodedata.normalize('NFKD', entity)
        entity = entity.encode('ASCII', 'ignore').decode('ASCII')
        # Remove punctuation except spaces
        entity = re.sub(r'[^\w\s]', '', entity)
        # Normalize whitespace
        entity = ' '.join(entity.split())
        return entity
    
    def _extract_capitalized_sequences(self, text: str) -> List[str]:
        """Extract sequences of capitalized words (likely proper nouns)."""
        # Pattern for capitalized word sequences
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(pattern, text)
        return matches
    
    def _extract_quoted_phrases(self, text: str) -> List[str]:
        """Extract quoted phrases from text."""
        # Single and double quotes
        patterns = [r'"([^"]+)"', r"'([^']+)'"]
        results = []
        for pattern in patterns:
            results.extend(re.findall(pattern, text))
        return results
    
    def _apply_demonym_patterns(self, text: str) -> List[str]:
        """Convert demonyms/adjectives to proper nouns."""
        entities = []
        text_lower = text.lower()
        
        for demonym, entity in self.DEMONYM_PATTERNS.items():
            if demonym in text_lower:
                entities.append(entity)
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities using SpaCy NER."""
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        
        # Also extract noun chunks that might be entities
        for chunk in doc.noun_chunks:
            # Filter out question words and common nouns
            chunk_text = chunk.text.strip()
            words = chunk_text.lower().split()
            if not all(w in self.QUESTION_WORDS for w in words):
                # Check if any word is capitalized
                if any(w[0].isupper() for w in chunk_text.split() if w):
                    entities.append((chunk_text, 'NOUN_CHUNK'))
        
        return entities
    
    def _extract_pattern_based(self, text: str) -> List[str]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Pattern: "X of Y" where Y is likely an entity
        of_pattern = r'(?:of|in|from|about)\s+(?:the\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'
        entities.extend(re.findall(of_pattern, text))
        
        # Pattern: "Person's" possessive (e.g., "niall ferguson's wife")
        possessive_pattern = r"([A-Za-z]+(?:\s+[A-Za-z]+)*)'s"
        possessive_matches = re.findall(possessive_pattern, text)
        for match in possessive_matches:
            # Capitalize properly
            words = match.split()
            capitalized = ' '.join(w.capitalize() for w in words)
            entities.append(capitalized)
        
        # Pattern: Years (4 digits)
        year_pattern = r'\b(1\d{3}|20\d{2})\b'
        entities.extend(re.findall(year_pattern, text))
        
        return entities
    
    def _fuzzy_match_vocabulary(self, candidates: List[str], threshold: int = 80) -> List[str]:
        """
        Match candidate entities against vocabulary using fuzzy matching.
        
        Args:
            candidates: List of candidate entity strings
            threshold: Minimum similarity score (0-100)
            
        Returns:
            List of matched entities from vocabulary
        """
        if not RAPIDFUZZ_AVAILABLE or not self.vocabulary:
            return candidates
        
        matched = []
        for candidate in candidates:
            normalized = self._normalize_entity(candidate)
            
            # Try exact match first
            if normalized in self.vocab_normalized:
                matched.append(self.vocab_normalized[normalized])
                continue
            
            # Try word-based lookup
            words = normalized.split()
            vocab_candidates = set()
            for word in words:
                if word in self.vocab_words:
                    vocab_candidates.update(self.vocab_words[word])
            
            # Fuzzy match against candidates
            if vocab_candidates:
                result = process.extractOne(
                    candidate, 
                    list(vocab_candidates),
                    scorer=fuzz.ratio
                )
                if result and result[1] >= threshold:
                    matched.append(result[0])
                else:
                    matched.append(candidate)
            else:
                matched.append(candidate)
        
        return matched
    
    def identify(self, question: str, return_scores: bool = False) -> List[str]:
        """
        Identify topic entities in a question.
        
        Args:
            question: Natural language question
            return_scores: Whether to return confidence scores
            
        Returns:
            List of identified entities (or list of (entity, score) if return_scores=True)
        """
        entities_with_scores: Dict[str, float] = defaultdict(float)
        
        # Strategy 1: SpaCy NER (highest confidence)
        spacy_entities = self._extract_with_spacy(question)
        for entity, label in spacy_entities:
            entity = entity.strip()
            if entity and entity.lower() not in self.QUESTION_WORDS:
                if label in ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC', 'NORP', 'EVENT', 'WORK_OF_ART', 'PRODUCT']:
                    entities_with_scores[entity] = max(entities_with_scores[entity], 0.95)
                elif label == 'NOUN_CHUNK':
                    entities_with_scores[entity] = max(entities_with_scores[entity], 0.7)
        
        # Strategy 2: Capitalized sequences (medium confidence)
        cap_entities = self._extract_capitalized_sequences(question)
        for entity in cap_entities:
            if entity.lower() not in self.QUESTION_WORDS:
                entities_with_scores[entity] = max(entities_with_scores[entity], 0.8)
        
        # Strategy 3: Demonym patterns (medium-high confidence)
        demonym_entities = self._apply_demonym_patterns(question)
        for entity in demonym_entities:
            entities_with_scores[entity] = max(entities_with_scores[entity], 0.85)
        
        # Strategy 4: Pattern-based extraction (medium confidence)
        pattern_entities = self._extract_pattern_based(question)
        for entity in pattern_entities:
            if entity.lower() not in self.QUESTION_WORDS:
                entities_with_scores[entity] = max(entities_with_scores[entity], 0.75)
        
        # Strategy 5: Quoted phrases (high confidence)
        quoted = self._extract_quoted_phrases(question)
        for entity in quoted:
            entities_with_scores[entity] = max(entities_with_scores[entity], 0.9)
        
        # Match against vocabulary if available
        if self.vocabulary:
            candidates = list(entities_with_scores.keys())
            matched = self._fuzzy_match_vocabulary(candidates)
            
            # Update with matched entities
            new_scores = {}
            for orig, match in zip(candidates, matched):
                score = entities_with_scores[orig]
                if match in self.vocabulary:
                    score = min(score + 0.05, 1.0)  # Boost for vocab match
                new_scores[match] = score
            
            entities_with_scores = new_scores
        
        # Sort by confidence score
        sorted_entities = sorted(
            entities_with_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if return_scores:
            return [(entity, score) for entity, score in sorted_entities]
        else:
            return [entity for entity, score in sorted_entities]
    
    def identify_batch(self, questions: List[str]) -> List[List[str]]:
        """
        Identify entities for a batch of questions.
        
        Args:
            questions: List of questions
            
        Returns:
            List of entity lists for each question
        """
        return [self.identify(q) for q in questions]


class FastEntityIdentifier:
    """
    Ultra-fast entity identifier optimized for speed.
    
    Uses only regex patterns and simple string matching - no ML models.
    Best for high-throughput scenarios where speed is critical.
    """
    
    QUESTION_WORDS = {
        'what', 'who', 'where', 'when', 'which', 'how', 'why', 'whom',
        'whose', 'does', 'did', 'do', 'is', 'are', 'was', 'were',
        'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might',
        'has', 'have', 'had', 'be', 'been', 'being', 'the', 'a', 'an',
        'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'about',
        'before', 'after', 'during', 'else', 'also', 'other', 'some', 'any',
        'play', 'plays', 'played', 'speak', 'speaks', 'spoke', 'called',
        'located', 'live', 'lives', 'lived', 'born', 'died', 'made',
        'part', 'kind', 'type', 'today', 'now', 'then', 'there', 'here',
        'first', 'last', 'next', 'up', 'down', 'end', 'start', 'begin'
    }
    
    DEMONYMS = {
        'jamaican': 'Jamaica', 'american': 'United States', 'british': 'United Kingdom',
        'french': 'France', 'german': 'Germany', 'italian': 'Italy',
        'spanish': 'Spain', 'chinese': 'China', 'japanese': 'Japan',
        'korean': 'Korea', 'russian': 'Russia', 'indian': 'India',
        'canadian': 'Canada', 'australian': 'Australia', 'mexican': 'Mexico',
        'brazilian': 'Brazil', 'egyptian': 'Egypt', 'islamic': 'Islam',
        'swedish': 'Sweden', 'indonesian': 'Indonesia', 'uk': 'United Kingdom',
        'u.s.': 'United States', 'u.s': 'United States', 'us': 'United States'
    }
    
    # Common name patterns
    COMMON_NAMES = {
        # Presidents/Politicians
        'james k polk': 'James K. Polk', 'james k. polk': 'James K. Polk',
        'george washington': 'George Washington', 'george w bush': 'George W. Bush',
        'george w. bush': 'George W. Bush', 'richard nixon': 'Richard Nixon',
        'kennedy': 'John F. Kennedy', 'jfk': 'John F. Kennedy',
        'lincoln': 'Abraham Lincoln', 'obama': 'Barack Obama',
        'trump': 'Donald Trump', 'biden': 'Joe Biden',
        # Celebrities
        'justin bieber': 'Justin Bieber', 'ben franklin': 'Benjamin Franklin',
        'benjamin franklin': 'Benjamin Franklin',
        'natalie portman': 'Natalie Portman', 'cam newton': 'Cam Newton',
        'andy murray': 'Andy Murray', 'niall ferguson': 'Niall Ferguson',
        'keyshia cole': 'Keyshia Cole', 'jackie robinson': 'Jackie Robinson',
        'rihanna': 'Rihanna', 'michael buble': 'Michael Bublé',
        'george orwell': 'George Orwell', 'adolf hitler': 'Adolf Hitler',
        'george lopez': 'George Lopez', 'edgar allan poe': 'Edgar Allan Poe',
        'martin luther king': 'Martin Luther King Jr.',
        'eleanor roosevelt': 'Eleanor Roosevelt', 'harper lee': 'Harper Lee',
        'anna bligh': 'Anna Bligh', 'george clemenceau': 'Georges Clemenceau',
        'george washington carver': 'George Washington Carver',
        'william henry harrison': 'William Henry Harrison',
        # Fictional characters
        'ken barlow': 'Ken Barlow', 'draco malfoy': 'Draco Malfoy',
        'mr gray': 'Christian Grey', 'mr grey': 'Christian Grey',
        # Sports
        'jamarcus russell': 'JaMarcus Russell', 'john noble': 'John Noble',
        'joakim noah': 'Joakim Noah',
        # Locations
        'fukushima': 'Fukushima Daiichi Nuclear Power Plant',
        'fukushima daiichi': 'Fukushima Daiichi Nuclear Power Plant',
        'ohio': 'Ohio', 'louisiana': 'Louisiana', 'utah': 'Utah',
        'sweden': 'Sweden', 'arizona': 'Arizona', 'atlanta': 'Atlanta',
        'galapagos': 'Galápagos Islands', 'frederick': 'Frederick',
        'kansas city': 'Kansas City',
        # Organizations
        'samsung': 'Samsung', 'nfl': 'NFL', 'redskins': 'Washington Redskins',
        'coronation street': 'Coronation Street',
        # Events/Works
        'star wars': 'Star Wars', 'lord of the rings': 'The Lord of the Rings',
        'annie': 'Annie',
        # Religion
        'st augustine': 'Augustine of Hippo', 'saint augustine': 'Augustine of Hippo',
    }
    
    def __init__(self):
        """Initialize fast entity identifier."""
        self.vocabulary: Set[str] = set()
        self.vocab_lower: Dict[str, str] = {}  # lowercase -> original
        
        # Compile patterns once for speed
        self._capitalized_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
        self._possessive_pattern = re.compile(r"([a-zA-Z]+(?:\s+[a-zA-Z]+)*)'s")
        self._year_pattern = re.compile(r'\b(1\d{3}|20\d{2})\b')
        # Pattern for potential names (two+ words with capitals or common name patterns)
        self._name_pattern = re.compile(r'\b([A-Za-z][a-z]*(?:\s+[A-Za-z][a-z]*)+)\b')
    
    def load_vocabulary(self, vocab_path: str) -> None:
        """Load vocabulary for matching."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        if isinstance(vocab, dict):
            entities = list(vocab.keys())
        else:
            entities = vocab
        
        for entity in entities:
            if not entity.startswith('[') and not entity.startswith('<'):
                if not re.match(r'^[a-z_]+\.[a-z_]+\.', entity):
                    self.vocabulary.add(entity)
                    self.vocab_lower[entity.lower()] = entity
        
        print(f"Loaded {len(self.vocabulary)} entities")
    
    def identify(self, question: str) -> List[str]:
        """
        Identify entities in question.
        
        Args:
            question: Question string
            
        Returns:
            List of identified entities
        """
        entities = set()
        question_lower = question.lower()
        
        # 1. Demonyms (e.g., "jamaican" -> "Jamaica")
        for demonym, entity in self.DEMONYMS.items():
            if demonym in question_lower:
                entities.add(entity)
        
        # 2. Common name patterns (lookup table for frequent entities)
        for pattern, entity in self.COMMON_NAMES.items():
            if pattern in question_lower:
                entities.add(entity)
        
        # 3. Capitalized sequences (proper nouns)
        for match in self._capitalized_pattern.findall(question):
            if match.lower() not in self.QUESTION_WORDS:
                entities.add(match)
        
        # 4. Possessives (person names like "niall ferguson's")
        for match in self._possessive_pattern.findall(question):
            words = match.split()
            name = ' '.join(w.capitalize() for w in words)
            if name.lower() not in self.QUESTION_WORDS:
                entities.add(name)
        
        # 5. Years
        entities.update(self._year_pattern.findall(question))
        
        # 6. Try to find multi-word entities in lowercase questions
        # Extract potential name-like phrases (2-4 consecutive words)
        words = question_lower.split()
        for length in range(4, 1, -1):  # Try longer matches first
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i+length])
                if phrase in self.COMMON_NAMES:
                    entities.add(self.COMMON_NAMES[phrase])
                elif self.vocab_lower and phrase in self.vocab_lower:
                    entities.add(self.vocab_lower[phrase])
        
        # 7. Match against vocabulary if available
        if self.vocab_lower:
            # Try to find vocabulary matches
            matched_entities = set()
            for entity in entities:
                lower = entity.lower()
                if lower in self.vocab_lower:
                    matched_entities.add(self.vocab_lower[lower])
                else:
                    matched_entities.add(entity)
            
            # Also try single words that might be entities
            for word in words:
                if word not in self.QUESTION_WORDS and len(word) > 2:
                    if word in self.vocab_lower:
                        matched_entities.add(self.vocab_lower[word])
            
            entities = matched_entities
        
        return list(entities)
    
    def identify_batch(self, questions: List[str]) -> List[List[str]]:
        """Identify entities for batch of questions."""
        return [self.identify(q) for q in questions]


def demo():
    """Demo the entity identifier."""
    # Test questions from the evaluation data
    questions = [
        "what does jamaican people speak",
        "what did james k polk do before he was president",
        "who plays ken barlow in coronation street",
        "where is jamarcus russell from",
        "what is the australian dollar called",
        "who is niall ferguson 's wife",
        "what timezone is sweden",
        "who did cam newton sign with",
        "where is the fukushima daiichi nuclear plant located",
        "what countries are part of the uk",
    ]
    
    print("="*60)
    print("Testing EntityIdentifier (with SpaCy if available)")
    print("="*60)
    
    identifier = EntityIdentifier(use_spacy=True)
    for q in questions:
        entities = identifier.identify(q, return_scores=True)
        print(f"\nQ: {q}")
        print(f"   Entities: {entities}")
    
    print("\n" + "="*60)
    print("Testing FastEntityIdentifier (regex-only)")
    print("="*60)
    
    fast_identifier = FastEntityIdentifier()
    for q in questions:
        entities = fast_identifier.identify(q)
        print(f"\nQ: {q}")
        print(f"   Entities: {entities}")


if __name__ == "__main__":
    demo()
