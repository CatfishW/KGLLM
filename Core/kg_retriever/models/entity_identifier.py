"""
SOTA Entity Identifier for Knowledge Graph Question Answering.

Combines zero-shot Named Entity Recognition with bi-encoder entity linking:
1. GLiNER for zero-shot NER (identifies mentions of entities)
2. Bi-Encoder for entity linking (links mentions to KB entities)
3. FAISS for fast approximate nearest neighbor search

Usage:
    model = EntityIdentifierModel.load_from_checkpoint("path/to/checkpoint.ckpt")
    results = model.identify("What language do Jamaican people speak?")
    # Returns: [LinkedEntity(mention="Jamaica", entity_id="m.03_r3", score=0.95)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
import numpy as np
import json
import re
from pathlib import Path
import logging

# Optional imports
try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EntityMention:
    """A recognized entity mention in text."""
    text: str
    start: int
    end: int
    label: str = "ENTITY"
    score: float = 1.0


@dataclass
class LinkedEntity:
    """An entity mention linked to a KB entity."""
    mention: str
    entity_id: str  # Freebase MID or Wikidata QID
    entity_name: str
    score: float
    label: str = "ENTITY"
    start: int = 0
    end: int = 0


class MentionEncoder(nn.Module):
    """
    Encodes entity mentions with their context for linking.
    
    Uses a pre-trained transformer to encode the mention text along with
    surrounding context from the question.
    """
    
    def __init__(
        self,
        encoder_name: str = "BAAI/bge-small-en-v1.5",
        hidden_size: int = 384,
        output_size: int = 256,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        encoder_hidden = self.encoder.config.hidden_size
        
        # Projection head for mention embeddings
        self.projection = nn.Sequential(
            nn.Linear(encoder_hidden, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size),
        )
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode mentions/questions.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            mention_mask: [batch_size, seq_len] - 1s for mention tokens
            
        Returns:
            embeddings: [batch_size, output_size]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        hidden_states = outputs.last_hidden_state  # [B, L, H]
        
        if mention_mask is not None:
            # Pool only mention tokens
            mention_mask = mention_mask.unsqueeze(-1).float()
            mention_sum = (hidden_states * mention_mask).sum(dim=1)
            mention_count = mention_mask.sum(dim=1).clamp(min=1)
            pooled = mention_sum / mention_count
        else:
            # Mean pooling over all tokens
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        return self.projection(pooled)


class EntityEncoder(nn.Module):
    """
    Encodes KB entities (name + description) for linking.
    """
    
    def __init__(
        self,
        encoder_name: str = "BAAI/bge-small-en-v1.5",
        hidden_size: int = 384,
        output_size: int = 256,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        encoder_hidden = self.encoder.config.hidden_size
        
        # Projection head for entity embeddings
        self.projection = nn.Sequential(
            nn.Linear(encoder_hidden, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size),
        )
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode entity names/descriptions.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            embeddings: [batch_size, output_size]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling
        pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        return self.projection(pooled)


class EntityIdentifierModel(pl.LightningModule):
    """
    SOTA Entity Identifier with bi-encoder linking.
    
    Architecture:
    1. NER: GLiNER for zero-shot entity recognition OR rule-based fallback
    2. Linking: Bi-encoder with contrastive learning for entity disambiguation
    3. Index: FAISS for fast approximate nearest neighbor search
    """
    
    # Demonym patterns for entity recognition
    DEMONYMS = {
        'jamaican': ('Jamaica', 'm.03_r3'), 'american': ('United States', 'm.09c7w0'),
        'british': ('United Kingdom', 'm.07ssc'), 'french': ('France', 'm.0f8l9c'),
        'german': ('Germany', 'm.0345h'), 'italian': ('Italy', 'm.03rjj'),
        'spanish': ('Spain', 'm.06mkj'), 'chinese': ('China', 'm.0d05w3'),
        'japanese': ('Japan', 'm.03_3d'), 'korean': ('Korea', 'm.0ctw_b'),
        'russian': ('Russia', 'm.06bnz'), 'indian': ('India', 'm.03rk0'),
        'canadian': ('Canada', 'm.0d060g'), 'australian': ('Australia', 'm.0chghy'),
        'mexican': ('Mexico', 'm.0b90_r'), 'brazilian': ('Brazil', 'm.015fr'),
        'swedish': ('Sweden', 'm.0d0vqn'), 'indonesian': ('Indonesia', 'm.03ryn'),
    }
    
    # Question words to filter
    QUESTION_WORDS = {
        'what', 'who', 'where', 'when', 'which', 'how', 'why', 'whom',
        'whose', 'does', 'did', 'do', 'is', 'are', 'was', 'were',
        'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for',
    }
    
    def __init__(
        self,
        encoder_name: str = "BAAI/bge-small-en-v1.5",
        hidden_size: int = 384,
        embedding_size: int = 256,
        use_gliner: bool = True,
        gliner_model: str = "urchade/gliner_medium-v2.1",
        entity_index_path: Optional[str] = None,
        freeze_encoder: bool = False,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        temperature: float = 0.05,
        max_seq_length: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize encoders
        self.mention_encoder = MentionEncoder(
            encoder_name=encoder_name,
            hidden_size=hidden_size,
            output_size=embedding_size,
            freeze_encoder=freeze_encoder,
        )
        
        self.entity_encoder = EntityEncoder(
            encoder_name=encoder_name,
            hidden_size=hidden_size,
            output_size=embedding_size,
            freeze_encoder=freeze_encoder,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        # GLiNER for NER (optional)
        self.gliner = None
        if use_gliner and GLINER_AVAILABLE:
            try:
                self.gliner = GLiNER.from_pretrained(gliner_model)
                logger.info(f"Loaded GLiNER model: {gliner_model}")
            except Exception as e:
                logger.warning(f"Failed to load GLiNER: {e}")
        
        # Entity index for fast search
        self.entity_index = None
        self.entity_id_map: Dict[int, str] = {}  # FAISS index -> entity_id
        self.entity_name_map: Dict[str, str] = {}  # entity_id -> entity_name
        
        if entity_index_path:
            self.load_entity_index(entity_index_path)
        
        # Training settings
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.temperature = temperature
        self.max_seq_length = max_seq_length
    
    def load_entity_index(self, index_path: str) -> None:
        """Load pre-built FAISS entity index."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping index loading")
            return
        
        index_file = Path(index_path) / "entity.index"
        id_map_file = Path(index_path) / "entity_id_map.json"
        name_map_file = Path(index_path) / "entity_names.json"
        
        if index_file.exists():
            self.entity_index = faiss.read_index(str(index_file))
            logger.info(f"Loaded FAISS index with {self.entity_index.ntotal} entities")
        
        if id_map_file.exists():
            with open(id_map_file, 'r') as f:
                self.entity_id_map = {int(k): v for k, v in json.load(f).items()}
        
        if name_map_file.exists():
            with open(name_map_file, 'r') as f:
                self.entity_name_map = json.load(f)
    
    def build_entity_index(
        self,
        entities: List[Dict[str, str]],
        batch_size: int = 256,
    ) -> None:
        """
        Build FAISS index from entity list.
        
        Args:
            entities: List of {"id": "m.xxx", "name": "Entity Name", "description": "..."}
            batch_size: Batch size for encoding
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, cannot build index")
            return
        
        logger.info(f"Building entity index for {len(entities)} entities...")
        
        all_embeddings = []
        self.entity_id_map = {}
        self.entity_name_map = {}
        
        self.entity_encoder.eval()
        
        with torch.no_grad():
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i+batch_size]
                
                # Encode entity names
                texts = [e.get("name", "") + ". " + e.get("description", "") for e in batch]
                encoded = self.tokenizer(
                    texts,
                    max_length=self.max_seq_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                
                if next(self.entity_encoder.parameters()).is_cuda:
                    encoded = {k: v.cuda() for k, v in encoded.items()}
                
                embeddings = self.entity_encoder(
                    encoded["input_ids"],
                    encoded["attention_mask"],
                )
                
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                all_embeddings.append(embeddings.cpu().numpy())
                
                # Store mappings
                for j, e in enumerate(batch):
                    idx = i + j
                    self.entity_id_map[idx] = e["id"]
                    self.entity_name_map[e["id"]] = e["name"]
                
                if (i + batch_size) % 10000 == 0:
                    logger.info(f"Encoded {i + batch_size}/{len(entities)} entities")
        
        # Build FAISS index
        embeddings_np = np.vstack(all_embeddings).astype('float32')
        
        # Use IVF index for large datasets
        d = embeddings_np.shape[1]
        if len(entities) > 100000:
            nlist = int(np.sqrt(len(entities)))
            quantizer = faiss.IndexFlatIP(d)
            self.entity_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.entity_index.train(embeddings_np)
            self.entity_index.add(embeddings_np)
        else:
            self.entity_index = faiss.IndexFlatIP(d)
            self.entity_index.add(embeddings_np)
        
        logger.info(f"Built FAISS index with {self.entity_index.ntotal} entities")
    
    def save_entity_index(self, output_path: str) -> None:
        """Save entity index to disk."""
        if self.entity_index is None:
            logger.warning("No entity index to save")
            return
        
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.entity_index, str(Path(output_path) / "entity.index"))
        
        with open(Path(output_path) / "entity_id_map.json", 'w') as f:
            json.dump({str(k): v for k, v in self.entity_id_map.items()}, f)
        
        with open(Path(output_path) / "entity_names.json", 'w') as f:
            json.dump(self.entity_name_map, f)
        
        logger.info(f"Saved entity index to {output_path}")
    
    def recognize_entities_gliner(self, question: str) -> List[EntityMention]:
        """Recognize entity mentions using GLiNER."""
        if self.gliner is None:
            return []
        
        # Entity types for KGQA
        labels = ["person", "location", "organization", "event", "work of art", "product"]
        
        try:
            entities = self.gliner.predict_entities(question, labels, threshold=0.3)
            
            return [
                EntityMention(
                    text=e["text"],
                    start=e["start"],
                    end=e["end"],
                    label=e["label"].upper(),
                    score=e["score"],
                )
                for e in entities
            ]
        except Exception as e:
            logger.warning(f"GLiNER prediction failed: {e}")
            return []
    
    def recognize_entities_rule(self, question: str) -> List[EntityMention]:
        """Fallback rule-based entity recognition."""
        mentions = []
        question_lower = question.lower()
        
        # 1. Demonym patterns
        for demonym, (name, _) in self.DEMONYMS.items():
            match = re.search(rf'\b{demonym}\b', question_lower)
            if match:
                mentions.append(EntityMention(
                    text=name,
                    start=match.start(),
                    end=match.end(),
                    label="LOCATION",
                    score=0.9,
                ))
        
        # 2. Capitalized sequences (proper nouns)
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        for match in re.finditer(pattern, question):
            text = match.group(1)
            if text.lower() not in self.QUESTION_WORDS:
                mentions.append(EntityMention(
                    text=text,
                    start=match.start(),
                    end=match.end(),
                    label="ENTITY",
                    score=0.8,
                ))
        
        # 3. Quoted phrases
        for match in re.finditer(r'"([^"]+)"', question):
            mentions.append(EntityMention(
                text=match.group(1),
                start=match.start(1),
                end=match.end(1),
                label="WORK_OF_ART",
                score=0.9,
            ))
        
        # 4. Possessives (e.g., "John's wife")
        for match in re.finditer(r"([A-Za-z]+(?:\s+[A-Za-z]+)*)'s", question):
            text = match.group(1)
            if text.lower() not in self.QUESTION_WORDS:
                mentions.append(EntityMention(
                    text=text.title(),
                    start=match.start(1),
                    end=match.end(1),
                    label="PERSON",
                    score=0.85,
                ))
        
        # Deduplicate by text
        seen = set()
        unique_mentions = []
        for m in sorted(mentions, key=lambda x: -x.score):
            if m.text not in seen:
                seen.add(m.text)
                unique_mentions.append(m)
        
        return unique_mentions
    
    def recognize_entities(self, question: str) -> List[EntityMention]:
        """
        Recognize entity mentions in a question.
        
        Uses GLiNER if available, falls back to rule-based recognition.
        """
        mentions = []
        
        # Try GLiNER first
        if self.gliner is not None:
            mentions = self.recognize_entities_gliner(question)
        
        # Fallback or supplement with rule-based
        if not mentions:
            mentions = self.recognize_entities_rule(question)
        else:
            # Add rule-based entities that weren't found by GLiNER
            rule_mentions = self.recognize_entities_rule(question)
            gliner_texts = {m.text.lower() for m in mentions}
            for rm in rule_mentions:
                if rm.text.lower() not in gliner_texts:
                    mentions.append(rm)
        
        return mentions
    
    def encode_mentions(
        self,
        question: str,
        mentions: List[EntityMention],
    ) -> torch.Tensor:
        """
        Encode mentions with context for linking.
        
        Returns:
            embeddings: [num_mentions, embedding_size]
        """
        if not mentions:
            return torch.zeros(0, self.hparams.embedding_size)
        
        # Encode each mention with question context
        texts = []
        for m in mentions:
            # Format: "[MENTION] entity_text [/MENTION] question_context"
            context = question[:m.start] + question[m.end:]
            text = f"{m.text}. {context}"
            texts.append(text)
        
        encoded = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        if next(self.mention_encoder.parameters()).is_cuda:
            encoded = {k: v.cuda() for k, v in encoded.items()}
        
        embeddings = self.mention_encoder(
            encoded["input_ids"],
            encoded["attention_mask"],
        )
        
        return F.normalize(embeddings, p=2, dim=-1)
    
    def link_entities(
        self,
        mentions: List[EntityMention],
        mention_embeddings: torch.Tensor,
        top_k: int = 5,
        candidate_entities: Optional[List[Dict[str, str]]] = None,
    ) -> List[LinkedEntity]:
        """
        Link mentions to KB entities using bi-encoder similarity.
        
        Args:
            mentions: List of entity mentions
            mention_embeddings: [num_mentions, embedding_size]
            top_k: Number of candidates per mention
            candidate_entities: Optional list of candidate entities to search
            
        Returns:
            List of LinkedEntity for each mention
        """
        if len(mentions) == 0:
            return []
        
        linked = []
        
        # Use FAISS index if available
        if self.entity_index is not None:
            embeddings_np = mention_embeddings.cpu().numpy().astype('float32')
            scores, indices = self.entity_index.search(embeddings_np, top_k)
            
            for i, mention in enumerate(mentions):
                # Check demonyms first for known mappings
                mention_lower = mention.text.lower()
                if mention_lower in self.DEMONYMS:
                    name, entity_id = self.DEMONYMS[mention_lower]
                    linked.append(LinkedEntity(
                        mention=mention.text,
                        entity_id=entity_id,
                        entity_name=name,
                        score=1.0,
                        label=mention.label,
                        start=mention.start,
                        end=mention.end,
                    ))
                    continue
                
                # Use FAISS results
                best_idx = indices[i][0]
                best_score = float(scores[i][0])
                
                if best_idx in self.entity_id_map:
                    entity_id = self.entity_id_map[best_idx]
                    entity_name = self.entity_name_map.get(entity_id, mention.text)
                    
                    linked.append(LinkedEntity(
                        mention=mention.text,
                        entity_id=entity_id,
                        entity_name=entity_name,
                        score=best_score,
                        label=mention.label,
                        start=mention.start,
                        end=mention.end,
                    ))
                else:
                    # Fallback: return mention as entity
                    linked.append(LinkedEntity(
                        mention=mention.text,
                        entity_id="",
                        entity_name=mention.text,
                        score=mention.score,
                        label=mention.label,
                        start=mention.start,
                        end=mention.end,
                    ))
        
        elif candidate_entities:
            # Search within provided candidates
            with torch.no_grad():
                # Encode candidates
                texts = [e.get("name", "") for e in candidate_entities]
                encoded = self.tokenizer(
                    texts,
                    max_length=self.max_seq_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                
                if mention_embeddings.is_cuda:
                    encoded = {k: v.cuda() for k, v in encoded.items()}
                
                candidate_embeddings = self.entity_encoder(
                    encoded["input_ids"],
                    encoded["attention_mask"],
                )
                candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)
                
                # Compute similarities
                similarities = torch.mm(mention_embeddings, candidate_embeddings.t())
                
                for i, mention in enumerate(mentions):
                    scores, indices = similarities[i].topk(min(top_k, len(candidate_entities)))
                    
                    best_idx = indices[0].item()
                    best_score = scores[0].item()
                    
                    entity = candidate_entities[best_idx]
                    linked.append(LinkedEntity(
                        mention=mention.text,
                        entity_id=entity.get("id", ""),
                        entity_name=entity.get("name", mention.text),
                        score=best_score,
                        label=mention.label,
                        start=mention.start,
                        end=mention.end,
                    ))
        
        else:
            # No index, return mentions with demonym lookups only
            for mention in mentions:
                mention_lower = mention.text.lower()
                if mention_lower in self.DEMONYMS:
                    name, entity_id = self.DEMONYMS[mention_lower]
                    linked.append(LinkedEntity(
                        mention=mention.text,
                        entity_id=entity_id,
                        entity_name=name,
                        score=1.0,
                        label=mention.label,
                        start=mention.start,
                        end=mention.end,
                    ))
                else:
                    linked.append(LinkedEntity(
                        mention=mention.text,
                        entity_id="",
                        entity_name=mention.text,
                        score=mention.score,
                        label=mention.label,
                        start=mention.start,
                        end=mention.end,
                    ))
        
        return linked
    
    @torch.no_grad()
    def identify(
        self,
        question: str,
        graph: Optional[List[List[str]]] = None,
        top_k: int = 5,
    ) -> List[LinkedEntity]:
        """
        Full entity identification pipeline.
        
        Args:
            question: Natural language question
            graph: Optional KG triples [(head, relation, tail), ...]
            top_k: Number of candidate entities per mention
            
        Returns:
            List of linked entities
        """
        self.eval()
        
        # Step 1: Recognize entity mentions
        mentions = self.recognize_entities(question)
        
        if not mentions:
            return []
        
        # Step 2: Encode mentions
        mention_embeddings = self.encode_mentions(question, mentions)
        
        # Step 3: Build candidate set from graph if provided
        candidate_entities = None
        if graph:
            # Extract unique entities from graph
            entities_set = set()
            for triple in graph:
                if len(triple) >= 3:
                    entities_set.add(triple[0])  # head
                    entities_set.add(triple[2])  # tail
            
            candidate_entities = [
                {"id": "", "name": e}
                for e in entities_set
                if e and not e.startswith("m.") and len(e) > 1
            ]
        
        # Step 4: Link mentions to entities
        linked = self.link_entities(
            mentions,
            mention_embeddings,
            top_k=top_k,
            candidate_entities=candidate_entities,
        )
        
        return linked
    
    @torch.no_grad()
    def identify_batch(
        self,
        questions: List[str],
        graphs: Optional[List[List[List[str]]]] = None,
    ) -> List[List[LinkedEntity]]:
        """Identify entities for a batch of questions."""
        results = []
        graphs = graphs or [None] * len(questions)
        
        for question, graph in zip(questions, graphs):
            results.append(self.identify(question, graph))
        
        return results
    
    def forward(
        self,
        mention_input_ids: torch.Tensor,
        mention_attention_mask: torch.Tensor,
        entity_input_ids: torch.Tensor,
        entity_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with contrastive loss.
        
        Args:
            mention_input_ids: [batch_size, seq_len]
            mention_attention_mask: [batch_size, seq_len]
            entity_input_ids: [batch_size, seq_len]
            entity_attention_mask: [batch_size, seq_len]
            labels: [batch_size] - positive entity indices
            
        Returns:
            Dictionary with loss and embeddings
        """
        # Encode mentions
        mention_embeddings = self.mention_encoder(
            mention_input_ids,
            mention_attention_mask,
        )
        mention_embeddings = F.normalize(mention_embeddings, p=2, dim=-1)
        
        # Encode entities
        entity_embeddings = self.entity_encoder(
            entity_input_ids,
            entity_attention_mask,
        )
        entity_embeddings = F.normalize(entity_embeddings, p=2, dim=-1)
        
        # Compute similarity scores
        scores = torch.mm(mention_embeddings, entity_embeddings.t()) / self.temperature
        
        # Contrastive loss (in-batch negatives)
        if labels is None:
            # Default: diagonal is positive
            labels = torch.arange(scores.size(0), device=scores.device)
        
        loss = F.cross_entropy(scores, labels)
        
        # Accuracy
        preds = scores.argmax(dim=-1)
        accuracy = (preds == labels).float().mean()
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "mention_embeddings": mention_embeddings,
            "entity_embeddings": entity_embeddings,
            "scores": scores,
        }
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        outputs = self.forward(
            mention_input_ids=batch["mention_input_ids"],
            mention_attention_mask=batch["mention_attention_mask"],
            entity_input_ids=batch["entity_input_ids"],
            entity_attention_mask=batch["entity_attention_mask"],
            labels=batch.get("labels"),
        )
        
        self.log("train/loss", outputs["loss"], prog_bar=True)
        self.log("train/accuracy", outputs["accuracy"], prog_bar=True)
        
        return outputs["loss"]
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        outputs = self.forward(
            mention_input_ids=batch["mention_input_ids"],
            mention_attention_mask=batch["mention_attention_mask"],
            entity_input_ids=batch["entity_input_ids"],
            entity_attention_mask=batch["entity_attention_mask"],
            labels=batch.get("labels"),
        )
        
        self.log("val/loss", outputs["loss"], prog_bar=True, sync_dist=True)
        self.log("val/accuracy", outputs["accuracy"], prog_bar=True, sync_dist=True)
        
        return outputs
    
    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Test step."""
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizer with warmup."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        
        # Linear warmup with cosine decay
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def demo():
    """Demo the entity identifier."""
    print("=" * 60)
    print("SOTA Entity Identifier Demo")
    print("=" * 60)
    
    # Initialize without GLiNER/FAISS for quick demo
    model = EntityIdentifierModel(
        encoder_name="BAAI/bge-small-en-v1.5",
        use_gliner=False,  # Disable for demo
    )
    
    questions = [
        "What language do Jamaican people speak?",
        "Who is Barack Obama's wife?",
        "Where was Albert Einstein born?",
        "What movies did Natalie Portman star in?",
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        
        # Recognize entities
        mentions = model.recognize_entities(question)
        print(f"   Mentions: {[(m.text, m.label, m.score) for m in mentions]}")
        
        # Full identify pipeline
        linked = model.identify(question)
        print(f"   Linked: {[(e.entity_name, e.entity_id, e.score) for e in linked]}")


if __name__ == "__main__":
    demo()
