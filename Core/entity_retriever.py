"""
Ultra-fast Entity Retriever for Knowledge Graph Question Answering.

Given predicted relation paths (like "location.country.languages_spoken -> language.human_language.region")
and a knowledge graph, retrieves the actual answer entities by traversing the graph.

Key features:
- O(1) edge lookup via hash-based indexing
- Batch processing for multiple paths
- Efficient caching for repeated queries
- Support for both forward and backward traversal

Usage:
    retriever = EntityRetriever()
    retriever.build_index(graph_triples)
    
    # Given question entity and relation path
    answers = retriever.retrieve(
        start_entity="Jamaica",
        relation_path=["location.country.languages_spoken"]
    )
    # Returns: ["Jamaican English", "Jamaican Creole English Language"]
"""

import json
from typing import List, Dict, Set, Optional, Tuple, Union
from collections import defaultdict
import time
import numpy as np


class EntityRetriever:
    """
    Ultra-fast entity retriever using hash-based graph indexing.
    
    Builds multiple indexes for O(1) lookup:
    - Forward index: entity -> relation -> set of object entities
    - Backward index: entity -> relation -> set of subject entities  
    - Relation index: relation -> list of (subject, object) pairs
    """
    
    def __init__(self):
        """Initialize the retriever."""
        # Forward index: subject -> relation -> set(objects)
        self.forward_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        # Backward index: object -> relation -> set(subjects)
        self.backward_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        # Relation index: relation -> list of (subject, object)
        self.relation_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        
        # Entity normalization map (lowercase -> original)
        self.entity_norm_map: Dict[str, Set[str]] = defaultdict(set)
        
        # All entities
        self.all_entities: Set[str] = set()
        
        # Cache for repeated queries
        self._cache: Dict[str, List[str]] = {}
        self._cache_max_size = 10000
        
        # Statistics
        self.num_triples = 0
        self.num_entities = 0
        self.num_relations = 0
    
    def build_index(self, triples: List[Union[List, Tuple]]) -> None:
        """
        Build graph index from triples.
        
        Args:
            triples: List of [subject, relation, object] triples
        """
        start_time = time.time()
        
        relations = set()
        
        for triple in triples:
            if len(triple) < 3:
                continue
                
            subj, rel, obj = triple[0], triple[1], triple[2]
            
            # Build forward index
            self.forward_index[subj][rel].add(obj)
            
            # Build backward index
            self.backward_index[obj][rel].add(subj)
            
            # Build relation index
            self.relation_index[rel].append((subj, obj))
            
            # Track entities
            self.all_entities.add(subj)
            self.all_entities.add(obj)
            
            # Build normalization map
            self.entity_norm_map[subj.lower()].add(subj)
            self.entity_norm_map[obj.lower()].add(obj)
            
            relations.add(rel)
        
        self.num_triples = len(triples)
        self.num_entities = len(self.all_entities)
        self.num_relations = len(relations)
        
        elapsed = time.time() - start_time
        print(f"Built index: {self.num_triples:,} triples, {self.num_entities:,} entities, "
              f"{self.num_relations:,} relations in {elapsed:.2f}s")
    
    def normalize_entity(self, entity: str) -> Set[str]:
        """
        Find matching entities in graph (case-insensitive).
        
        Args:
            entity: Entity string to match
            
        Returns:
            Set of matching entities from the graph
        """
        # Try exact match first
        if entity in self.all_entities:
            return {entity}
        
        # Try lowercase match
        lower = entity.lower()
        if lower in self.entity_norm_map:
            return self.entity_norm_map[lower]
        
        # No match found
        return set()
    
    def traverse_relation(
        self,
        entities: Set[str],
        relation: str,
        direction: str = "forward"
    ) -> Set[str]:
        """
        Traverse one relation edge from given entities.
        
        Args:
            entities: Set of starting entities
            relation: Relation to traverse
            direction: "forward" or "backward"
            
        Returns:
            Set of reached entities
        """
        result = set()
        
        if direction == "forward":
            for entity in entities:
                if entity in self.forward_index:
                    result.update(self.forward_index[entity].get(relation, set()))
        else:
            for entity in entities:
                if entity in self.backward_index:
                    result.update(self.backward_index[entity].get(relation, set()))
        
        return result
    
    def retrieve(
        self,
        start_entity: str,
        relation_path: List[str],
        direction: str = "forward",
        max_results: int = 100
    ) -> List[str]:
        """
        Retrieve answer entities by traversing relation path.
        
        Args:
            start_entity: Starting entity
            relation_path: List of relations to traverse
            direction: "forward" for subject->object, "backward" for object->subject
            max_results: Maximum number of results to return
            
        Returns:
            List of answer entities
        """
        # Check cache
        cache_key = f"{start_entity}|{'->'.join(relation_path)}|{direction}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Normalize start entity
        current_entities = self.normalize_entity(start_entity)
        if not current_entities:
            return []
        
        # Traverse path
        for relation in relation_path:
            if not current_entities:
                break
            current_entities = self.traverse_relation(current_entities, relation, direction)
        
        result = list(current_entities)[:max_results]
        
        # Cache result
        if len(self._cache) < self._cache_max_size:
            self._cache[cache_key] = result
        
        return result
    
    def retrieve_with_path_string(
        self,
        start_entity: str,
        path_string: str,
        max_results: int = 100
    ) -> List[str]:
        """
        Retrieve entities using a path string (e.g., "rel1 -> rel2 -> rel3").
        
        Args:
            start_entity: Starting entity
            path_string: Path string with " -> " separator
            max_results: Maximum results
            
        Returns:
            List of answer entities
        """
        # Parse path string
        relations = [r.strip() for r in path_string.split('->')]
        relations = [r for r in relations if r]
        
        return self.retrieve(start_entity, relations, max_results=max_results)
    
    def retrieve_batch(
        self,
        queries: List[Tuple[str, List[str]]],
        direction: str = "forward",
        max_results: int = 100
    ) -> List[List[str]]:
        """
        Retrieve entities for multiple queries.
        
        Args:
            queries: List of (start_entity, relation_path) tuples
            direction: Traversal direction
            max_results: Max results per query
            
        Returns:
            List of result lists
        """
        return [
            self.retrieve(entity, path, direction, max_results)
            for entity, path in queries
        ]
    
    def retrieve_all_paths(
        self,
        start_entity: str,
        path_strings: List[str],
        max_results_per_path: int = 50
    ) -> Dict[str, List[str]]:
        """
        Retrieve entities for multiple paths from same start entity.
        
        Args:
            start_entity: Starting entity
            path_strings: List of path strings
            max_results_per_path: Max results per path
            
        Returns:
            Dictionary mapping path string to results
        """
        results = {}
        for path in path_strings:
            results[path] = self.retrieve_with_path_string(
                start_entity, path, max_results_per_path
            )
        return results
    
    def find_entities_for_relation(
        self,
        relation: str,
        position: str = "object"
    ) -> List[str]:
        """
        Find all entities in a specific position for a relation.
        
        Args:
            relation: Relation name
            position: "subject" or "object"
            
        Returns:
            List of entities
        """
        if relation not in self.relation_index:
            return []
        
        if position == "object":
            return list(set(obj for _, obj in self.relation_index[relation]))
        else:
            return list(set(subj for subj, _ in self.relation_index[relation]))
    
    def get_neighbors(
        self,
        entity: str,
        direction: str = "both",
        relations: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Set[str]]]:
        """
        Get all neighbors of an entity.
        
        Args:
            entity: Entity to query
            direction: "forward", "backward", or "both"
            relations: Optional filter for specific relations
            
        Returns:
            Dictionary with "forward" and/or "backward" keys, 
            each mapping relation -> set of entities
        """
        result = {}
        
        if direction in ("forward", "both") and entity in self.forward_index:
            forward = self.forward_index[entity]
            if relations:
                result["forward"] = {r: forward[r] for r in relations if r in forward}
            else:
                result["forward"] = dict(forward)
        
        if direction in ("backward", "both") and entity in self.backward_index:
            backward = self.backward_index[entity]
            if relations:
                result["backward"] = {r: backward[r] for r in relations if r in backward}
            else:
                result["backward"] = dict(backward)
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._cache.clear()


class MultiHopEntityRetriever(EntityRetriever):
    """
    Extended retriever with multi-hop reasoning capabilities.
    
    Supports:
    - Automatic bidirectional path exploration
    - Path disambiguation
    - Beam search for multi-hop paths
    """
    
    def retrieve_bidirectional(
        self,
        start_entity: str,
        relation_path: List[str],
        max_results: int = 100
    ) -> List[str]:
        """
        Try both forward and backward traversal.
        
        Some paths work better in reverse direction. This method
        tries both and returns combined results.
        
        Args:
            start_entity: Starting entity
            relation_path: List of relations
            max_results: Maximum results
            
        Returns:
            Combined list of entities from both directions
        """
        # Try forward
        forward_results = self.retrieve(
            start_entity, relation_path, "forward", max_results
        )
        
        # Try backward with reversed path
        backward_results = self.retrieve(
            start_entity, relation_path[::-1], "backward", max_results
        )
        
        # Combine (forward results first as they're usually more relevant)
        combined = []
        seen = set()
        
        for entity in forward_results + backward_results:
            if entity not in seen:
                combined.append(entity)
                seen.add(entity)
                if len(combined) >= max_results:
                    break
        
        return combined
    
    def find_connecting_paths(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 3,
        max_paths: int = 10
    ) -> List[Dict]:
        """
        Find paths connecting two entities.
        
        Args:
            start_entity: Starting entity
            end_entity: Target entity
            max_depth: Maximum path length
            max_paths: Maximum number of paths to return
            
        Returns:
            List of path dictionaries with entities and relations
        """
        start_entities = self.normalize_entity(start_entity)
        end_entities = self.normalize_entity(end_entity)
        
        if not start_entities or not end_entities:
            return []
        
        paths = []
        
        # BFS to find paths
        from collections import deque
        
        # (current_entity, path_entities, path_relations)
        queue = deque()
        for se in start_entities:
            queue.append((se, [se], []))
        
        visited = {(e, tuple()) for e in start_entities}
        
        while queue and len(paths) < max_paths:
            current, path_entities, path_relations = queue.popleft()
            
            if len(path_relations) >= max_depth:
                continue
            
            # Explore forward edges
            if current in self.forward_index:
                for rel, objects in self.forward_index[current].items():
                    for obj in objects:
                        state = (obj, tuple(path_relations + [rel]))
                        if state not in visited:
                            visited.add(state)
                            
                            new_entities = path_entities + [obj]
                            new_relations = path_relations + [rel]
                            
                            if obj in end_entities:
                                paths.append({
                                    "entities": new_entities,
                                    "relations": new_relations,
                                    "path_string": " -> ".join(new_relations)
                                })
                            else:
                                queue.append((obj, new_entities, new_relations))
        
        return paths
    
    def rank_paths(
        self,
        paths: List[str],
        query_relations: Set[str]
    ) -> List[Tuple[str, float]]:
        """
        Rank paths by relevance to query relations.
        
        Args:
            paths: List of path strings
            query_relations: Set of relations expected in the answer
            
        Returns:
            List of (path, score) tuples sorted by score descending
        """
        scored_paths = []
        
        for path in paths:
            relations = set(r.strip() for r in path.split('->'))
            
            # Calculate overlap with query relations
            overlap = len(relations & query_relations)
            total = len(relations | query_relations)
            
            if total > 0:
                score = overlap / total
            else:
                score = 0.0
            
            scored_paths.append((path, score))
        
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        return scored_paths


class GraphRetrieverIndex:
    """
    Efficient index for retrieving subgraphs by relation patterns.
    
    Precomputes relation pattern -> entity mappings for fast retrieval.
    """
    
    def __init__(self):
        """Initialize the index."""
        # pattern -> list of (subject, objects) pairs
        self.pattern_index: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
        
        # Single relation -> objects index
        self.single_rel_objects: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.num_patterns = 0
    
    def build_from_retriever(
        self,
        retriever: EntityRetriever,
        max_hop: int = 4
    ) -> None:
        """
        Build pattern index from an entity retriever.
        
        Args:
            retriever: EntityRetriever with built index
            max_hop: Maximum path length to index
        """
        print("Building pattern index...")
        start_time = time.time()
        
        # Index single-hop patterns
        for rel, pairs in retriever.relation_index.items():
            for subj, obj in pairs:
                self.single_rel_objects[rel].add(obj)
            
            # Group by subject
            subj_to_objs = defaultdict(list)
            for subj, obj in pairs:
                subj_to_objs[subj].append(obj)
            
            for subj, objs in subj_to_objs.items():
                self.pattern_index[rel].append((subj, objs))
        
        self.num_patterns = len(self.pattern_index)
        
        elapsed = time.time() - start_time
        print(f"Built pattern index: {self.num_patterns} patterns in {elapsed:.2f}s")
    
    def get_objects_for_pattern(self, pattern: str) -> Set[str]:
        """
        Get all object entities for a single-hop pattern.
        
        Args:
            pattern: Relation pattern (single relation)
            
        Returns:
            Set of object entities
        """
        return self.single_rel_objects.get(pattern, set())
    
    def get_subjects_for_pattern(self, pattern: str) -> Set[str]:
        """
        Get all subject entities for a pattern.
        
        Args:
            pattern: Relation pattern
            
        Returns:
            Set of subject entities
        """
        if pattern in self.pattern_index:
            return set(subj for subj, _ in self.pattern_index[pattern])
        return set()


def demo():
    """Demo the entity retriever."""
    # Sample graph triples (simplified from evaluation data)
    triples = [
        ["Jamaica", "location.country.official_language", "Jamaican English"],
        ["Jamaica", "location.country.languages_spoken", "Jamaican Creole English Language"],
        ["Jamaica", "location.location.containedby", "Americas"],
        ["Jamaican Creole English Language", "language.human_language.region", "Americas"],
        ["Jamaica", "meteorology.cyclone_affected_area.cyclones", "Tropical Storm Keith"],
        ["Tropical Storm Keith", "meteorology.tropical_cyclone.affected_areas", "Jamaica"],
        ["Jamaica", "sports.sport_country.athletic_performances", "m.0b6562_"],
        ["m.0b6562_", "sports.competitor_competition_relationship.competitors", "Kerron Stewart"],
        ["Kerron Stewart", "people.person.nationality", "Jamaica"],
        ["James K. Polk", "government.politician.government_positions_held", "m.04j60kc"],
        ["m.04j60kc", "government.government_position_held.office_position_or_title", "United States Representative"],
        ["James K. Polk", "government.politician.government_positions_held", "m.04j5sk8"],
        ["m.04j5sk8", "government.government_position_held.office_position_or_title", "Governor of Tennessee"],
    ]
    
    print("="*60)
    print("Testing EntityRetriever")
    print("="*60)
    
    retriever = EntityRetriever()
    retriever.build_index(triples)
    
    # Test retrieval
    print("\n--- Test 1: Single-hop retrieval ---")
    results = retriever.retrieve("Jamaica", ["location.country.languages_spoken"])
    print(f"Jamaica -> location.country.languages_spoken: {results}")
    
    print("\n--- Test 2: Multi-hop retrieval ---")
    results = retriever.retrieve(
        "Jamaica", 
        ["location.location.containedby", "language.human_language.region"],
        direction="forward"
    )
    print(f"Jamaica -> containedby -> human_language.region: {results}")
    
    print("\n--- Test 3: Path string retrieval ---")
    path = "government.politician.government_positions_held -> government.government_position_held.office_position_or_title"
    results = retriever.retrieve_with_path_string("James K. Polk", path)
    print(f"James K. Polk with path '{path}': {results}")
    
    print("\n--- Test 4: Get neighbors ---")
    neighbors = retriever.get_neighbors("Jamaica", direction="forward")
    print(f"Jamaica neighbors: {len(neighbors.get('forward', {}))} relations")
    for rel, entities in list(neighbors.get('forward', {}).items())[:3]:
        print(f"  {rel}: {list(entities)[:2]}")
    
    print("\n--- Test 5: MultiHopEntityRetriever ---")
    multi_retriever = MultiHopEntityRetriever()
    multi_retriever.build_index(triples)
    
    # Find connecting paths
    paths = multi_retriever.find_connecting_paths("Jamaica", "Jamaican English", max_depth=2)
    print(f"Paths from Jamaica to Jamaican English:")
    for p in paths:
        print(f"  {p['path_string']}: {p['entities']}")


if __name__ == "__main__":
    demo()
