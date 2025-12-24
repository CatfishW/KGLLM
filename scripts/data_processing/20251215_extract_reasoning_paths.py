"""
Extract reasoning paths from webqsp_rog dataset.
For each question, finds paths in the knowledge graph that connect 
question entities to answer entities.

Output format for training:
{
    "id": str,
    "question": str,
    "answer": list[str],
    "q_entity": list[str],
    "a_entity": list[str],
    "reasoning_paths": list[dict]  # Each path has: entities, relations, triples
}
"""

import pandas as pd
import json
import os
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple
import numpy as np


def normalize_entity(entity: str) -> str:
    """Normalize entity for matching (lowercase, strip)."""
    if entity is None:
        return ""
    return str(entity).strip().lower()


def build_graph(triples: List[np.ndarray]) -> Tuple[Dict, Set[str], Dict[str, str], Dict[str, Set[str]]]:
    """
    Build adjacency lists for forward traversal plus normalization indexes.
    
    Returns:
        forward_graph: {entity: [(relation, target_entity), ...]}
        graph_entities: set of all entities appearing in triples
        entity_norm_map: {entity: normalized_entity}
        norm_lookup: {normalized_entity: {entities...}}
    """
    forward_graph = defaultdict(list)
    graph_entities: Set[str] = set()
    entity_norm_map: Dict[str, str] = {}
    norm_lookup: Dict[str, Set[str]] = defaultdict(set)

    def register_entity(entity: str) -> None:
        if entity not in entity_norm_map:
            normalized = normalize_entity(entity)
            entity_norm_map[entity] = normalized
            norm_lookup[normalized].add(entity)
        graph_entities.add(entity)
    
    for triple in triples:
        if len(triple) != 3:
            continue
        # Convert to strings to ensure hashability
        subj, rel, obj = str(triple[0]), str(triple[1]), str(triple[2])
        forward_graph[subj].append((rel, obj))
        register_entity(subj)
        register_entity(obj)
    
    return forward_graph, graph_entities, entity_norm_map, norm_lookup


def find_paths_bfs(
    forward_graph: Dict, 
    start_entities: Set[str], 
    target_entities: Set[str],
    max_depth: int = 3
) -> List[Dict]:
    """
    Find all paths from start_entities to target_entities using BFS.
    
    Returns list of paths, each path is a dict with:
        - entities: [e1, e2, ..., en]
        - relations: [r1, r2, ..., rn-1]
        - triples: [(e1, r1, e2), (e2, r2, e3), ...]
    """
    paths = []
    
    # BFS state: (current_entity, path_entities, path_relations)
    queue = deque((entity, [entity], []) for entity in start_entities)
    
    visited_paths = set()  # To avoid duplicate paths
    
    while queue:
        current, path_entities, path_relations = queue.popleft()
        
        # Check if we reached a target
        if current in target_entities and len(path_entities) > 1:
            # Build path representation
            path_key = tuple(path_entities) + tuple(path_relations)
            if path_key not in visited_paths:
                visited_paths.add(path_key)
                
                triples = []
                for i in range(len(path_relations)):
                    triples.append({
                        "subject": path_entities[i],
                        "relation": path_relations[i],
                        "object": path_entities[i + 1]
                    })
                
                paths.append({
                    "entities": path_entities,
                    "relations": path_relations,
                    "triples": triples,
                    "path_length": len(path_relations)
                })
        
        # Continue BFS if not too deep
        if len(path_relations) < max_depth:
            for relation, neighbor in forward_graph.get(current, []):
                if neighbor not in path_entities:  # Avoid cycles
                    new_entities = path_entities + [neighbor]
                    new_relations = path_relations + [relation]
                    queue.append((neighbor, new_entities, new_relations))
    
    return paths


def find_entity_matches(
    entity: str, 
    graph_entities: Set[str],
    entity_norm_map: Dict[str, str],
    norm_lookup: Dict[str, Set[str]]
) -> Set[str]:
    """
    Find matching entities in graph considering exact and partial matches.
    """
    matches = set()
    entity_norm = normalize_entity(entity)
    
    # First try exact match
    if entity in graph_entities:
        matches.add(entity)
    
    # Try normalized matching lookup
    matches.update(norm_lookup.get(entity_norm, set()))
    
    # Try partial matching (entity is contained in graph entity or vice versa)
    if not matches:
        for ge, ge_norm in entity_norm_map.items():
            if entity_norm in ge_norm or ge_norm in entity_norm:
                matches.add(ge)
    
    return matches


def to_string_list(val) -> List[str]:
    """Convert value to list of strings, handling various types."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, np.ndarray):
        return [str(x) for x in val.tolist()]
    if isinstance(val, list):
        return [str(x) for x in val]
    return [str(val)]


def extract_reasoning_paths_for_sample(row: Dict, max_depth: int = 3) -> Dict:
    """
    Extract reasoning paths for a single sample.
    """
    result = {
        "id": row["id"],
        "question": row["question"],
        "answer": to_string_list(row["answer"]),
        "q_entity": to_string_list(row["q_entity"]),
        "a_entity": to_string_list(row["a_entity"]),
        "reasoning_paths": [],
        "num_paths": 0,
        "has_valid_path": False
    }
    
    graph_triples = row["graph"]
    if isinstance(graph_triples, np.ndarray):
        graph_triples = graph_triples.tolist()
    
    if graph_triples is None or len(graph_triples) == 0:
        return result
    
    # Build graph and supporting indexes
    forward_graph, graph_entities, entity_norm_map, norm_lookup = build_graph(graph_triples)
    
    # Find matching entities for question and answer entities
    q_entity_matches = set()
    for qe in result["q_entity"]:
        q_entity_matches.update(find_entity_matches(qe, graph_entities, entity_norm_map, norm_lookup))
    
    a_entity_matches = set()
    for ae in result["a_entity"]:
        a_entity_matches.update(find_entity_matches(ae, graph_entities, entity_norm_map, norm_lookup))
    
    if not q_entity_matches or not a_entity_matches:
        return result
    
    # Find paths from question entities to answer entities
    paths = find_paths_bfs(forward_graph, q_entity_matches, a_entity_matches, max_depth)
    
    # Also try reverse direction (answer to question) and reverse the paths
    reverse_paths = find_paths_bfs(forward_graph, a_entity_matches, q_entity_matches, max_depth)
    for path in reverse_paths:
        # Reverse the path
        reversed_path = {
            "entities": path["entities"][::-1],
            "relations": path["relations"][::-1],
            "triples": [],
            "path_length": path["path_length"],
            "direction": "reversed"
        }
        for i in range(len(reversed_path["relations"])):
            reversed_path["triples"].append({
                "subject": reversed_path["entities"][i],
                "relation": reversed_path["relations"][i],
                "object": reversed_path["entities"][i + 1]
            })
        paths.append(reversed_path)
    
    result["reasoning_paths"] = paths
    result["num_paths"] = len(paths)
    result["has_valid_path"] = len(paths) > 0
    
    return result


def format_path_as_string(path: Dict) -> str:
    """Format a reasoning path as a human-readable string."""
    if not path or not path.get("triples"):
        return ""
    
    parts = []
    for triple in path["triples"]:
        parts.append(f"({triple['subject']}) --[{triple['relation']}]--> ({triple['object']})")
    
    return " | ".join(parts)


def format_path_as_relation_chain(path: Dict) -> str:
    """Format path as relation chain: rel1 -> rel2 -> rel3"""
    if not path or not path.get("relations"):
        return ""
    return " -> ".join(path["relations"])


def process_parquet_files(data_dir: str, split: str = "train") -> List[Dict]:
    """
    Process all parquet files for a given split.
    """
    all_results = []
    
    # Find all parquet files for the split
    parquet_files = [f for f in os.listdir(data_dir) if f.startswith(split) and f.endswith(".parquet")]
    
    print(f"Found {len(parquet_files)} parquet files for split '{split}'")
    
    for pf in sorted(parquet_files):
        print(f"\nProcessing {pf}...")
        df = pd.read_parquet(os.path.join(data_dir, pf))
        records = df.to_dict(orient="records")
        
        for idx, row in enumerate(records):
            result = extract_reasoning_paths_for_sample(row)
            all_results.append(result)
            
            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(records)} samples...")
    
    return all_results


def create_training_dataset(results: List[Dict]) -> List[Dict]:
    """
    Create the final training dataset with simplified format.
    
    Output format per sample:
    {
        "id": str,
        "question": str,
        "answer": str (joined if multiple),
        "reasoning_paths": [
            {
                "path_string": "entity1 --[rel1]--> entity2 --[rel2]--> entity3",
                "relation_chain": "rel1 -> rel2",
                "entities": [...],
                "relations": [...]
            }
        ]
    }
    """
    training_data = []
    
    for result in results:
        if not result["has_valid_path"]:
            continue
        
        sample = {
            "id": result["id"],
            "question": result["question"],
            "answer": ", ".join(result["answer"]) if isinstance(result["answer"], list) else result["answer"],
            "q_entity": result["q_entity"],
            "a_entity": result["a_entity"],
            "reasoning_paths": []
        }
        
        for path in result["reasoning_paths"]:
            path_entry = {
                "path_string": format_path_as_string(path),
                "relation_chain": format_path_as_relation_chain(path),
                "entities": path["entities"],
                "relations": path["relations"],
                "path_length": path["path_length"]
            }
            sample["reasoning_paths"].append(path_entry)
        
        training_data.append(sample)
    
    return training_data


def main():
    data_dir = "Data/webqsp_rog"
    output_dir = "Data/webqsp_labeled"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process train split
    print("="*60)
    print("Processing TRAIN split")
    print("="*60)
    train_results = process_parquet_files(data_dir, "train")
    
    # Create training dataset (only samples with valid paths)
    train_data = create_training_dataset(train_results)
    
    # Statistics
    total_samples = len(train_results)
    samples_with_paths = len(train_data)
    total_paths = sum(len(s["reasoning_paths"]) for s in train_data)
    
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples processed: {total_samples}")
    print(f"Samples with valid paths: {samples_with_paths} ({100*samples_with_paths/total_samples:.1f}%)")
    print(f"Total reasoning paths extracted: {total_paths}")
    print(f"Average paths per sample: {total_paths/samples_with_paths:.2f}" if samples_with_paths > 0 else "N/A")
    
    # Save full results
    output_file = os.path.join(output_dir, "train_with_paths.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved training data to: {output_file}")
    
    # Save a compact version for quick loading
    compact_data = []
    for sample in train_data:
        compact_sample = {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "relation_chains": [p["relation_chain"] for p in sample["reasoning_paths"]],
            "path_strings": [p["path_string"] for p in sample["reasoning_paths"]]
        }
        compact_data.append(compact_sample)
    
    compact_file = os.path.join(output_dir, "train_compact.json")
    with open(compact_file, "w", encoding="utf-8") as f:
        json.dump(compact_data, f, indent=2, ensure_ascii=False)
    print(f"Saved compact data to: {compact_file}")
    
    # Show some examples
    print(f"\n{'='*60}")
    print("EXAMPLE OUTPUTS")
    print(f"{'='*60}")
    for sample in train_data[:5]:
        print(f"\nID: {sample['id']}")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Number of paths: {len(sample['reasoning_paths'])}")
        if sample['reasoning_paths']:
            print("Sample path:")
            path = sample['reasoning_paths'][0]
            print(f"  Relation chain: {path['relation_chain']}")
            print(f"  Full path: {path['path_string']}")


if __name__ == "__main__":
    main()

