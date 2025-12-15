
import os
import networkx as nx
import pandas as pd
from datasets import load_dataset
import multiprocessing
from tqdm import tqdm
import ast
import json

# Configuration
OUTPUT_DIR = "/data/Yanlai/KGLLM/Data/CWQ/shortest_paths"
NUM_WORKERS = 2  # Reduced further to avoid OOM and hangs


def process_single_item(item):
    """
    Process a single item from the dataset.
    """
    try:
        q_id = item.get('id', '')
        question = item.get('question', '')
        
        def safe_eval(val):
            if isinstance(val, str):
                try:
                    return ast.literal_eval(val)
                except:
                    return [val]
            return val

        a_entities = safe_eval(item.get('a_entity', []))
        q_entities = safe_eval(item.get('q_entity', []))
        answers = safe_eval(item.get('answer', []))
        graph_triples = safe_eval(item.get('graph', []))
        
        # Build Graph
        G = nx.MultiDiGraph()
        for triple in graph_triples:
            if len(triple) == 3:
                h, r, t = triple
                G.add_edge(h, t, relation=r)
        
        final_paths = []
        
        valid_sources = [n for n in q_entities if n in G]
        valid_targets = [n for n in a_entities if n in G]
        
        # We find meaningful shortest paths. 
        # Strategy: Iterate all source-target pairs, find all shortest paths.
        
        for source in valid_sources:
            for target in valid_targets:
                try:
                     if nx.has_path(G, source, target):
                        # nx.all_shortest_paths returns list of nodes: [n1, n2, n3]
                        for path_nodes in nx.all_shortest_paths(G, source, target):
                             # Reconstruct path details (relations)
                             # Since it's a MultiGraph, there might be multiple edges between nodes.
                             # We pick the first one for simplicity, or we could explode all combinations.
                             # Expanding all combinations can lead to explosion, so we pick first edge.
                             
                             entities = path_nodes
                             relations = []
                             
                             valid_path = True
                             for i in range(len(path_nodes) - 1):
                                 u = path_nodes[i]
                                 v = path_nodes[i+1]
                                 edge_data = G.get_edge_data(u, v)
                                 if not edge_data:
                                     valid_path = False
                                     break
                                 # Pick first relation
                                 relations.append(edge_data[0]['relation'])
                             
                             if valid_path:
                                 final_paths.append({
                                     "entities": entities,
                                     "relations": relations,
                                     "relation_chain": " -> ".join(relations)
                                 })
                except nx.NetworkXNoPath:
                    continue
        
        # Deduplicate paths based on relation chain or exact content?
        # Dataset logic deduplicates by relation_chain, so we can leave duplicates or remove them here.
        # Let's simple-dedupe here to save space
        unique_paths = []
        seen = set()
        for p in final_paths:
            sig = tuple(p['entities'] + p['relations'])
            if sig not in seen:
                seen.add(sig)
                unique_paths.append(p)

        return {
            "id": q_id,
            "question": question,
            "answer": str(answers),
            "q_entity": str(q_entities),
            "a_entity": str(a_entities),
            "graph": str(graph_triples),
            "paths": "[]",
            "shortest_gt_paths": str(unique_paths).replace("'", '"') # JSON-like string, though dataset uses ast.literal_eval/json.loads check
        }
    except Exception as e:
         print(f"Error processing {item.get('id', '?')}: {e}")
         return {
            "id": item.get('id', ''),
            "question": item.get('question', ''),
            "answer": str(item.get('answer', [])),
            "q_entity": str(item.get('q_entity', [])),
            "a_entity": str(item.get('a_entity', [])),
            "graph": str(item.get('graph', [])),
            "paths": "[]",
            "shortest_gt_paths": "[]"
         }

def process_dataset(split_name, dataset_split):
    print(f"Processing {split_name} split with {len(dataset_split)} samples...")
    
    # Convert dataset to list of dicts for multiprocessing
    data_list = list(dataset_split)
    
    results = []
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(process_single_item, data_list), total=len(data_list)))
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, f"{split_name}.parquet")
    # Remap 'validation' to 'val' to match user's custom naming convention if needed
    if split_name == 'validation':
        output_path = os.path.join(OUTPUT_DIR, "val.parquet")
    elif split_name == 'test':
        output_path = os.path.join(OUTPUT_DIR, "test.parquet")
    else:
        output_path = os.path.join(OUTPUT_DIR, "train.parquet")
        
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading rmanluo/RoG-cwq...")
    dataset = load_dataset("rmanluo/RoG-cwq")
    
    # Process train
    if 'train' in dataset:
        process_dataset('train', dataset['train'])
        
    # Process validation
    if 'validation' in dataset:
        process_dataset('validation', dataset['validation'])
    elif 'dev' in dataset:
        process_dataset('validation', dataset['dev'])
        
    # Process test
    if 'test' in dataset:
        process_dataset('test', dataset['test'])
