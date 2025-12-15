
import networkx as nx
from datasets import load_dataset
import ast

def debug_rog_sample():
    print("Loading one sample from rmanluo/RoG-cwq...")
    dataset = load_dataset("rmanluo/RoG-cwq", split="train", streaming=True)
    item = next(iter(dataset))
    
    print(f"ID: {item.get('id')}")
    print(f"Question: {item.get('question')}")
    
    # Parse
    def safe_eval(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return [val]
        return val

    a_entities = safe_eval(item.get('a_entity', []))
    q_entities = safe_eval(item.get('q_entity', []))
    graph_triples = safe_eval(item.get('graph', []))
    
    print(f"Q Entities ({len(q_entities)}): {q_entities}")
    print(f"A Entities ({len(a_entities)}): {a_entities}")
    print(f"Graph Triples ({len(graph_triples)}): {graph_triples[:3]} ...")
    
    # Build Graph
    G = nx.MultiDiGraph()
    for triple in graph_triples:
        if len(triple) == 3:
            h, r, t = triple
            G.add_edge(h, t, relation=r)
            
    print(f"Graph Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Check connectivity
    valid_sources = [n for n in q_entities if n in G]
    valid_targets = [n for n in a_entities if n in G]
    
    print(f"Valid Sources in Graph: {valid_sources}")
    print(f"Valid Targets in Graph: {valid_targets}")
    
    if valid_sources and valid_targets:
        for s in valid_sources:
            for t in valid_targets:
                if nx.has_path(G, s, t):
                    print(f"Path exists between {s} and {t}")
                    paths = list(nx.all_shortest_paths(G, s, t))
                    print(f"  Found {len(paths)} shortest paths")
                    print(f"  Example path: {paths[0]}")
                else:
                    print(f"No path between {s} and {t}")
    else:
        print("No valid source-target pairs found in graph!")

if __name__ == "__main__":
    debug_rog_sample()
