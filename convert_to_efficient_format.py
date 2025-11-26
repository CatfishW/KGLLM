"""
Convert labeled data to efficient formats:
1. Parquet - best for storage, compression, and fast loading
2. JSONL - best for streaming during training (one sample per line)
3. Deduplicated paths - remove redundant path data

Also creates a simplified training format optimized for LLM training.
"""

import json
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict
from collections import defaultdict


def load_json_streaming(filepath: str):
    """Load large JSON file."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    return data


def create_training_format(data: List[Dict]) -> List[Dict]:
    """
    Create simplified format optimized for LLM training.
    
    Output format:
    {
        "id": str,
        "question": str,
        "answer": str,
        "q_entity": list[str],
        "a_entity": list[str],
        "paths": [  # List of full paths with entities
            {
                "full_path": "(e1) --[r1]--> (e2) --[r2]--> (e3)",
                "relation_chain": "r1 -> r2",
                "entities": ["e1", "e2", "e3"],
                "relations": ["r1", "r2"]
            },
            ...
        ]
    }
    """
    training_data = []
    
    for sample in data:
        # Deduplicate by full_path string (keeps full information)
        seen_paths = set()
        unique_paths = []
        
        for p in sample["reasoning_paths"]:
            path_key = p.get("path_string", "")
            if path_key and path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_paths.append({
                    "full_path": p.get("path_string", ""),
                    "relation_chain": p.get("relation_chain", ""),
                    "entities": p.get("entities", []),
                    "relations": p.get("relations", [])
                })
        
        # Sort by path length (shorter paths first)
        unique_paths.sort(key=lambda x: (len(x.get("relations", [])), x.get("full_path", "")))
        
        training_sample = {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "q_entity": sample.get("q_entity", []),
            "a_entity": sample.get("a_entity", []),
            "paths": unique_paths
        }
        training_data.append(training_sample)
    
    return training_data


def create_llm_instruction_format(data: List[Dict]) -> List[Dict]:
    """
    Create instruction-tuning format for LLM training.
    
    Format suitable for supervised fine-tuning:
    {
        "instruction": "Given the question, generate reasoning paths...",
        "input": question,
        "output": formatted full paths with entities
    }
    """
    instruction_data = []
    
    instruction_template = (
        "Given a question about a knowledge graph, generate the full reasoning paths "
        "(including entities and relations) that connect the question entity to the answer entity."
    )
    
    for sample in data:
        # Deduplicate by full_path string
        seen_paths = set()
        unique_paths = []
        
        for p in sample["reasoning_paths"]:
            path_key = p.get("path_string", "")
            if path_key and path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_paths.append(path_key)
        
        # Sort by length
        unique_paths.sort(key=lambda x: (x.count("-->"), x))
        
        # Format output as newline-separated full paths
        output = "\n".join(unique_paths[:10])  # Limit to top 10 paths
        
        instruction_sample = {
            "id": sample["id"],
            "instruction": instruction_template,
            "input": f"Question: {sample['question']}\nAnswer: {sample['answer']}",
            "output": output
        }
        instruction_data.append(instruction_sample)
    
    return instruction_data


def save_jsonl(data: List[Dict], filepath: str):
    """Save data as JSONL (one JSON object per line)."""
    print(f"Saving JSONL to {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} samples")


def save_parquet(data: List[Dict], filepath: str):
    """Save data as Parquet with compression."""
    print(f"Saving Parquet to {filepath}...")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Convert list columns to string representation for Parquet compatibility
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)
    
    # Save with snappy compression (good balance of speed and size)
    df.to_parquet(filepath, compression='snappy', index=False)
    print(f"Saved {len(data)} samples")


def create_path_vocabulary(data: List[Dict]) -> Dict:
    """
    Extract unique relations and paths for vocabulary building.
    """
    relations = set()
    relation_chains = defaultdict(int)
    
    for sample in data:
        for path in sample["reasoning_paths"]:
            chain = path.get("relation_chain", "")
            if chain:
                relation_chains[chain] += 1
                for rel in path.get("relations", []):
                    relations.add(rel)
    
    return {
        "unique_relations": sorted(list(relations)),
        "num_unique_relations": len(relations),
        "unique_chains": len(relation_chains),
        "top_chains": sorted(relation_chains.items(), key=lambda x: -x[1])[:100]
    }


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)


def main():
    input_file = "Data/webqsp_labeled/train_with_paths.json"
    output_dir = "Data/webqsp_labeled"
    
    # Load original data
    data = load_json_streaming(input_file)
    
    print("\n" + "="*60)
    print("CREATING EFFICIENT FORMATS")
    print("="*60)
    
    # 1. Create simplified training format (deduplicated paths)
    print("\n[1/4] Creating simplified training format...")
    training_data = create_training_format(data)
    
    # Save as JSONL
    training_jsonl_path = os.path.join(output_dir, "train_paths.jsonl")
    save_jsonl(training_data, training_jsonl_path)
    
    # Save as Parquet
    training_parquet_path = os.path.join(output_dir, "train_paths.parquet")
    save_parquet(training_data, training_parquet_path)
    
    # 2. Create LLM instruction format
    print("\n[2/4] Creating LLM instruction-tuning format...")
    instruction_data = create_llm_instruction_format(data)
    
    instruction_jsonl_path = os.path.join(output_dir, "train_instruction.jsonl")
    save_jsonl(instruction_data, instruction_jsonl_path)
    
    instruction_parquet_path = os.path.join(output_dir, "train_instruction.parquet")
    save_parquet(instruction_data, instruction_parquet_path)
    
    # 3. Extract vocabulary
    print("\n[3/4] Extracting path vocabulary...")
    vocab = create_path_vocabulary(data)
    
    vocab_path = os.path.join(output_dir, "path_vocabulary.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"Saved vocabulary to {vocab_path}")
    
    # Save relations list
    relations_path = os.path.join(output_dir, "relations_list.txt")
    with open(relations_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab["unique_relations"]))
    print(f"Saved {vocab['num_unique_relations']} unique relations to {relations_path}")
    
    # 4. Print statistics and file sizes
    print("\n[4/4] Computing statistics...")
    print("\n" + "="*60)
    print("FILE SIZE COMPARISON")
    print("="*60)
    
    files_to_check = [
        ("Original JSON", input_file),
        ("Training JSONL", training_jsonl_path),
        ("Training Parquet", training_parquet_path),
        ("Instruction JSONL", instruction_jsonl_path),
        ("Instruction Parquet", instruction_parquet_path),
    ]
    
    for name, path in files_to_check:
        if os.path.exists(path):
            size = get_file_size_mb(path)
            print(f"{name:25s}: {size:8.2f} MB")
    
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    print(f"Total samples: {len(data)}")
    print(f"Unique relations: {vocab['num_unique_relations']}")
    print(f"Unique relation chains: {vocab['unique_chains']}")
    
    # Sample output
    print("\n" + "="*60)
    print("SAMPLE OUTPUT (Training Format)")
    print("="*60)
    for sample in training_data[:3]:
        print(f"\nID: {sample['id']}")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Q_Entity: {sample['q_entity']}")
        print(f"A_Entity: {sample['a_entity']}")
        print(f"Paths ({len(sample['paths'])} unique):")
        for p in sample['paths'][:3]:
            print(f"  Full: {p['full_path']}")
            print(f"  Relations: {p['relation_chain']}")
            print(f"  Entities: {p['entities']}")
            print()
        if len(sample['paths']) > 3:
            print(f"  ... and {len(sample['paths'])-3} more paths")
    
    print("\n" + "="*60)
    print("SAMPLE OUTPUT (Instruction Format)")
    print("="*60)
    sample = instruction_data[0]
    print(f"ID: {sample['id']}")
    print(f"Instruction: {sample['instruction']}")
    print(f"Input: {sample['input']}")
    print(f"Output:\n{sample['output']}")


if __name__ == "__main__":
    main()

