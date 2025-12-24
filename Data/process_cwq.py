import os
import json
import pandas as pd
import requests
import ast
import re

# Constants
DATA_DIR = "/data/Yanlai/KGLLM/Data/CWQ/shortest_paths"
CWQ_URLS = {
    "train": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AAAIHeWX0cPpbpwK6w06BCxva/ComplexWebQuestions_train.json?dl=1",
    "dev": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AADH8beLbOUWxwvY_K38E3ADa/ComplexWebQuestions_dev.json?dl=1",
    "test": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AABr4ysSy_Tg8Wfxww4i_UWda/ComplexWebQuestions_test.json?dl=1"
}

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")
        raise Exception("Download failed")

def extract_entities_from_sparql(sparql):
    # Basic extraction of MIDs (m.xxxx) from SPARQL
    # This is a heuristic.
    if not sparql:
        return []
    mids = re.findall(r'ns:(m\.[a-zA-Z0-9_]+)', sparql)
    return list(set(mids))

def process_cwq_split(split_name):
    os.makedirs(DATA_DIR, exist_ok=True)
    json_path = os.path.join(DATA_DIR, f"ComplexWebQuestions_{split_name}.json")
    
    if not os.path.exists(json_path):
        download_file(CWQ_URLS[split_name], json_path)
    
    print(f"Loading {split_name} data...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    processed_rows = []
    
    for item in data:
        # CWQ item structure keys: ID, question, answers, webqsp_question, webqsp_parsings, etc.
        
        # 1. ID
        q_id = item.get('ID', '')
        
        # 2. Question
        question = item.get('question', '')
        
        # 3. Answer
        # item['answers'] is usually a list of dicts like [{"answer": "...", "aliases": ...}]
        # We need to extract the answer strings.
        answers_raw = item.get('answers', [])
        answers_list = [ans['answer'] for ans in answers_raw]
        
        # 4. Q Entity
        # Try to use provided entities if available, else extract from SPARQL
        # CWQ doesn't always explicitly list 'q_entity' like WebQSP, 
        # but often has 'webqsp_question_entity' if it extends a specific question.
        # Let's rely on extracting from the SPARQL query as a fallback or primary method 
        # to ensure we capture relevant nodes.
        sparql = item.get('sparql', '')
        q_entities = extract_entities_from_sparql(sparql)
        
        # 5. Empty/Placeholder columns
        a_entity = [] # We don't have easy answer mapping to MIDs without KG lookup
        graph = []    # Subgraph not provided
        paths = []
        shortest_gt_paths = [] # To be filled by compensation
        
        row = {
            "id": q_id,
            "question": question,
            "answer": str(answers_list),      # Serialized list
            "q_entity": str(q_entities),      # Serialized list
            "a_entity": str(a_entity),        # Serialized list
            "graph": str(graph),              # Serialized list
            "paths": str(paths),              # Serialized list
            "shortest_gt_paths": str(shortest_gt_paths) # Serialized list
        }
        processed_rows.append(row)
        
    df = pd.DataFrame(processed_rows)
    output_path = os.path.join(DATA_DIR, f"{split_name}.parquet")
    # For compatibility, 'val' usually corresponds to 'dev'
    if split_name == 'dev':
        output_path = os.path.join(DATA_DIR, "val.parquet")
        
    print(f"Saving {len(df)} rows to {output_path}...")
    df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    process_cwq_split('train')
    process_cwq_split('dev')
    # process_cwq_split('test') # Optional
