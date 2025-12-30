
import re
import os

results_json = "/data/Yanlai/KGLLM/Core/kg_retriever/outputs_eval_pretty/webqsp_test/results.json"
parse_limit = 10

try:
    read_size = 1000000 
    
    with open(results_json, 'r') as rj:
        chunk = rj.read(read_size) 
        
    print(f"Read {len(chunk)} bytes")
    
    examples_found = 0
    # Find all start indices of objects using the regex from main.py
    starts = [m.start() for m in re.finditer(r'^\s*\{', chunk, re.MULTILINE)]
    print(f"Found {len(starts)} potential starts")
    
    sample_examples = []
    
    for i, start in enumerate(starts):
        if examples_found >= parse_limit: break
        
        # Take a slice that should contain the object
        block = chunk[start:start+5000] 
        
        # Just check if this block looks like a valid result entry
        # in main.py: if '"question":' not in block or '"top_pred_path":' not in block:
        if '"question":' not in block or '"top_pred_path":' not in block:
            print(f"Block {i} missing keys")
            continue
            
        # Check if rank is 1
        if '"rank": 1' in block or '"rank": 1.0' in block:
            # Extract question
            q_match = re.search(r'"question":\s*"([^"]+)"', block)
            p_match = re.search(r'"top_pred_path":\s*\[(.*?)\]', block, re.DOTALL)
            
            if q_match:
                print(f"Found question: {q_match.group(1)}")
            else:
                print("Question regex failed")

            if p_match:
                 print("Found path")
            else:
                 print("Path regex failed")

            if q_match and p_match:
                question = q_match.group(1)
                
                # Clean up path string
                path_raw = p_match.group(1)
                path_list = [p.strip(' "\n') for p in path_raw.split(',')]
                path_list = [p for p in path_list if p] # filter empty
                
                sample_examples.append({
                    "question": question,
                    "top_pred_path": path_list,
                    "gt_path": [], 
                    "rank": 1
                })
                examples_found += 1
        else:
            print(f"Block {i} rank not 1")
            
    print(f"Total examples parsed: {len(sample_examples)}")

except Exception as e:
    print(f"Error: {e}")
