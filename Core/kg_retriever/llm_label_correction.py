
import os
import asyncio
import pandas as pd
import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import json
import sys

# Add path to import local modules
sys.path.insert(0, os.getcwd())
from data.question_classifier_dataset import load_all_datasets, extract_label, LABEL_NAMES

API_BASE = 'https://game.agaii.org/llm/v1'
API_KEY = 'EMPTY' 
MODEL = 'gpt-4o'
CONCURRENCY = 50  # Number of concurrent requests

async def classify_sample(client, semaphore, row, heuristic_label_name):
    question = row['question']
    
    # Get answer for context
    answer = row.get('answer', '')
    a_entity = row.get('a_entity', '')
    
    # Simplify answer/entity for prompt
    if isinstance(answer, str):
        answer = answer[:200]
    elif isinstance(answer, (list, np.ndarray)):
        answer = str(list(answer)[:5])
        
    if isinstance(a_entity, str):
        a_entity = a_entity[:200]
    elif isinstance(a_entity, (list, np.ndarray)):
        a_entity = str(list(a_entity)[:5])

    prompt = f"""You are a data labeler for a KGQA Question Classifier.
Classify the following question into one of 3 classes:
1. 'one_hop': Answerable with a single relation path.
2. 'multi_hop': Requires multiple relations/hops.
3. 'numeric': Asks for a count, quantity, year, or number.

Question: {question}
Answer Entities: {a_entity}
Answer Values: {answer}
Heuristic Label: {heuristic_label_name}

Reason step-by-step then provide the final label in JSON format: {{"reasoning": "...", "label": "..."}}
"""

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=100,
                temperature=0.1
            )
            content = response.choices[0].message.content
            # extracting json from content
            try:
                # Find JSON part
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    data = json.loads(json_str)
                    return data.get('label', heuristic_label_name), data.get('reasoning', '')
            except:
                pass
            return heuristic_label_name, "Parse Error"
        except Exception as e:
            return heuristic_label_name, f"API Error: {str(e)}"

async def main():
    print("Loading datasets...")
    train_df, val_df, test_df = load_all_datasets()
    
    # Combine for processing (we can split later or just process train)
    # User said "use it to train", so mainly train.
    # But for validation accuracy we should probably process val too.
    
    # Let's process train_df first.
    print(f"Processing training data: {len(train_df)} samples")
    
    # Apply heuristic first
    print("Applying heuristics...")
    records = []
    for idx, row in train_df.iterrows():
        lbl_idx = extract_label(row)
        if lbl_idx is not None:
            records.append({
                'idx': idx,
                'row': row,
                'heuristic_label': LABEL_NAMES[lbl_idx]
            })
    
    print(f"Valid samples after heuristic: {len(records)}")
    
    # Limit for testing?
    limit = 10  # TEST MODE - Remove for full run
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        limit = len(records)
        print("FULL RUN ENABLED")
    else:
        print(f"TEST RUN: Processing first {limit} samples only. Use --full to run all.")
        records = records[:limit]

    client = AsyncOpenAI(base_url=API_BASE, api_key=API_KEY)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    tasks = []
    for rec in records:
        tasks.append(classify_sample(client, semaphore, rec['row'], rec['heuristic_label']))
    
    results = []
    # Use tqdm
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        res = await f
        results.append(res)
    
    # Update DataFrame
    # Note: results are in specific order? No, as_completed is mixed.
    # Wait, I need to match results to records. 
    # Let's rewrite loop to keep index.
    
    # Better approach:
    # We'll just zip inputs and outputs if we use gather.
    # But we want progress bar.
    
    print("Restarting async with gather for order preservation...")
    tasks = [classify_sample(client, semaphore, rec['row'], rec['heuristic_label']) for rec in records]
    
    # We can stick to gather and wrap in tqdm
    results = await tqdm.gather(*tasks)
    
    # Store results
    corrections = 0
    new_data = []
    for i, (final_label, reasoning) in enumerate(results):
        rec = records[i]
        orig_label = rec['heuristic_label']
        
        # Normalize label
        final_label = final_label.lower()
        if 'one' in final_label and 'hop' in final_label: final_label = 'one_hop'
        if 'multi' in final_label and 'hop' in final_label: final_label = 'multi_hop'
        if 'numeric' in final_label: final_label = 'numeric'
        
        if final_label not in LABEL_NAMES:
            final_label = orig_label  # Fallback
            
        if final_label != orig_label:
            corrections += 1
            # print(f"Change: {orig_label} -> {final_label} | Q: {rec['row']['question'][:50]}")
            
        # Create row for new dataframe
        new_row = rec['row'].to_dict()
        new_row['llm_label'] = final_label
        new_row['llm_reasoning'] = reasoning
        new_data.append(new_row)
        
    print(f"Processing complete. Corrections: {corrections}/{len(records)} ({100*corrections/len(records):.1f}%)")
    
    # Save
    out_df = pd.DataFrame(new_data)
    out_path = 'train_llm_corrected.parquet'
    out_df.to_parquet(out_path)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    asyncio.run(main())
