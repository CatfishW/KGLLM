
import pandas as pd
import json
import os
import pyarrow.parquet as pq
import pyarrow as pa

# Config matching the augment script
OUTPUT_DIR = "/data/Yanlai/KGLLM/Data/LLM_Rephrased_CWQ_WebQSP"
INPUT_FILES = {
    "cwq": "/data/Yanlai/KGLLM/Data/CWQ/shortest_paths/train.parquet",
    "webqsp": "/data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/train.parquet"
}

def merge_dataset(name, path):
    json_path = os.path.join(OUTPUT_DIR, f"{name}_rephrased.json")
    if not os.path.exists(json_path):
        print(f"Skipping {name}: JSON not found at {json_path}")
        return

    print(f"Loading rephrased questions for {name}...")
    with open(json_path, 'r') as f:
        rephrased_questions = json.load(f)
    
    print(f"Reading original parquet schema from {path}...")
    parquet_file = pq.ParquetFile(path)
    
    # Check length
    if parquet_file.metadata.num_rows != len(rephrased_questions):
        print(f"ERROR: Length mismatch! Parquet: {parquet_file.metadata.num_rows}, JSON: {len(rephrased_questions)}")
        return

    output_parquet_path = os.path.join(OUTPUT_DIR, f"{name}_rephrased_train.parquet")
    print(f"Merging and saving to {output_parquet_path}...")
    
    # We will write a new parquet file, iterating over batches of the original
    # and replacing the question column.
    
    schema = parquet_file.schema.to_arrow_schema()
    writer = None
    
    current_idx = 0
    batch_size = 1000
    
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df_batch = batch.to_pandas()
        
        # Get slice of rephrased questions
        batch_questions = rephrased_questions[current_idx : current_idx + len(df_batch)]
        df_batch['question'] = batch_questions
        
        table = pa.Table.from_pandas(df_batch, schema=schema)
        
        if writer is None:
            writer = pq.ParquetWriter(output_parquet_path, schema)
            
        writer.write_table(table)
        current_idx += len(df_batch)
        print(f"Processed {current_idx}/{len(rephrased_questions)}", end='\r')
        
    if writer:
        writer.close()
    print(f"\nDone saving {name}.")

if __name__ == "__main__":
    for name, path in INPUT_FILES.items():
        merge_dataset(name, path)
