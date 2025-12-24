
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
    
    # SPECIAL HANDLING FOR CWQ: Split into parts to avoid >2GB LFS limit
    is_cwq = (name == "cwq")
    MAX_ROWS_PER_PART = 15000
    current_part_idx = 1
    rows_in_current_part = 0
    
    schema = parquet_file.schema.to_arrow_schema()
    writer = None
    
    if is_cwq:
        current_output_path = os.path.join(OUTPUT_DIR, f"{name}_rephrased_train_part{current_part_idx}.parquet")
        writer = pq.ParquetWriter(current_output_path, schema)
        print(f"  Writing Part {current_part_idx} to {current_output_path}...")
    else:
        writer = pq.ParquetWriter(output_parquet_path, schema)
    
    current_idx = 0
    batch_size = 1000
    
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df_batch = batch.to_pandas()
        
        # Get slice of rephrased questions
        batch_questions = rephrased_questions[current_idx : current_idx + len(df_batch)]
        df_batch['question'] = batch_questions
        
        table = pa.Table.from_pandas(df_batch, schema=schema)
        
        # Check if we need to rotate file for CWQ
        if is_cwq and (rows_in_current_part + len(df_batch) > MAX_ROWS_PER_PART):
            # Write what fits? No, just rotate on batch boundary for simplicity
            # But wait, batch size is small (1000). 
            # If current rows > max, close and open new.
            if rows_in_current_part >= MAX_ROWS_PER_PART:
                 writer.close()
                 current_part_idx += 1
                 rows_in_current_part = 0
                 current_output_path = os.path.join(OUTPUT_DIR, f"{name}_rephrased_train_part{current_part_idx}.parquet")
                 writer = pq.ParquetWriter(current_output_path, schema)
                 print(f"  Writing Part {current_part_idx} to {current_output_path}...")

        writer.write_table(table)
        current_idx += len(df_batch)
        rows_in_current_part += len(df_batch)
        print(f"Processed {current_idx}/{len(rephrased_questions)}", end='\r')
        
    if writer:
        writer.close()
    print(f"\nDone saving {name}.")

if __name__ == "__main__":
    for name, path in INPUT_FILES.items():
        merge_dataset(name, path)
