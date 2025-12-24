
import pandas as pd
import json
import os
import pyarrow.parquet as pq
import pyarrow as pa
import time

INPUT_PARQUET = "/data/Yanlai/KGLLM/Data/CWQ/shortest_paths/train.parquet"
INPUT_JSON = "/data/Yanlai/KGLLM/Data/LLM_Rephrased_CWQ_WebQSP/cwq_rephrased.json"
OUTPUT_DIR = "/data/Yanlai/KGLLM/Data/LLM_Rephrased_CWQ_WebQSP"

def debug_merge():
    print("Start debug merge...", flush=True)
    
    if not os.path.exists(INPUT_PARQUET):
        print(f"Missing parquet: {INPUT_PARQUET}")
        return
    if not os.path.exists(INPUT_JSON):
        print(f"Missing json: {INPUT_JSON}")
        return
        
    print("Loading JSON...", flush=True)
    with open(INPUT_JSON, 'r') as f:
        qs = json.load(f)
    print(f"Loaded {len(qs)} questions.", flush=True)
    
    print("Opening Parquet...", flush=True)
    pf = pq.ParquetFile(INPUT_PARQUET)
    print(f"Parquet rows: {pf.metadata.num_rows}", flush=True)
    
    if len(qs) != pf.metadata.num_rows:
        print("MISMATCH!", flush=True)
        # But we proceed for debug
        
    schema = pf.schema.to_arrow_schema()
    
    out_path = os.path.join(OUTPUT_DIR, "debug_cwq_part1.parquet")
    print(f"Writing to {out_path}", flush=True)
    
    writer = pq.ParquetWriter(out_path, schema)
    
    count = 0
    for i, batch in enumerate(pf.iter_batches(batch_size=100)):
        if i % 10 == 0:
            print(f"Batch {i}, size {count}", flush=True)
        
        df = batch.to_pandas()
        
        # Dummy replacement
        df['question'] = ["REPHRASED"] * len(df)
        
        table = pa.Table.from_pandas(df, schema=schema)
        writer.write_table(table)
        count += len(df)
        
        if count > 500:
            print("Stopping early for debug.", flush=True)
            break
            
    writer.close()
    print(f"Done. Wrote {count} rows.", flush=True)

if __name__ == "__main__":
    debug_merge()
