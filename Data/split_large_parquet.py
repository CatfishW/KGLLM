
import pandas as pd
import os
import math

INPUT_FILE = "/data/Yanlai/KGLLM/Data/LLM_Rephrased_CWQ_WebQSP/cwq_rephrased_train.parquet"
OUTPUT_DIR = "/data/Yanlai/KGLLM/Data/LLM_Rephrased_CWQ_WebQSP"
MAX_ROWS_PER_FILE = 15000 # Adjust based on size. 27k rows -> 2.3GB. So ~14k rows should be < 2GB.

def split_parquet():
    print(f"Reading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    
    num_parts = math.ceil(total_rows / MAX_ROWS_PER_FILE)
    print(f"Splitting into {num_parts} parts...")
    
    for i in range(num_parts):
        start_idx = i * MAX_ROWS_PER_FILE
        end_idx = min((i + 1) * MAX_ROWS_PER_FILE, total_rows)
        
        df_part = df.iloc[start_idx:end_idx]
        output_path = os.path.join(OUTPUT_DIR, f"cwq_rephrased_train_part{i+1}.parquet")
        print(f"Saving part {i+1} to {output_path} ({len(df_part)} rows)...")
        df_part.to_parquet(output_path)
        
    print("Done.")

if __name__ == "__main__":
    split_parquet()
