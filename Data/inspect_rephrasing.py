
import pandas as pd
import random
import os

# Paths
PATHS = {
    "WebQSP": {
        "orig": "/data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/train.parquet",
        "new": "/data/Yanlai/KGLLM/Data/LLM_Rephrased_CWQ_WebQSP/webqsp_rephrased_train.parquet"
    },
    "CWQ": {
        "orig": "/data/Yanlai/KGLLM/Data/CWQ/shortest_paths/train.parquet",
        "new": "/data/Yanlai/KGLLM/Data/LLM_Rephrased_CWQ_WebQSP/cwq_rephrased_train.parquet"
    }
}

def inspect_dataset(name):
    print(f"\n{'='*20} {name} Examples {'='*20}")
    orig_path = PATHS[name]["orig"]
    new_path = PATHS[name]["new"]
    
    if not os.path.exists(new_path):
        print(f"Rephrased file for {name} not found.")
        return

    # Load only question column to be fast
    try:
        df_orig = pd.read_parquet(orig_path, columns=['question'])
        df_new = pd.read_parquet(new_path, columns=['question'])
    except Exception as e:
        print(f"Error reading {name}: {e}")
        return
        
    if len(df_orig) != len(df_new):
        print(f"Warning: Length mismatch! Orig: {len(df_orig)}, New: {len(df_new)}")
        # We can still truncate to min length for inspection
    
    limit = min(len(df_orig), len(df_new))
    indices = random.sample(range(limit), 10)
    
    for i, idx in enumerate(indices):
        q_orig = df_orig.iloc[idx]['question']
        q_new = df_new.iloc[idx]['question']
        print(f"\nExample {i+1} (Index {idx}):")
        print(f"  Original:  {q_orig}")
        print(f"  Rephrased: {q_new}")

if __name__ == "__main__":
    for dataset in ["WebQSP", "CWQ"]:
        inspect_dataset(dataset)
