
import pandas as pd
import json
import ast

def inspect_file(path):
    print(f"--- Inspecting {path} ---")
    df = pd.read_parquet(path)
    print(f"Columns: {df.columns.tolist()}")
    if len(df) > 0:
        row = df.iloc[0]
        print("\nFirst row sample:")
        for col in ['graph', 'shortest_gt_paths', 'paths']:
            if col in row:
                val = row[col]
                print(f"\n{col} type: {type(val)}")
                print(f"{col} snippet: {str(val)[:200]}")
                
                # Try parsing
                if isinstance(val, str):
                    try:
                        json.loads(val)
                        print(f"  -> Valid ISO JSON")
                    except Exception as e:
                        print(f"  -> Not JSON: {e}")
                        try:
                            ast.literal_eval(val)
                            print(f"  -> Valid Python literal")
                        except Exception as e2:
                            print(f"  -> Not Python literal: {e2}")

inspect_file('/data/Yanlai/KGLLM/Data/CWQ/shortest_paths/train.parquet')
