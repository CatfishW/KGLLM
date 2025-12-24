"""
Explore webqsp_rog parquet data structure
"""
import pandas as pd
import os

# Load one parquet file to explore structure
data_path = "Data/webqsp_combined"
train_file = "train_combined.parquet"

df = pd.read_parquet(os.path.join(data_path, train_file))

print("="*60)
print("DataFrame Shape:", df.shape)
print("="*60)
print("\nColumn Names:")
print(df.columns.tolist())
print("="*60)
print("\nColumn Types:")
print(df.dtypes)
print("="*60)
print("\nFirst 3 rows (showing all columns):")
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_columns', None);
pd.set_option('display.width', None)

for idx, row in df.head(1).iterrows():
    print(f"\n--- Row {idx} ---")
    for col in df.columns:
        val = row[col]
        # If it's a list/sequence, show first few items
        if isinstance(val, list):
            print(f"{col}: (len={len(val)})")
            if len(val) > 0:
                print(f"  First 2 items: {val[:2]}")
        else:
            print(f"{col}: {val}")

