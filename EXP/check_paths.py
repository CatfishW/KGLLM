"""
Script to check and compare paths between original webqsp_rog and derived test.parquet
"""
import pandas as pd
import json
import numpy as np

# Load original ROG data
print("=" * 60)
print("Loading original webqsp_rog data...")
rog_df1 = pd.read_parquet('Data/webqsp_rog/test-00000-of-00002-9ee8d68f7d951e1f.parquet')
rog_df2 = pd.read_parquet('Data/webqsp_rog/test-00001-of-00002-773a7b8213e159f5.parquet')
rog_df = pd.concat([rog_df1, rog_df2], ignore_index=True)
print(f"Original ROG shape: {rog_df.shape}")
print(f"Original ROG columns: {rog_df.columns.tolist()}")

# Load derived test.parquet
print("\n" + "=" * 60)
print("Loading derived test.parquet...")
derived_df = pd.read_parquet('Data/webqsp_final/test.parquet')
print(f"Derived shape: {derived_df.shape}")
print(f"Derived columns: {derived_df.columns.tolist()}")

# Check a sample from original
print("\n" + "=" * 60)
print("Sample from ORIGINAL (webqsp_rog):")
sample_id = rog_df.iloc[0]['id']
print(f"ID: {sample_id}")
print(f"Question: {rog_df.iloc[0]['question']}")
print(f"Answer: {rog_df.iloc[0]['answer']}")

# Check graph structure
graph = rog_df.iloc[0]['graph']
if isinstance(graph, str):
    graph = json.loads(graph)
elif isinstance(graph, np.ndarray):
    graph = graph.tolist()
print(f"Graph type: {type(graph)}")
graph_len = len(graph) if graph is not None and not isinstance(graph, np.ndarray) else (graph.size if isinstance(graph, np.ndarray) else 0)
print(f"Graph length: {graph_len}")
if graph_len > 0:
    print(f"First triple: {graph[0]}")

# Check choices
choices = rog_df.iloc[0]['choices']
print(f"Choices type: {type(choices)}")
if isinstance(choices, np.ndarray):
    print(f"Choices shape: {choices.shape}")
    if choices.size > 0:
        print(f"First choice: {choices[0]}")
elif isinstance(choices, (list, str)):
    if isinstance(choices, str):
        choices = json.loads(choices)
    print(f"Choices count: {len(choices)}")
    if len(choices) > 0:
        print(f"First choice: {json.dumps(choices[0], indent=2)}")

# Check same sample in derived
print("\n" + "=" * 60)
print("Same sample from DERIVED (test.parquet):")
derived_sample = derived_df[derived_df['id'] == sample_id]
if len(derived_sample) > 0:
    row = derived_sample.iloc[0]
    print(f"ID: {row['id']}")
    print(f"Question: {row['question']}")
    
    paths = row['paths']
    if isinstance(paths, str):
        paths = json.loads(paths)
    print(f"Paths count: {len(paths) if paths else 0}")
    if paths and len(paths) > 0:
        for i, p in enumerate(paths[:3]):
            print(f"\nPath {i+1}:")
            print(json.dumps(p, indent=2))
else:
    print(f"Sample {sample_id} not found in derived data")

# Check how many samples have non-empty choices in original
print("\n" + "=" * 60)
print("Checking choices distribution in original data...")
non_empty_choices = 0
for i in range(len(rog_df)):
    c = rog_df.iloc[i]['choices']
    if isinstance(c, np.ndarray):
        if c.size > 0:
            non_empty_choices += 1
    elif c is not None:
        try:
            if len(c) > 0:
                non_empty_choices += 1
        except:
            pass
print(f"Samples with non-empty choices: {non_empty_choices} / {len(rog_df)}")
