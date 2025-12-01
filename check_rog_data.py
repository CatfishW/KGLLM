import pandas as pd
import json

# Check webqsp_rog data for WebQTrn-9
print("Checking webqsp_rog data for WebQTrn-9...")
df1 = pd.read_parquet('Data/webqsp_rog/train-00000-of-00002-d810a36ed97bc2cc.parquet')
df2 = pd.read_parquet('Data/webqsp_rog/train-00001-of-00002-e53244e71082a392.parquet')
df = pd.concat([df1, df2])

sample = df[df['id'] == 'WebQTrn-9']
if len(sample) > 0:
    s = sample.iloc[0]
    print(f"Found WebQTrn-9 in rog data")
    print(f"  Question: {s.get('question')}")
    print(f"  Answer: {s.get('answer')}")
    print(f"  Has choices: {'choices' in s}")
    choices = s.get('choices', [])
    print(f"  Choices type: {type(choices)}")
    print(f"  Choices length: {len(choices) if hasattr(choices, '__len__') else 'N/A'}")
    if hasattr(choices, '__len__') and len(choices) > 0:
        print(f"  First choice: {choices[0] if isinstance(choices, list) else 'Not a list'}")
else:
    print("WebQTrn-9 not found in rog data")

# Check validation data for all samples with empty paths
print("\nChecking validation data for samples with empty paths...")
df_val = pd.read_parquet('Data/webqsp_final/val.parquet')
empty_samples = []
for idx, row in df_val.iterrows():
    paths_str = row.get('paths', '[]')
    if isinstance(paths_str, str):
        paths = json.loads(paths_str)
    else:
        paths = paths_str
    if not isinstance(paths, list) or len(paths) == 0:
        empty_samples.append({
            'id': row.get('id'),
            'question': row.get('question')
        })

print(f"\nSamples with empty paths ({len(empty_samples)}):")
for sample in empty_samples[:10]:  # Show first 10
    print(f"  {sample['id']}: {sample['question']}")
if len(empty_samples) > 10:
    print(f"  ... and {len(empty_samples) - 10} more")

