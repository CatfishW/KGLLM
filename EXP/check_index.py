"""Check index metadata format."""
import pickle

data = pickle.load(open('EXP/index/metadata.pkl', 'rb'))
meta = data['metadata']

print('Total indexed paths:', len(meta))
print('\nSample indexed relation_chains:')
for m in meta[:10]:
    print(f"  '{m['relation_chain']}'")

# Check format - are they space-separated or dot-separated?
print('\n\nAnalyzing format:')
sample = meta[0]['relation_chain']
print(f"Sample: '{sample}'")
print(f"Contains ' -> ': {' -> ' in sample}")
print(f"Contains '.': {'.' in sample}")
