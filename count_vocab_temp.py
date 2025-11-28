import json
import os

file_path = "Data/webqsp_final/vocab.json"
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Vocab size: {len(data)}")
except Exception as e:
    print(f"Error: {e}")
