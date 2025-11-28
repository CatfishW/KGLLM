from transformers import AutoTokenizer
import torch

def test_tokenizer_empty():
    tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    node_strings = []
    print(f"Testing tokenizer with input: {node_strings}")
    
    try:
        node_encoding = tokenizer(
            node_strings,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        print("Success!")
        print(node_encoding)
    except Exception as e:
        print(f"Caught expected exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tokenizer_empty()
