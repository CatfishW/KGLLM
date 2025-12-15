
import asyncio
import aiohttp
import pandas as pd
import json
import os
from tqdm.asyncio import tqdm

# Configuration
API_BASE_URL = "https://game.agaii.org/llm/v1"
API_KEY = "EMPTY" # Usually local/internal APIs don't check key, but standard is often required
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct-FP8"
CONCURRENCY_LIMIT = 50 # Adjust based on server capacity

INPUT_FILES = {
    "cwq": "/data/Yanlai/KGLLM/Data/CWQ/shortest_paths/train.parquet",
    "webqsp": "/data/Yanlai/KGLLM/Data/webqsp_final/shortest_paths/train.parquet"
}
OUTPUT_DIR = "/data/Yanlai/KGLLM/Data/LLM_Rephrased_CWQ_WebQSP"

async def get_model_name(session):
    try:
        async with session.get(f"{API_BASE_URL}/models") as response:
            if response.status == 200:
                data = await response.json()
                return data['data'][0]['id']
    except Exception as e:
        print(f"Error fetching model name: {e}")
        return MODEL_NAME

async def rephrase_question(session, question, model, sem):
    async with sem:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Rephrase the following question naturally while preserving its exact meaning. Do not output anything else, just the rephrased question."},
                {"role": "user", "content": f"Rephrase this question: {question}"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        try:
            async with session.post(f"{API_BASE_URL}/chat/completions", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content'].strip()
                else:
                    return None # Return None on failure to retry or skip
        except Exception as e:
            return None

async def process_dataset(name, path, model):
    print(f"Processing {name} from {path}...", flush=True)
    if not os.path.exists(path):
        print(f"ERROR: Path does not exist: {path}", flush=True)
        return

    try:
        print("Reading parquet questions only...", flush=True)
        # Only read question column to save memory during processing
        df_questions = pd.read_parquet(path, engine='pyarrow', columns=['question'])
        print(f"Loaded {len(df_questions)} questions", flush=True)
    except Exception as e:
        print(f"ERROR loading parquet: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    # For testing/demo purposes, we can uncomment this to run on a small subset
    # df = df_questions.head(10)
    
    questions = df_questions['question'].tolist()
    print(f"First 3 questions: {questions[:3]}")

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    print("Starting rephrasing...")
    async with aiohttp.ClientSession() as session:
        tasks = [rephrase_question(session, q, model, sem) for q in questions]
        # Use gather to ensure order matches original list
        results = await tqdm.gather(*tasks, desc=f"Rephrasing {name}")
    
    print("Rephrasing complete.")
    
    # Check for failures
    success_count = sum(1 for r in results if r is not None)
    print(f"Successfully rephrased {success_count}/{len(questions)} questions.")

    # Fill invalid results with original questions
    final_questions = [r if r else q for r, q in zip(results, questions)]
    
    # Save as JSON first to avoid OOM and allow checkpointing
    json_output_path = os.path.join(OUTPUT_DIR, f"{name}_rephrased.json")
    print(f"Saving rephrased questions to {json_output_path}...")
    with open(json_output_path, 'w') as f:
        json.dump(final_questions, f, indent=2)
    print(f"Saved {len(final_questions)} questions to JSON.")
    
    # We will merge later manually or with a lightweight script
    return

async def main():
    print("Initializing...")
    async with aiohttp.ClientSession() as session:
        model = await get_model_name(session)
        print(f"Using model: {model}")
    
    for name, path in INPUT_FILES.items():
        print(f"--- Starting {name} ---")
        await process_dataset(name, path, model)
        print(f"--- Finished {name} ---")

if __name__ == "__main__":
    asyncio.run(main())
