#!/usr/bin/env python3
"""
Augment CWQ and WebQSP datasets by rephrasing questions using LLM.
Uses batched async inference for efficiency with checkpointing support.
"""

import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import time
import signal
import sys

LLM_API_URL = "https://game.agaii.org/llm/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct-FP8"

REPHRASE_PROMPT = """Rephrase the following question in a different way while preserving its exact meaning. Do not add or remove any information. Output ONLY the rephrased question, nothing else.

Question: {question}"""

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    global shutdown_flag
    print("\n\nReceived interrupt signal. Saving checkpoint and exiting gracefully...")
    shutdown_flag = True


async def rephrase_single(session: aiohttp.ClientSession, question: str, semaphore: asyncio.Semaphore, retry_count: int = 3) -> str:
    """Rephrase a single question using the LLM API."""
    if shutdown_flag:
        return question
        
    async with semaphore:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": REPHRASE_PROMPT.format(question=question)}
            ],
            "max_tokens": 200,
            "temperature": 0.7,
        }
        
        for attempt in range(retry_count):
            if shutdown_flag:
                return question
            try:
                async with session.post(LLM_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        result = await response.json()
                        rephrased = result["choices"][0]["message"]["content"].strip()
                        # Remove thinking tags if present (Qwen3 sometimes includes them)
                        if "<think>" in rephrased:
                            rephrased = rephrased.split("</think>")[-1].strip()
                        return rephrased
                    else:
                        error_text = await response.text()
                        print(f"API error (attempt {attempt+1}): {response.status} - {error_text[:100]}")
                        await asyncio.sleep(1)
            except asyncio.TimeoutError:
                print(f"Timeout for question (attempt {attempt+1}): {question[:50]}...")
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                return question
            except Exception as e:
                print(f"Error (attempt {attempt+1}): {e}")
                await asyncio.sleep(1)
        
        # Return original if all retries failed
        return question


async def process_batch(questions: list[str], start_idx: int = 0, existing_rephrased: list[str] = None, 
                        batch_size: int = 50, max_concurrent: int = 20) -> tuple[list[str], int]:
    """Process a batch of questions concurrently with checkpoint support."""
    global shutdown_flag
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Initialize with existing results
    if existing_rephrased:
        rephrased = existing_rephrased.copy()
    else:
        rephrased = []
    
    last_completed_idx = start_idx
    
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(start_idx, len(questions), batch_size), 
                      desc="Processing batches", 
                      initial=start_idx // batch_size,
                      total=len(questions) // batch_size + 1):
            if shutdown_flag:
                break
                
            batch = questions[i:i + batch_size]
            tasks = [rephrase_single(session, q, semaphore) for q in batch]
            batch_results = await asyncio.gather(*tasks)
            rephrased.extend(batch_results)
            last_completed_idx = i + len(batch)
            
            # Small delay between batches to avoid overwhelming the API
            await asyncio.sleep(0.5)
        
        return rephrased, last_completed_idx


def save_checkpoint(df: pd.DataFrame, rephrased: list[str], output_path: Path, last_idx: int):
    """Save checkpoint with partial results."""
    checkpoint_path = output_path.with_suffix('.checkpoint.parquet')
    checkpoint_df = df.iloc[:len(rephrased)].copy()
    checkpoint_df['question_rephrased'] = rephrased
    checkpoint_df.to_parquet(checkpoint_path, index=False)
    
    # Save progress info
    info_path = output_path.with_suffix('.checkpoint.json')
    with open(info_path, 'w') as f:
        json.dump({'last_idx': last_idx, 'total': len(df)}, f)
    
    print(f"Checkpoint saved at index {last_idx}/{len(df)}")


def load_checkpoint(output_path: Path) -> tuple[int, list[str]]:
    """Load checkpoint if exists."""
    checkpoint_path = output_path.with_suffix('.checkpoint.parquet')
    info_path = output_path.with_suffix('.checkpoint.json')
    
    if checkpoint_path.exists() and info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
        checkpoint_df = pd.read_parquet(checkpoint_path)
        rephrased = checkpoint_df['question_rephrased'].tolist()
        print(f"Resuming from checkpoint at index {info['last_idx']}/{info['total']}")
        return info['last_idx'], rephrased
    
    return 0, []


def cleanup_checkpoint(output_path: Path):
    """Remove checkpoint files after successful completion."""
    checkpoint_path = output_path.with_suffix('.checkpoint.parquet')
    info_path = output_path.with_suffix('.checkpoint.json')
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if info_path.exists():
        info_path.unlink()


def augment_dataset(input_path: Path, output_path: Path, batch_size: int = 50, max_concurrent: int = 20):
    """Augment a single dataset file."""
    global shutdown_flag
    
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*60}")
    
    # Check if already completed
    if output_path.exists():
        print(f"Output file already exists: {output_path.name}. Skipping...")
        return True
    
    # Load data
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples")
    
    # Load checkpoint if exists
    start_idx, existing_rephrased = load_checkpoint(output_path)
    
    # Get questions
    questions = df['question'].tolist()
    
    # Rephrase using async batch processing
    start_time = time.time()
    rephrased_questions, last_idx = asyncio.run(
        process_batch(questions, start_idx, existing_rephrased, batch_size, max_concurrent)
    )
    elapsed = time.time() - start_time
    
    if shutdown_flag:
        # Save checkpoint and exit
        save_checkpoint(df, rephrased_questions, output_path, last_idx)
        return False
    
    if len(rephrased_questions) != len(questions):
        print(f"Warning: Incomplete processing ({len(rephrased_questions)}/{len(questions)})")
        save_checkpoint(df, rephrased_questions, output_path, last_idx)
        return False
    
    print(f"Rephrasing completed in {elapsed:.1f}s ({len(questions)/elapsed:.1f} questions/sec)")
    
    # Add rephrased questions as a new column
    df['question_rephrased'] = rephrased_questions
    
    # Save augmented data
    df.to_parquet(output_path, index=False)
    print(f"Saved augmented data to: {output_path.name}")
    
    # Cleanup checkpoint files
    cleanup_checkpoint(output_path)
    
    # Show some examples
    print("\nSample rephrasing examples:")
    for i in range(min(3, len(df))):
        print(f"  Original:   {df['question'].iloc[i]}")
        print(f"  Rephrased:  {df['question_rephrased'].iloc[i]}")
        print()
    
    return True


def main():
    global shutdown_flag
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="Augment QA datasets with rephrased questions")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent API requests")
    parser.add_argument("--datasets", nargs="+", default=["cwq", "webqsp"], help="Datasets to augment")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to augment")
    args = parser.parse_args()
    
    data_dir = Path(__file__).parent
    
    for dataset in args.datasets:
        for split in args.splits:
            if shutdown_flag:
                print("Shutdown requested. Exiting...")
                sys.exit(0)
                
            input_file = data_dir / f"{dataset}_{split}.parquet"
            output_file = data_dir / f"{dataset}_{split}_augmented.parquet"
            
            if input_file.exists():
                success = augment_dataset(input_file, output_file, args.batch_size, args.max_concurrent)
                if not success:
                    print(f"Processing interrupted for {input_file.name}. Run again to resume.")
                    sys.exit(0)
            else:
                print(f"Skipping {input_file.name} - file not found")
    
    print("\n" + "="*60)
    print("All datasets augmented successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
