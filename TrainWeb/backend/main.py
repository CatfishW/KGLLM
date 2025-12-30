from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from log_parser import LogParser
import os
import uvicorn

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev/tunnel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint for tunnel/monitoring."""
    return {"status": "ok"}

# Configuration
LOG_DIR = "/data/Yanlai/KGLLM/Core/kg_retriever/"

def get_latest_log_file():
    """Find the most recently modified .log file in the directory, preferring training logs."""
    import glob
    # Get all log files
    all_logs = glob.glob(os.path.join(LOG_DIR, "*.log"))
    if not all_logs:
        return None
    
    # Filter for training logs (exclude eval logs)
    training_logs = [f for f in all_logs if "training_" in os.path.basename(f) and "eval" not in os.path.basename(f)]
    
    # If we have training logs, sort those. Otherwise fallback to all logs.
    candidates = training_logs if training_logs else all_logs
    
    # Sort by modification time, newest first
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]

# State management
class AppState:
    def __init__(self):
        self.current_log_file = None
        self.parser = None
        self.last_log_check = 0
        self.cached_eval = {"time": 0, "data": None}

state = AppState()

def get_parser():
    """Get or update the global parser instance."""
    import time
    
    # Check for new logs every 5 seconds at most
    now = time.time()
    if now - state.last_log_check > 5:
        latest = get_latest_log_file()
        state.last_log_check = now
        
        # If log file changed or parser not initialized
        if latest != state.current_log_file:
            state.current_log_file = latest
            state.parser = LogParser(latest) if latest else None
    
    # Update parser (incremental read)
    if state.parser:
        state.parser.update()
        
    return state.parser

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/info")
def get_info():
    parser = get_parser()
    if not parser:
        return {"config": {}, "error": "No log file found"}
    
    data = parser.get_data()
    return {"config": data.get("config", {}), "log_file": os.path.basename(parser.log_path)}

@app.get("/metrics")
def get_metrics():
    parser = get_parser()
    if not parser:
        return {"metrics": [], "error": "No log file found"}
    
    data = parser.get_data()
    return {"metrics": data.get("metrics", [])}

@app.get("/logs")
def get_logs():
    parser = get_parser()
    if not parser:
        return {"logs": [], "error": "No log file found"}
    
    data = parser.get_data()
    return {"logs": data.get("raw_logs", [])}

@app.get("/architecture")
def get_architecture():
    parser = get_parser()
    if not parser:
        return {"architecture": {}, "error": "No log file found"}
    
    arch = parser.parse_model_architecture()
    return {"architecture": arch, "log_file": os.path.basename(parser.log_path)}

@app.get("/eval")
def get_eval_results(limit: int = 3):
    """Find all evaluation results and return best ones."""
    import time
    
    # Check cache (30s TTL)
    now = time.time()
    # DISABLE CACHE FOR DEBUGGING
    # if state.cached_eval["data"] and now - state.cached_eval["time"] < 30:
    #    return state.cached_eval["data"]
        
    import glob
    import csv
    import json
    
    print(f"Fetching eval results with limit={limit}...")
    
    # ... (rest of logic) ...
    # Instead of rewriting the whole function again to wrap it, let's optimize the existing one.
    # Actually, we should refactor it to cache the HEAVY PART (file reading).
    
    # We will implement a helper `_fetch_eval_data` and cache that.
    
    if state.cached_eval["data"] and now - state.cached_eval["time"] < 60: # 60s cache
         # Check if we need to slicing. The cached data contains 'all_results' which has 'examples'.
         # We just need to slice examples for each result? No, the cache stores everything.
         # But the previous implementation sliced examples BEFORE appending to results.
         # So we should probably cache the RAW results before slicing?
         # Or just cache the result with a default high limit (e.g. 10) and slice on return.
         pass

    # Actually, let's just do simple caching. 
    # If cache exists and valid, return it (ignoring limit for now or dealing with it)
    # But wait, limit changes what we parse!
    # If we request limit=50, but cache has limit=3, we miss data.
    # So we should only use cache if it was generated with a limit >= requested limit.
    # Or, simpler: Just cache with a high limit (e.g., 50) and slice in the return.
    
    pass 
    
    # RE-IMPLEMENTATION WITH CACHING
    
    # 1. Check if we have valid cache
    if state.cached_eval["data"] and (now - state.cached_eval["time"] < 60):
        # We assume cache has enough data (we will always fetch max 50 for cache)
        # But wait, we can't easily deep-slice the cache without modifying it.
        # So for now, let's just re-run if limit > 3? No that's slow.
        # Let's just return the cached object but slice the examples in 'all_results'.
        
        cached = state.cached_eval["data"]
        # Deep copy to avoid modifying cache? Or just construct new response.
        # It's expensive to deep copy.
        # Let's just return cached data. 
        # If the user asks for limit=3 and we cached limit=50, the UI receives 50. 
        # Is that bad? 50 examples isn't huge JSON. It's fine.
        return cached

    eval_dirs = []
    results = []
    
    # Find all directories with eval results
    base_dir = "/data/Yanlai/KGLLM/Core/kg_retriever"
    
    # Pattern 1: outputs_eval_* directories
    for d in glob.glob(os.path.join(base_dir, "outputs_eval_*")):
        if os.path.isdir(d):
            eval_dirs.append(d)
    
    # Pattern 2: outputs_*_pretty directories
    for d in glob.glob(os.path.join(base_dir, "outputs_*_pretty")):
        if os.path.isdir(d):
            eval_dirs.append(d)
    
    # Pattern 3: outputs_fullkg/eval_results
    fullkg_eval = os.path.join(base_dir, "outputs_fullkg/eval_results")
    if os.path.isdir(fullkg_eval):
        eval_dirs.append(fullkg_eval)
    
    # Deduplicate directories
    eval_dirs = sorted(list(set(eval_dirs)))
    
    # Always fetch enough examples
    parse_limit = max(limit, 50)

    for eval_dir in eval_dirs:
        # Look for metrics.csv in subdirectories (dataset-specific)
        for metrics_file in glob.glob(os.path.join(eval_dir, "**/metrics.csv"), recursive=True):
            try:
                with open(metrics_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Get parent directory name as dataset
                        parent_dir = os.path.dirname(metrics_file)
                        dataset = os.path.basename(parent_dir)
                        eval_name = os.path.basename(eval_dir)
                        
                        # Check for results.json with sample examples
                        results_json = os.path.join(parent_dir, "results.json")
                        sample_examples = []
                        
                        if os.path.exists(results_json):
                            try:
                                # Logic to parse sample predictions
                                # We need enough data to find 'parse_limit' examples.
                                # Heuristic: each example is approx 1KB max. So 1000 examples = 1MB.
                                # But let's afford 5MB to be safe if limit is high.
                                read_size = 5000000 if parse_limit > 50 else 1000000 
                                
                                import re
                                with open(results_json, 'r') as rj:
                                    chunk = rj.read(read_size) 
                                    
                                examples_found = 0
                                # Find all start indices of objects using a simpler regex
                                starts = [m.start() for m in re.finditer(r'^\s*\{', chunk, re.MULTILINE)]
                                seen_ids = set()
                                
                                for start in starts:
                                    if examples_found >= parse_limit: break
                                    
                                    # Take a slice that should contain the object
                                    block = chunk[start:start+5000] # Increased buffer
                                    
                                    # Just check if this block looks like a valid result entry
                                    if '"question":' not in block or '"top_pred_path":' not in block:
                                        continue
                                        
                                    # Check if rank is 1 (we only want top-1 for samples usually)
                                    if '"rank": 1' in block or '"rank": 1.0' in block:
                                        # Extract id
                                        id_match = re.search(r'"id":\s*"([^"]+)"', block)
                                        obj_id = id_match.group(1) if id_match else None
                                        
                                        # Extract question
                                        q_match = re.search(r'"question":\s*"([^"]+)"', block)
                                        
                                        # Deduplicate
                                        dedup_key = obj_id if obj_id else (q_match.group(1) if q_match else None)
                                        if dedup_key and dedup_key in seen_ids:
                                            continue
                                        if dedup_key:
                                            seen_ids.add(dedup_key)

                                        p_match = re.search(r'"top_pred_path":\s*\[(.*?)\]', block, re.DOTALL)
                                        
                                        if q_match and p_match:
                                            question = q_match.group(1)
                                            # Clean up path string
                                            path_raw = p_match.group(1)
                                            path_list = [p.strip(' "\n') for p in path_raw.split(',')]
                                            path_list = [p for p in path_list if p] # filter empty
                                            
                                            sample_examples.append({
                                                "question": question,
                                                "top_pred_path": path_list,
                                                "gt_path": [], 
                                                "rank": 1
                                            })
                                            
                                            # Try to find gt_path in the same block
                                            gt_match = re.search(r'"gt_path":\s*\[(.*?)\]', block, re.DOTALL)
                                            if gt_match:
                                                gt_raw = gt_match.group(1)
                                                gt_list = [p.strip(' "\n') for p in gt_raw.split(',')]
                                                gt_list = [p for p in gt_list if p]
                                                sample_examples[-1]["gt_path"] = gt_list

                                            examples_found += 1
                                    
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                print(f"Error parsing examples in {results_json}: {e}")
                                pass
                        
                        results.append({
                            "eval_name": eval_name,
                            "dataset": dataset,
                            "hits@1": float(row.get("hits@1", 0)),
                            "hits@3": float(row.get("hits@3", 0)),
                            "hits@5": float(row.get("hits@5", 0)),
                            "hits@10": float(row.get("hits@10", 0)),
                            "mrr": float(row.get("mrr", 0)),
                            "total": int(row.get("total", 0)),
                            "path": metrics_file,
                            "examples": sample_examples # Return all parsed examples (up to 50)
                        })
            except Exception as e:
                continue
    
    # Sort by Hits@1 descending
    results.sort(key=lambda x: x["hits@1"], reverse=True)
    
    # Group by dataset
    by_dataset = {}
    for r in results:
        ds = r["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(r)
    
    response_data = {
        "all_results": results,
        "by_dataset": by_dataset,
        "total_evals": len(results)
    }
    
    # Update cache
    state.cached_eval = {
        "time": now,
        "data": response_data
    }
    
    return response_data

if __name__ == "__main__":
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    uvicorn.run(app, host="0.0.0.0", port=32026)

