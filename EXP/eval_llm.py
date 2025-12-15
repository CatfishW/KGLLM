
import os
import json
import logging
import argparse
import pandas as pd
import requests
import re
from typing import List, Dict, Any, Set
from tqdm import tqdm
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import pipeline components
try:
    from EXP.pipeline import EnhancedKGRAGPipeline
    from EXP.config import RAGConfig
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from EXP.pipeline import EnhancedKGRAGPipeline
    from EXP.config import RAGConfig

def load_test_data(parquet_path: str) -> pd.DataFrame:
    """Load and preprocess test data."""
    logger.info(f"Loading data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Parse JSON columns if they are strings
    json_cols = ['q_entity', 'a_entity', 'graph', 'answer']
    for col in json_cols:
        if col in df.columns:
            # Check if first element is string
            if isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(lambda x: json.loads(x) if x else [])
    
    return df

class LocalGraphTraverser:
    """Traverses a local subgraph (list of triples)."""
    def __init__(self, triples: List[List[str]]):
        self.adj = {} # sub -> {pred -> [obj]}
        for s, p, o in triples:
            if s not in self.adj: self.adj[s] = {}
            if p not in self.adj[s]: self.adj[s][p] = []
            self.adj[s][p].append(o)
            
    def get_paths(self, start_entity: str, relation_chain: List[str]) -> List[str]:
        """Find all paths following the relation chain from start_entity."""
        results = []
        
        # DFS
        # State: (current_node, depth, path_history)
        stack = [(start_entity, 0, [start_entity])]
        
        # We need to match relation_chain[depth]
        # But we must find all full paths
        
        # Recursive approach for simplicity
        self._dfs(start_entity, relation_chain, 0, [start_entity], [], results)
        return results

    def _dfs(self, current_node: str, chain: List[str], depth: int, nodes: List[str], rels: List[str], results: List[str]):
        if depth == len(chain):
            # Formatted path string
            # e.g. "Entity --rel--> Entity2 --rel2--> Entity3"
            path_str = nodes[0]
            for i, r in enumerate(rels):
                path_str += f" --{r}--> {nodes[i+1]}"
            results.append(path_str)
            return

        target_rel = chain[depth]
        
        # Check adjacent
        if current_node in self.adj:
            # We match partial relations? 
            # The index stores "location.country.capital", splits by "."?
            # Or the index stores the FULL relation string as one hop?
            # Usually relation chain in index is list of relations.
            
            # Relation matching logic
            # In WebQSP, relations are like "location.country.capital" (one string).
            # So chain[depth] should allow exact match.
             
            if target_rel in self.adj[current_node]:
                for next_node in self.adj[current_node][target_rel]:
                    self._dfs(next_node, chain, depth+1, nodes + [next_node], rels + [target_rel], results)


class LLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.model = self._detect_model()
        
    def _detect_model(self) -> str:
        try:
            r = requests.get(f"{self.base_url}/v1/models")
            if r.status_code == 200:
                data = r.json()
                if 'data' in data and len(data['data']) > 0:
                    model_id = data['data'][0]['id']
                    logger.info(f"Detected model: {model_id}")
                    return model_id
        except Exception as e:
            logger.error(f"Failed to detect model: {e}")
        return "default"

    def query(self, prompt: str) -> str:
        try:
            # Standard OpenAI Chat Completion format
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 100
            }
            r = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
            else:
                logger.error(f"LLM Error {r.status_code}: {r.text}")
                return ""
        except Exception as e:
            logger.error(f"LLM Request Failed: {e}")
            return ""

def compute_metrics(predictions: List[str], ground_truths: List[List[str]]) -> Dict[str, float]:
    """Compute F1 and Hits (Exact Match approximation)."""
    f1_scores = []
    hits = []
    
    for pred, gts in zip(predictions, ground_truths):
        pred_norm = normalize_answer(pred)
        
        # Hits/Exact Match if ANY ground truth is in prediction (relaxed)
        # Or standard SQuAD F1
        best_f1 = 0
        is_hit = 0
        
        for gt in gts:
            gt_norm = normalize_answer(gt)
            if gt_norm in pred_norm or pred_norm in gt_norm: # Simple substring match as proxy
                if gt_norm == pred_norm:
                    best_f1 = 1.0
                    is_hit = 1
                else:
                    # Token f1
                    f1 = f1_score(pred_norm, gt_norm)
                    if f1 > best_f1: best_f1 = f1
                    if f1 > 0.5: is_hit = 1 # Loose hit
        
        f1_scores.append(best_f1)
        hits.append(is_hit)
        
    return {
        "f1": sum(f1_scores)/len(f1_scores) if f1_scores else 0,
        "hits": sum(hits)/len(hits) if hits else 0
    }

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

import string
from collections import Counter

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="Data/webqsp_final/test.parquet")
    parser.add_argument("--output_file", type=str, default="EXP/eval_results.json")
    parser.add_argument("--num_samples", type=int, default=100) # Limit for speed
    parser.add_argument("--top_k_retrieval", type=int, default=10) # Get top 10 chains
    args = parser.parse_args()
    
    # Load Data
    df = load_test_data(args.test_file)
    if args.num_samples:
        df = df.head(args.num_samples)
    
    # Init Pipeline (Baseline + Reranker + Better Embeddings)
    config = RAGConfig(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        use_reranker=True,
        rerank_top_k=50,
        index_dir="EXP/index",
    )
    pipeline = EnhancedKGRAGPipeline(
        index_path=config.index_dir,
        config=config
    )
    
    # Init LLM
    llm = LLMClient("https://game.agaii.org/llm")
    
    results = []
    
    logger.info("Starting evaluation loop...")
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                question = row['question']
                q_entities = row['q_entity'] # List of strings
                subgraph = row['graph'] # List of [s, p, o]
                ground_truth = row['answer']
                
                # 1. Retrieve
                retrieved_chains = pipeline.retriever.retrieve(question, top_k=args.top_k_retrieval)
                
                # 2. Instantiate
                traverser = LocalGraphTraverser(subgraph)
                instantiated_paths = []
                
                for p in retrieved_chains:
                    for ent in q_entities:
                        paths = traverser.get_paths(ent, p.relations)
                        instantiated_paths.extend(paths)
                        
                # Deduplicate
                instantiated_paths = list(set(instantiated_paths))
                
                # 3. Prompting
                if instantiated_paths:
                    context_paths = instantiated_paths[:15]
                    path_text = "\n".join([f"- {p}" for p in context_paths])
                    context_str = f"Relevant Paths:\n{path_text}"
                else:
                    context_str = "No relevant paths found."
                    
                prompt = f"""Question: {question}

{context_str}

Reason step-by-step to find the answer.
The answer must be based on the paths.
At the end, output the answer prefixed with 'Final Answer:'.

Example:
Reasoning: ...
Final Answer: Javascript

Final Answer:"""

                # 4. LLM Gen
                prediction_raw = llm.query(prompt)
                
                # Parse
                if "Final Answer:" in prediction_raw:
                    prediction = prediction_raw.split("Final Answer:")[-1].strip()
                else:
                    prediction = prediction_raw # Fallback
                
                # Check Context Recall
                context_hit = 0
                for path_str in instantiated_paths:
                    for gt in ground_truth:
                        if normalize_answer(gt) in normalize_answer(path_str):
                            context_hit = 1
                            break
                    if context_hit: break
                
                results.append({
                    "question": question,
                    "prediction": prediction,
                    "prediction_raw": prediction_raw,
                    "ground_truth": ground_truth,
                    "retrieved_initial": [p.relation_chain for p in retrieved_chains],
                    "instantiated_paths": instantiated_paths,
                    "context_hit": context_hit
                })
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue
                
    finally:
        # Compute Metrics (on whatever we have)
        if results:
            preds = [r['prediction'] for r in results]
            gts = [r['ground_truth'] for r in results]
            context_hits = [r.get('context_hit',0) for r in results]
            
            metrics = compute_metrics(preds, gts)
            context_recall = sum(context_hits) / len(context_hits) if context_hits else 0
            
            print("========================================")
            print(f"Evaluation Results (N={len(results)})")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Zero-Shot Hits@1 (Answer): {metrics['hits']:.4f}")
            print(f"Context Recall (Retrieval): {context_recall:.4f}")
            print("========================================")
            
            # Save
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
