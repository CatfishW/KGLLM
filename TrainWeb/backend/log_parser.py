import re
import os
from typing import Dict, List, Optional
from datetime import datetime

class LogParser:
    def __init__(self, log_path: str):
        self.log_path = log_path
        # Pattern to capture progress bars like:
        # Epoch 2:  10%|█         | 148/1429 [00:23<03:22,  6.32it/s, v_num=1852, train/loss=3.260, train/acc=0.125]
        self.progress_pattern = re.compile(
            r"Epoch\s+(\d+):.*train/loss=([0-9.]+),\s+train/acc=([0-9.]+)"
        )
        self.config_pattern = re.compile(r"([A-Za-z_]+):\s+(.*)")
        
        # Patterns for model architecture
        self.layer_pattern = re.compile(r"\((\w+)\):\s+(\w+)\((.*)\)")
        self.param_pattern = re.compile(r"(\d+(?:,\d+)*)\s+(?:Trainable|Non-trainable)\s+params")
        self.total_param_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*([MKB])?\s+(?:Total|Trainable)\s+params", re.IGNORECASE)

        # State for incremental parsing
        self.metrics = []
        self.config = {}
        self.raw_lines = []
        self.last_pos = 0
        self.last_mtime = 0
        self.architecture = None

    def update(self):
        """Update parsed data by reading only new lines from the file."""
        if not os.path.exists(self.log_path):
            return

        try:
            stat = os.stat(self.log_path)
            # Check if file has changed
            if stat.st_size == self.last_pos and stat.st_mtime == self.last_mtime:
                return

            # If file shrunk (was overwritten), reset
            if stat.st_size < self.last_pos:
                self.last_pos = 0
                self.metrics = []
                self.raw_lines = []
                self.config = {}
                self.architecture = None

            with open(self.log_path, 'r', errors='ignore') as f:
                f.seek(self.last_pos)
                new_lines = f.readlines()
                self.last_pos = f.tell()
                self.last_mtime = stat.st_mtime

                for line in new_lines:
                    # Keep last 1000 raw lines
                    self.raw_lines.append(line)
                    if len(self.raw_lines) > 1000:
                        self.raw_lines = self.raw_lines[-1000:]

                    # Parse metrics
                    match = self.progress_pattern.search(line)
                    if match:
                        epoch = int(match.group(1))
                        loss = float(match.group(2))
                        acc = float(match.group(3))
                        self.metrics.append({
                            "epoch": epoch,
                            "loss": loss,
                            "accuracy": acc,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    # Parse config (simple key-value pairs at start of file usually)
                    # We assume config is mostly at the beginning, but checking always is cheap enough for regex
                    conf_match = self.config_pattern.match(line.strip())
                    if conf_match:
                        key, val = conf_match.group(1), conf_match.group(2)
                        if " " not in key and len(key) < 30:
                             self.config[key] = val

        except Exception as e:
            print(f"Error updating log: {e}")

    def get_data(self) -> Dict:
        """Get the current cached data."""
        return {
            "metrics": self.metrics,
            "config": self.config,
            "raw_logs": self.raw_lines
        }

    def parse_model_architecture(self) -> Dict:
        """Parse model architecture based on current training log."""
        # Use cached architecture if available
        if self.architecture:
            return self.architecture

        MODELS_DIR = "/data/Yanlai/KGLLM/Core/kg_retriever/models"
        
        # Default architecture
        architecture = {
            "model_name": "Unknown Model",
            "model_type": "unknown",
            "total_params": 0,
            "trainable_params": 0,
            "frozen_params": 0,
            "hidden_dim": 768,
            "diffusion_steps": 10,
            "dropout": 0.1,
            "num_heads": 8,
            "encoder_name": "unknown",
            "batch_size": 0,
            "learning_rate": "0",
            "modules": [],
            "training_config": {}
        }
        
        try:
            # First, read the current log to detect which model is being trained
            if not os.path.exists(self.log_path):
                return architecture
            
            # Read first 10KB separately for config detection (usually enough)
            with open(self.log_path, 'r', errors='ignore') as f:
                log_content = f.read(20000) 
            
            # Detect model type from log - order matters (more specific first)
            if "Discrete Rank Diffusion" in log_content or "DCDR" in log_content or "discrete_diffusion_reranker" in log_content:
                architecture["model_type"] = "dcdr"
                architecture["model_name"] = "DiscreteRankDiffusion"
            elif "Diffusion Path Scorer" in log_content or "diffusion_reranker" in log_content:
                architecture["model_type"] = "diffusion_reranker"
                architecture["model_name"] = "DiffusionPathScorer"
            elif "GNN" in log_content or "gnn_retriever" in log_content:
                architecture["model_type"] = "gnn"
                architecture["model_name"] = "GNNRetriever"
            elif "PathRanker" in log_content or "path_ranker" in log_content:
                architecture["model_type"] = "path_ranker"
                architecture["model_name"] = "PathRankerModel"
            elif "DiffusionRetriever" in log_content or "diffusion_retriever" in log_content:
                architecture["model_type"] = "diffusion_retriever"
                architecture["model_name"] = "DiffusionRetriever"
            
            # Extract config from log header
            encoder_match = re.search(r'Encoder:\s*(.+)', log_content)
            if encoder_match:
                architecture["encoder_name"] = encoder_match.group(1).strip()
            
            hidden_match = re.search(r'Hidden dim:\s*(\d+)', log_content)
            if hidden_match:
                architecture["hidden_dim"] = int(hidden_match.group(1))
            
            # Extra DCDR-specific config
            layers_match = re.search(r'Transformer layers:\s*(\d+)', log_content)
            if layers_match:
                architecture["num_layers"] = int(layers_match.group(1))
            
            swap_match = re.search(r'Swap rate:\s*([\d.]+)', log_content)
            if swap_match:
                architecture["swap_rate"] = float(swap_match.group(1))
            
            pl_temp_match = re.search(r'PL temperature:\s*([\d.]+)', log_content)
            if pl_temp_match:
                architecture["pl_temperature"] = float(pl_temp_match.group(1))
            
            steps_match = re.search(r'Diffusion steps:\s*(\d+)', log_content)
            if steps_match:
                architecture["diffusion_steps"] = int(steps_match.group(1))
            
            batch_match = re.search(r'Batch size:\s*(\d+)\s*x\s*(\d+)\s*=\s*(\d+)', log_content)
            if batch_match:
                architecture["batch_size"] = int(batch_match.group(3))
                architecture["training_config"]["per_gpu_batch"] = int(batch_match.group(1))
                architecture["training_config"]["gpus"] = int(batch_match.group(2))
            
            lr_match = re.search(r'Learning rate:\s*([\d.e-]+)', log_content)
            if lr_match:
                architecture["learning_rate"] = lr_match.group(1)
            
            # Parse PyTorch Lightning model summary from log
            param_matches = re.findall(r'\|\s*(\w+)\s*\|\s*\w+\s*\|\s*([\d.]+\s*[MKB]?)\s*\|', log_content)
            
            # Look for param summary lines
            trainable_match = re.search(r'(\d+(?:\.\d+)?)\s*([MK])?\s+Trainable params', log_content)
            total_match = re.search(r'(\d+(?:\.\d+)?)\s*([MK])?\s+Total params', log_content)
            frozen_match = re.search(r'(\d+(?:\.\d+)?)\s*([MK])?\s+Non-trainable params', log_content)
            
            def parse_param(match):
                if not match:
                    return 0
                val = float(match.group(1))
                suffix = match.group(2) if match.lastindex >= 2 else None
                if suffix == 'M':
                    return int(val * 1_000_000)
                elif suffix == 'K':
                    return int(val * 1_000)
                return int(val)
            
            architecture["trainable_params"] = parse_param(trainable_match)
            architecture["total_params"] = parse_param(total_match)
            architecture["frozen_params"] = parse_param(frozen_match)
            
            # Generate modules based on detected model type
            hidden = architecture["hidden_dim"]
            dropout = architecture["dropout"]
            encoder_short = architecture["encoder_name"].split('/')[-1] if '/' in architecture["encoder_name"] else architecture["encoder_name"]
            num_layers = architecture.get("num_layers", 4)
            
            if architecture["model_type"] == "dcdr":
                architecture["modules"] = self._get_dcdr_modules(hidden, dropout, encoder_short, num_layers)
            elif architecture["model_type"] == "diffusion_reranker":
                architecture["modules"] = self._get_diffusion_reranker_modules(hidden, dropout, encoder_short)
            elif architecture["model_type"] == "gnn":
                architecture["modules"] = self._get_gnn_modules(hidden, dropout, encoder_short)
            elif architecture["model_type"] == "path_ranker":
                architecture["modules"] = self._get_path_ranker_modules(hidden, dropout, encoder_short)
            else:
                # Generic fallback
                architecture["modules"] = [
                    {"name": "Encoder", "type": "encoder", "frozen": True, "params": "~100M", "description": encoder_short, "layers": []},
                    {"name": "Classifier", "type": "mlp", "frozen": False, "params": "~5M", "description": "Output layer", "layers": []},
                ]

            self.architecture = architecture
                
        except Exception as e:
            architecture["error"] = str(e)
        
        return architecture
    
    def _get_dcdr_modules(self, hidden, dropout, encoder, num_layers):
        """Modules for Discrete Conditional Diffusion Reranking (DCDR)."""
        return [
            {
                "name": "1. Input Processing",
                "type": "flow",
                "frozen": "N/A",
                "params": "--",
                "description": "Encode Query (q) & Candidates (d)",
                "layers": []
            },
            {
                "name": "Text Encoder",
                "type": "encoder",
                "frozen": True,
                "params": f"~{33 if 'small' in encoder.lower() else 109}M",
                "description": f"Pre-trained {encoder}",
                "layers": [
                    {"name": "Embeddings", "spec": f"(vocab, {hidden})"},
                    {"name": "BertEncoder", "spec": f"{'6' if 'small' in encoder.lower() else '12'} layers"},
                    {"name": "MeanPooling", "spec": "attention-weighted avg"},
                ]
            },
            {
                "name": "2. Forward Diffusion",
                "type": "flow",
                "frozen": "N/A",
                "params": "--",
                "description": "Sample t ~ U(0,T), Add noise to ranks",
                "layers": [
                   {"name": "Noising", "spec": "x_t = x_0 * alpha + eps * sigma"} 
                ]
            },
            {
                "name": "Time Embedding",
                "type": "diffusion",
                "frozen": False,
                "params": f"~{(hidden*hidden*3)//1000000:.1f}M",
                "description": "Embed timestep t",
                "layers": [
                    {"name": "Sinusoidal", "spec": f"dim={hidden//2} → sin/cos"},
                    {"name": "Linear1", "spec": f"Linear({hidden}, {hidden*2})"},
                    {"name": "GELU", "spec": "activation"},
                    {"name": "Linear2", "spec": f"Linear({hidden*2}, {hidden})"},
                ]
            },
            {
                "name": "Position Embedding",
                "type": "diffusion",
                "frozen": False,
                "params": f"~{(200*hidden)//1000:.0f}K",
                "description": "Embed noisy rank positions",
                "layers": [
                    {"name": "Embedding", "spec": f"Embedding(200, {hidden})"},
                ]
            },
            {
                "name": "3. Feature Combination",
                "type": "flow",
                "frozen": "N/A",
                "params": "--",
                "description": "h = BERT(d) + Time(t) + Pos(rank)",
                "layers": []
            },
            {
                "name": "Denoising Transformer",
                "type": "attention",
                "frozen": False,
                "params": f"~{(hidden*hidden*4*num_layers)//1000000:.1f}M",
                "description": f"Self-Attention over candidates ({num_layers}L)",
                "layers": [
                    {"name": "TransformerEncoderLayer", "spec": f"x{num_layers}"},
                    {"name": "Self-Attention", "spec": f"MultiHead({hidden}, 8 heads)"},
                    {"name": "FFN", "spec": f"Linear({hidden}, {hidden*4}) → GELU → Linear"},
                    {"name": "LayerNorm", "spec": f"norm({hidden})"},
                ]
            },
            {
                "name": "Query Cross-Attention",
                "type": "attention",
                "frozen": False,
                "params": f"~{(hidden*hidden*4)//1000000:.1f}M",
                "description": "Attend to Query",
                "layers": [
                    {"name": "Candidates → Query", "spec": f"CrossAttn({hidden})"},
                    {"name": "Q/K/V proj", "spec": f"Linear({hidden}, {hidden}) x3"},
                ]
            },
            {
                "name": "Rank Head",
                "type": "mlp",
                "frozen": False,
                "params": f"~{(hidden*hidden*2)//1000000:.1f}M",
                "description": "Predict clean scores",
                "layers": [
                    {"name": "Linear1", "spec": f"Linear({hidden*2}, {hidden})"},
                    {"name": "GELU", "spec": "activation"},
                    {"name": "Dropout", "spec": f"p={dropout}"},
                    {"name": "Linear2", "spec": f"Linear({hidden}, 1)"},
                ]
            },
            {
                "name": "4. Loss Calculation",
                "type": "flow",
                "frozen": "N/A",
                "params": "--",
                "description": "Plackett-Luce(scores, ground_truth)",
                "layers": []
            },
        ]
    
    def _get_diffusion_reranker_modules(self, hidden, dropout, encoder):
        return [
            {
                "name": "Text Encoder",
                "type": "encoder",
                "frozen": True,
                "params": f"~{33 if 'small' in encoder.lower() else 109}M",
                "description": f"Pre-trained {encoder}",
                "layers": [
                    {"name": "Embeddings", "spec": f"(vocab, {hidden})"},
                    {"name": "Encoder", "spec": f"{'6' if 'small' in encoder.lower() else '12'} layers"},
                    {"name": "Pooler", "spec": f"Linear({hidden}, {hidden})"},
                ]
            },
            {
                "name": "Cross Attention",
                "type": "attention",
                "frozen": False,
                "params": f"~{(hidden*hidden*4)//1000000:.1f}M",
                "description": "Question-Path matching",
                "layers": [
                    {"name": "Q/K/V proj", "spec": f"Linear({hidden}, {hidden}) x3"},
                    {"name": "Attention", "spec": "softmax(QK^T/√d) × V"},
                    {"name": "Out proj", "spec": f"Linear({hidden}, {hidden})"},
                ]
            },
            {
                "name": "Noise Predictor",
                "type": "diffusion",
                "frozen": False,
                "params": f"~{(hidden*hidden*3)//1000000:.1f}M",
                "description": "Diffusion denoising",
                "layers": [
                    {"name": "Input", "spec": f"Linear({hidden}*3, {hidden})"},
                    {"name": "Hidden", "spec": f"Linear({hidden}, {hidden}) + GELU"},
                    {"name": "Output", "spec": f"Linear({hidden}, {hidden})"},
                ]
            },
            {
                "name": "Time Embedding",
                "type": "diffusion",
                "frozen": False,
                "params": f"~{(hidden*hidden*2)//1000000:.1f}M",
                "description": "Sinusoidal + MLP",
                "layers": [
                    {"name": "Sinusoidal", "spec": f"dim={hidden//2}"},
                    {"name": "MLP", "spec": f"Linear({hidden}, {hidden}) x2"},
                ]
            },
            {
                "name": "Score Head",
                "type": "mlp",
                "frozen": False,
                "params": f"~{(hidden*hidden)//1000000:.1f}M",
                "description": "Path ranking",
                "layers": [
                    {"name": "Linear1", "spec": f"Linear({hidden}*2, {hidden})"},
                    {"name": "Output", "spec": f"Linear({hidden}, 1)"},
                ]
            },
            {
                "name": "Hop Predictor",
                "type": "mlp",
                "frozen": False,
                "params": "~0.3M",
                "description": "Auxiliary task",
                "layers": [
                    {"name": "Linear1", "spec": f"Linear({hidden}, {hidden//2})"},
                    {"name": "Output", "spec": f"Linear({hidden//2}, 5)"},
                ]
            },
        ]
    
    def _get_discrete_diffusion_modules(self, hidden, dropout, encoder):
        return [
            {"name": "Text Encoder", "type": "encoder", "frozen": True, "params": "~100M", "description": f"Pre-trained {encoder}", "layers": []},
            {"name": "Discrete Diffusion", "type": "diffusion", "frozen": False, "params": "~5M", "description": "Discrete token diffusion", "layers": []},
            {"name": "Score Head", "type": "mlp", "frozen": False, "params": "~1M", "description": "Path ranking", "layers": []},
        ]
    
    def _get_gnn_modules(self, hidden, dropout, encoder):
        return [
            {"name": "Text Encoder", "type": "encoder", "frozen": True, "params": "~100M", "description": f"Pre-trained {encoder}", "layers": []},
            {"name": "GNN Layers", "type": "attention", "frozen": False, "params": "~10M", "description": "Graph Neural Network", "layers": []},
            {"name": "Readout", "type": "mlp", "frozen": False, "params": "~1M", "description": "Graph pooling", "layers": []},
        ]
    
    def _get_path_ranker_modules(self, hidden, dropout, encoder):
        return [
            {"name": "Text Encoder", "type": "encoder", "frozen": True, "params": "~100M", "description": f"Pre-trained {encoder}", "layers": []},
            {"name": "Bi-Encoder", "type": "attention", "frozen": False, "params": "~5M", "description": "Question-Path matching", "layers": []},
            {"name": "Score Head", "type": "mlp", "frozen": False, "params": "~1M", "description": "Final scoring", "layers": []},
        ]

