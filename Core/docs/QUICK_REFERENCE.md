# Quick Reference: SOTA Relation Chain Generation Methods

## 🎯 One-Line Recommendations

- **Fastest & Best Quality**: Autoregressive Transformer + Beam Search
- **Best with Pretrained Models**: T5-base/large
- **Best for Graph Structure**: GNN + Autoregressive Decoder
- **Current Method**: Discrete Diffusion (slower but very diverse)

---

## ⚙️ Configuration Quick Switches

### Switch to Autoregressive Model:
```yaml
model_type: autoregressive
decoding_strategy: beam_search
beam_width: 5
```

### Switch to T5:
```yaml
model_type: t5
pretrained_model: t5-base
fine_tune: true
```

### Switch to GNN+Decoder:
```yaml
model_type: gnn_decoder
gnn_type: gat
gnn_layers: 3
decoder_layers: 6
```

---

## 📝 Key Parameters

### Decoding Strategies:
- `greedy`: Fastest, lowest quality
- `beam_search`: Best quality, moderate speed (recommended)
- `nucleus`: Good diversity, adaptive
- `top_k`: Controlled diversity
- `contrastive`: Reduces repetition

### Beam Search Parameters:
- `beam_width`: 5-20 (higher = better quality, slower)
- `length_penalty`: 0.6-0.8 (penalize longer sequences)

### Sampling Parameters:
- `temperature`: 1.0 (lower = more deterministic)
- `top_p`: 0.9-0.95 (nucleus sampling threshold)
- `top_k`: 10-50 (top-k sampling)

---

## 🚀 Implementation Priority

1. **Phase 1** (Easy): Add beam search to current model
2. **Phase 2** (Medium): Implement autoregressive transformer
3. **Phase 3** (Advanced): Add T5 or GNN+Decoder

---

## 📊 Expected Performance

| Method | Inference (ms) | Quality | Diversity |
|--------|---------------|---------|-----------|
| Diffusion | 200-1000 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Autoregressive | 10-50 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| T5 | 10-50 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| GNN+Decoder | 20-100 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 🔗 Files

- **Full Documentation**: `SOTA_RELATION_CHAIN_METHODS.md`
- **Summary**: `SOTA_METHODS_SUMMARY.md`
- **Configs**: `configs/autoregressive.yaml`, `configs/t5.yaml`, `configs/gnn_decoder.yaml`

