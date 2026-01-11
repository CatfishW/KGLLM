# Anti-GPTZero Paper Humanizer - FREE VERSION

Humanize academic papers to bypass AI detection (GPTZero, Turnitin) using **100% FREE** open-weight models.

## 🏆 Best Option: HuggingFace PEGASUS (Recommended)

**Completely free, runs locally, best quality for academic text.**

```bash
# Install dependencies (one-time)
pip install transformers torch sentencepiece requests

# Run (model downloads automatically on first run, ~2GB)
python humanize_paper.py lam_main_latest.tex --api huggingface --model pegasus
```

## All Free Options

| Method | Quality | Speed | Size | Requirements |
|--------|---------|-------|------|--------------|
| **HuggingFace PEGASUS** ⭐ | Best | Medium | 2GB | GPU recommended |
| HuggingFace T5 | Good | Fast | 900MB | None |
| Rule-Based | Basic | Instant | 0 | None |
| Ollama | Best | Medium | 4GB+ | Ollama install |
| Web Scraping | Good | Slow | 0 | Selenium |

---

## Quick Start

```bash
# Install
pip install transformers torch sentencepiece requests

# Run with PEGASUS (RECOMMENDED)
python humanize_paper.py lam_main_latest.tex

# Output: lam_main_latest_humanized.tex
```

## HuggingFace Models

| Model | HuggingFace ID | Size | Best For |
|-------|---------------|------|----------|
| `pegasus` | tuner007/pegasus_paraphrase | ~2GB | Academic text |
| `t5` | humarin/chatgpt_paraphraser_on_T5_base | ~900MB | Faster processing |

```bash
# Use PEGASUS (default, best quality)
python humanize_paper.py paper.tex --model pegasus

# Use T5 (smaller, faster)
python humanize_paper.py paper.tex --model t5
```

## Other Options

### Rule-Based (No Downloads)
```bash
python humanize_paper.py paper.tex --api rulebased
```

### Ollama (Requires Installation)
```bash
# Install Ollama first: https://ollama.ai/
ollama pull llama3
python humanize_paper.py paper.tex --api ollama --model llama3
```

## Command Reference

```bash
python humanize_paper.py INPUT.tex \
    --api huggingface \   # huggingface, ollama, rulebased, webscrape, apify
    --model pegasus \     # pegasus, t5 (for HuggingFace) or llama3, mistral (for Ollama)
    --output OUTPUT.tex \ # Custom output path
    --delay 2.0 \         # Delay between sections
    --verbose             # Debug logging
```

## What Gets Preserved

✅ Mathematical equations (`$...$`, `\begin{equation}`)  
✅ Citations (`\cite{...}`)  
✅ Tables, figures, algorithms  
✅ All LaTeX formatting  

## Output Files

- `lam_main_latest_humanized.tex` - Humanized paper
- `lam_main_latest_humanized_report.txt` - Report
- `humanize_paper.log` - Log file

## Tips

1. **First run** takes longer (model download ~2GB)
2. **GPU** speeds up processing significantly
3. Test output with [gptzero.me](https://gptzero.me)
