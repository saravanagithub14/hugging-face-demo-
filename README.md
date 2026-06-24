# 🤗 HuggingFace in 5 Scenes

A Jupyter notebook demo that covers the HuggingFace ecosystem from model discovery to local inference.

The demo is implemented in:
- `hf_demo_fixed (1).ipynb`

## What You'll Learn

| Scene | Subtopic | Key Concept |
|-------|----------|-------------|
| 1 | **HF Hub** | Discover models using `huggingface_hub` |
| 2 | **Transformers Library** | Tokenization, model input, logits, probabilities |
| 3 | **Datasets Library** | Load and inspect HuggingFace datasets |
| 4 | **Pipelines** | Run sentiment, NER, translation, summarization |
| 5 | **Open-Source Models** | Local inference with Flan-T5 and GPT-style models |

---

## Setup

1. Open this folder in VS Code or Jupyter Lab.
2. Create and activate a Python environment.

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install the notebook dependencies.

```bash
pip install transformers datasets huggingface-hub torch sentencepiece sacremoses
```

> ⚠️ First run will download models and datasets. Ensure a stable internet connection.

## Run the Notebook

Open `hf_demo_fixed (1).ipynb` in VS Code or Jupyter and run the cells top to bottom.

---

## Scene-by-Scene Guide

### 🏪 Scene 1 — HuggingFace Hub
- Uses `HfApi()` from `huggingface_hub`
- Lists top `text-classification` models
- Prints model IDs and download counts

### ⚙️ Scene 2 — Transformers Library
- Loads `distilbert-base-uncased-finetuned-sst-2-english`
- Tokenizes text and shows token IDs
- Runs a forward pass and converts logits to sentiment probabilities

### 📦 Scene 3 — Datasets Library
- Loads the `stanfordnlp/imdb` dataset
- Prints dataset metadata and a row preview
- Shows labels and example text

### 🪄 Scene 4 — Pipelines
- Runs sentiment analysis with `pipeline("sentiment-analysis")`
- Runs NER with `pipeline("ner", aggregation_strategy="simple")`
- Loads translation and summarization models directly using `AutoTokenizer` and `AutoModelForSeq2SeqLM`
- Demonstrates English→French translation and text summarization without `pipeline()`

### 🤖 Scene 5 — Running Open-Source Models Locally
- Loads `google/flan-t5-small` and runs an English→French prompt
- Loads `distilgpt2` for local text generation
- Shows how to run open-source models without external API keys

---

## Models Used in the Notebook

| Model | Used In |
|-------|---------|
| `distilbert-base-uncased-finetuned-sst-2-english` | Scene 2, Scene 4 sentiment |
| `sshleifer/distilbart-cnn-6-6` | Scene 4 summarization |
| `Helsinki-NLP/opus-mt-en-fr` | Scene 4 translation |
| `google/flan-t5-small` | Scene 5 local seq2seq inference |
| `distilgpt2` | Scene 5 local text generation |

> Note: The notebook does not use `facebook/opt-125m` or `dslim/bert-base-NER`.

---

## Troubleshooting

- If the first run is slow, models are downloading and will be cached.
- If you see `sentencepiece` errors, install `sentencepiece` and `sacremoses`.
- For GPU acceleration, install the CUDA build of PyTorch from https://pytorch.org.

---

## Teaching Notes

This notebook is a quick, hands-on tour of how HuggingFace works:

`Hub → Transformers → Datasets → Pipelines → Local Models`

Use the notebook as an interactive workshop: run each cell, inspect the outputs, and compare raw model usage with the simplified pipeline API.
