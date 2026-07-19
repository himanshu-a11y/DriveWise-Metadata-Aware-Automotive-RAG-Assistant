# DriveWise 🚗 — Metadata-Aware Automotive RAG Assistant

DriveWise is a Retrieval-Augmented Generation (RAG) assistant that answers natural-language
questions about a car using only that car's brochure content. Instead of relying on a
model's general knowledge (which can be outdated or simply wrong), it retrieves the exact
relevant section of the correct brochure and generates an answer grounded in that text,
along with the source it came from.

## Features

- **Brand & model selection** — the user picks a specific car before asking anything
- **Metadata-aware filtering** — every chunk is tagged with brand, model, section, page
  number, and brochure version, so retrieval never mixes up two different cars
- **Structured chunking** — brochures are split by logical section (engine, mileage,
  safety, dimensions, interior, infotainment) instead of arbitrary character counts
- **Semantic retrieval** — sentence-transformer embeddings (`all-MiniLM-L6-v2`) + a FAISS
  vector index for fast similarity search
- **Cross-encoder re-ranking** — a second, more accurate pass (`ms-marco-MiniLM-L-6-v2`)
  re-scores retrieved chunks before generation
- **Context window control** — only the most relevant chunk(s) reach the language model
- **Grounded generation** — a local LLM (`flan-t5-small`) answers strictly from retrieved
  context, with an honest "not found" response when the brochure doesn't cover a question
- **Full source attribution** — every answer cites brand, model, section, page number,
  file, and a unique chunk reference
- **Query logging** — every request logs the query, retrieved chunks, response time, and
  success/failure status
- **Evaluation suite** — Answer Correctness, Faithfulness, and Context Relevance, measured
  against a test set

## Architecture

```
User selects brand/model + asks a question
        |
        v
Metadata Filtering  ->  keep only chunks for the selected car
        |
        v
Embedding + Vector Search (FAISS)  ->  retrieve top candidate chunks
        |
        v
Cross-Encoder Re-Ranking  ->  reorder by true relevance
        |
        v
Context Window Control  ->  keep only the best chunk(s)
        |
        v
LLM Answer Generation (flan-t5-small)  ->  grounded, source-cited answer
        |
        v
Logging & Evaluation
```

## Getting Started

```bash
pip install -r requirements.txt
```

```bash
python app.py              # command-line interface
streamlit run app_web.py   # web dashboard
```

Or open `DriveWise.ipynb` in Jupyter and run the cells top to bottom for a guided,
step-by-step walkthrough of the pipeline.

The embedding, re-ranking, and generation models (~250MB total) download automatically
from Hugging Face the first time the app runs, and are cached locally after that.

## Dataset

Problem statement and dataset available on Kaggle:
[DriveWise Problem Statement](https://www.kaggle.com/datasets/salvaderron0013/drive-wise-problem-statement)

## Tech Stack

Python · sentence-transformers · FAISS · Hugging Face Transformers (flan-t5-small) ·
Cross-Encoder re-ranking · Streamlit
