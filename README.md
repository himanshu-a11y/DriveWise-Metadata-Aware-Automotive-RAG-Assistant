# DriveWise 🚗

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/AI-HuggingFace-F9AB00?logo=huggingface)](https://huggingface.co/)
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-Sentence%20Transformers-orange)](#)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-red)](#)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Data Science Internship Project · Metadata-Aware Automotive RAG Assistant**  
**Author:** Himanshu &nbsp;|&nbsp; **Domain:** Generative AI · Natural Language Processing · Retrieval-Augmented Generation

</div>

---

## 📌 Executive Summary
Purchasing a vehicle involves analyzing massive, 50+ page official brochures to find specific specifications. **DriveWise AI** solves this by providing a highly structured **Retrieval-Augmented Generation (RAG)** assistant that answers natural-language questions using **only** a selected car's official brochure content.

Instead of relying on a generalized AI model's training memory (which can hallucinate or be outdated), DriveWise retrieves the exact relevant section of the brochure and generates an answer grounded strictly in that text, complete with transparent source attribution.

**Key Value Propositions:**
- **Metadata-Aware Filtering:** Every chunk is tagged (Brand, Model, Section, Page). The RAG pipeline filters by this metadata first, ensuring answers never mix up different cars.
- **Intelligent Chunking:** Brochures are logically split by sections (e.g., Engine, Safety) rather than random text limits, preserving full context.
- **Advanced Retrieval Pipeline:** A fast bi-encoder (`all-MiniLM-L6-v2`) + FAISS for semantic search, followed by a powerful cross-encoder (`ms-marco-MiniLM-L-6-v2`) for precision re-ranking.
- **Grounded Generation:** A local LLM (`flan-t5-small`) answers questions based solely on retrieved context, gracefully declining out-of-scope questions.
- **Interactive Executive Streamlit Dashboard:** A commercial-grade web application for real-time querying and source-tracking.

---

## 🏆 RAG Evaluation Suite
The pipeline includes a custom evaluation module measuring standard RAG metrics against a predefined test set:

| Metric | Description | Goal |
|:---|:---|:---:|
| 🥇 **Answer Correctness** | Does the generated answer contain the expected factual keyword? | **100%** |
| 🥈 **Faithfulness** | Is the answer strictly grounded in the retrieved brochure text? | **High** |
| 🥉 **Context Relevance** | Did the vector search find the correct brochure section? | **100%** |

---

## 📁 Project Architecture & Structure

```text
DriveWise/
│
├── 📓 DriveWise.ipynb      # Step-by-step Jupyter Notebook walkthrough (Executes whole pipeline)
│
├── 🐍 src/                 # Production Modular Python Package
│   ├── chunker.py          # Data ingestion & metadata-aware chunking
│   ├── retriever.py        # Embeddings & FAISS vector search
│   ├── reranker.py         # Cross-encoder relevance scoring engine
│   ├── generator.py        # LLM answer generation with FLAN-T5
│   ├── pipeline.py         # End-to-end RAG orchestration pipeline
│   ├── evaluator.py        # RAG metric evaluation suite
│   └── logger.py           # JSONL query & performance logger
│
├── 📂 data/
│   └── brochures/          # Raw automotive text data (Hyundai, Tata, Maruti Suzuki)
│
├── 📊 app_web.py           # Commercial Streamlit Web Application
├── 💻 app.py               # Command-Line Interface (CLI)
├── 📦 requirements.txt     # Python dependencies
└── 📝 README.md            # Project homepage
```

---

## 🚀 Getting Started (Local Setup)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard
```bash
streamlit run app_web.py
```
*(The embedding and generation models will download automatically from Hugging Face on the first run, totaling ~250MB).*

### 3. Run the Command-Line Interface
```bash
python app.py
```

---

<div align="center">

**Prepared by Himanshu · Data Science Internship Project**

</div>
