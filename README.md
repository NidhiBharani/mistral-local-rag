# Mistral RAG Demo

A local Retrieval-Augmented Generation (RAG) system for medical Q&A on **refractive eye surgery and vision correction**. Answers are grounded in a curated corpus of peer-reviewed ophthalmology papers using a quantized Mistral-7B model — no cloud API required.

> For full architecture diagrams, design decisions, and configuration reference, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Features

- Chat interface powered by Streamlit
- Mistral-7B-Instruct (4-bit NF4 quantized) running locally on GPU
- Semantic retrieval via Qdrant vector database with MMR re-ranking
- Automatic ingestion of new PDFs on startup
- Evaluation with RAGAS (metrics) and Giskard (model scanning)

## Quick Start

### Prerequisites

- Docker
- Python 3.9+
- CUDA-capable GPU (tested on NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM)

### 1. Start Qdrant

```bash
cd RAG/
docker compose up -d
```

### 2. Install dependencies

```bash
pip install -r RAG/requirements.txt
```

### 3. Add documents

Place `.pdf` files into the `data/` directory. New files are detected and ingested automatically on app startup.

### 4. Run the app

```bash
cd RAG/
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`. The Qdrant dashboard is at `http://localhost:6333/dashboard`.

## Project Structure

```
mistral-demo/
├── RAG/                    # Core application (LLM, retriever, Streamlit UI)
│   ├── embed/              # Embedding model + Qdrant collection management
│   ├── ingest/             # PDF loading and chunking
│   ├── retrievers/         # MultiQueryRetriever (prepared, disabled)
│   └── app/                # Caching and prompt utilities
├── RAGAS/                  # Synthetic test set generation
├── giskard/                # Model quality scanning
└── data/                   # Ophthalmology PDF corpus
```

## Stack

| Component | Technology |
|---|---|
| LLM | Mistral-7B-Instruct-v0.1 (4-bit quantized) |
| Embeddings | sentence-transformers/all-mpnet-base-v2 |
| Vector DB | Qdrant |
| Orchestration | LangChain |
| UI | Streamlit |
| Evaluation | RAGAS + Giskard |

## Evaluation

**Generate a test set (requires `OPENAI_API_KEY`):**

```bash
cd RAGAS/
pip install -r requirements.txt
python testquestiongenerator.py
```

This produces 10 synthetic Q&A pairs (50% simple, 25% reasoning, 25% multi-context) for benchmarking retrieval and answer quality.
