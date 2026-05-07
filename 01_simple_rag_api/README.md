# 01 — Simple RAG API

**Date:** May 1, 2026

## Overview
Day 01 of the **30 Days of RAG** series: a minimal FastAPI RAG endpoint using local embeddings + a local FAISS index.

**No API keys required.** The “LLM” is a small mock generator so you can run everything offline.

## Architecture
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** FAISS (Local)
- **Framework:** FastAPI
- **LLM:** Local Mock LLM

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ingest data:
   ```bash
   python -m app.ingest
   ```
3. Run the API:
   ```bash
   uvicorn api.main:app --reload
   ```

Note: if your system uses `python3`, replace `python` with `python3` in commands.

## Example Request
```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is AI?"}'
```

## Extension Ideas
- Replace the mock LLM with an actual local LLM using Ollama or vLLM.
- Implement more advanced document chunking.
