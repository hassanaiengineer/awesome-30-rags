# 12 — Local RAG Agent (Agno + Ollama + Qdrant)

**Date:** TBD

**Built by Hassan Khan** • RAG Series

Local agent that uses Qdrant for retrieval and Ollama for inference.

## Run

```bash
pip install -r requirements.txt
python local_rag_agent.py
```

## Environment (optional)

- `QDRANT_URL` (default: `http://localhost:6333/`)
- `QDRANT_COLLECTION` (default: `local-rag-index`)
- `RAG_SOURCE_URL` (optional: preload a single URL; no default)
