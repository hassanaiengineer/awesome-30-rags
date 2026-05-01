# 06 — Hybrid RAG API

**Date:** TBD

**Built by Hassan Khan** • RAG Series

Hybrid retrieval pattern (scaffolding) behind a FastAPI API.

## Run

```bash
pip install -r requirements.txt
python -m app.ingest
uvicorn api.main:app --reload
```

## Endpoints

- `GET /meta`
- `POST /ask` — `{ "query": "..." }`
