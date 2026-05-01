# 07 — Graph RAG API

**Date:** TBD

**Built by Hassan Khan** • RAG Series

Graph-style retrieval pattern (scaffolding) exposed through FastAPI.

## Run

```bash
pip install -r requirements.txt
python -m app.ingest
uvicorn api.main:app --reload
```

## Endpoints

- `GET /meta`
- `POST /ask` — `{ "query": "..." }`
