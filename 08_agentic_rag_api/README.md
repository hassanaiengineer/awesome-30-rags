# 08 — Agentic RAG API

**Date:** TBD

**Built by Hassan Khan** • RAG Series

Agentic orchestration pattern (scaffolding) as a FastAPI service.

## Run

```bash
pip install -r requirements.txt
python -m app.ingest
uvicorn api.main:app --reload
```

## Endpoints

- `GET /meta`
- `POST /ask` — `{ "query": "..." }`
