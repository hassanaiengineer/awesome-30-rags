# 05 — Corrective RAG API

**Date:** TBD

**Built by Hassan Khan** • RAG Series

Corrective flow: retrieve → answer → evaluate → refine.

## Run

```bash
pip install -r requirements.txt
python -m app.ingest
uvicorn api.main:app --reload
```

## Endpoints

- `GET /meta`
- `POST /ask` — `{ "query": "..." }`
