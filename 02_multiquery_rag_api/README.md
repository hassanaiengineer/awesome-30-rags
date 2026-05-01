# 02 — Multi-Query RAG API

**Date:** TBD

**Built by Hassan Khan** • RAG Series

Generates multiple query variants to improve recall, then retrieves/answers via FastAPI.

## Run

```bash
pip install -r requirements.txt
python -m app.ingest
uvicorn api.main:app --reload
```

## Endpoints

- `GET /meta`
- `POST /ask` — `{ "query": "..." }`
