# 03 — Self-RAG API

**Date:** TBD

**Built by Hassan Khan** • RAG Series

Retrieval + self-evaluation loop that can regenerate when answers are weak.

## Run

```bash
pip install -r requirements.txt
python -m app.ingest
uvicorn api.main:app --reload
```

## Endpoints

- `GET /meta`
- `POST /ask` — `{ "query": "..." }`
