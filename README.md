# 30 Days of RAG (Tutorial Series)

**Date:** May 1, 2026  
**Maintainer:** Hassan Khan 
**Email:** `hassanaiengineer@gmail.com`  
**LinkedIn:** https://www.linkedin.com/in/hassan-khan-4961b722b/

This repository is a **hands-on 30-day RAG series**: small projects (FastAPI + Streamlit) covering retrieval patterns, routing, hybrid search, knowledge graphs, multimodal (vision) RAG, and debugging/failure diagnostics.

Motivation and goals: see `MOTIVATION.md`.

## Tech Stack (Across the Series)

Not every tutorial uses every tool, but you’ll see many of these across days:

- FastAPI: https://fastapi.tiangolo.com/
- Uvicorn: https://www.uvicorn.org/
- Streamlit: https://streamlit.io/
- Sentence Transformers: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss
- Qdrant: https://qdrant.tech/
- Neo4j: https://neo4j.com/
- Ollama (local LLMs): https://ollama.com/

## How to Run a Tutorial

Each numbered folder is intended to be **runnable on its own**:

```bash
cd 01_simple_rag_api
pip install -r requirements.txt
python -m app.ingest
uvicorn api.main:app --reload
```

Some tutorials need API keys or services (Qdrant, Neo4j, etc.). Use `.env.example` as a starting point.

Note: if your system uses `python3`, replace `python` with `python3` in commands.

## Quick Start

Most tutorials follow this pattern:

```bash
cd 10_hybrid_search_rag_raglite
pip install -r requirements.txt
streamlit run main.py
```

API tutorials (01–08) use FastAPI:

```bash
cd 01_simple_rag_api
pip install -r requirements.txt
python -m app.ingest
uvicorn api.main:app --reload
```

## Tutorials

| Day | Folder | Type | What you’ll build | Key deps / services |
|---:|---|---|---|---|
| 01 | `01_simple_rag_api/` | FastAPI | Minimal RAG API: local embeddings + FAISS + mock generator | Offline (no keys) |
| 02 | `02_multiquery_rag_api/` | FastAPI | Multi-query expansion to improve recall | Offline (no keys) |
| 03 | `03_self_rag_api/` | FastAPI | Self-RAG loop: answer → evaluate → regenerate | Offline (no keys) |
| 04 | `04_tool_augmented_rag_api/` | FastAPI | Tool-augmented pattern scaffolding behind an API | Offline (no keys) |
| 05 | `05_corrective_rag_api/` | FastAPI | Corrective flow: retrieve → answer → evaluate → refine | Offline (no keys) |
| 06 | `06_hybrid_rag_api/` | FastAPI | Hybrid retrieval scaffolding (semantic + keyword merge) | Offline (no keys) |
| 07 | `07_graph_rag_api/` | FastAPI | Graph-style expansion (NetworkX) to broaden retrieval | Offline (no keys) |
| 08 | `08_agentic_rag_api/` | FastAPI | Agentic orchestration scaffolding exposed as an API | Offline (no keys) |
| 09 | `09_rag_chain_pdf_chat/` | Streamlit | PDF chat: upload → chunk → Chroma → retrieve → Gemini answer | `GOOGLE_API_KEY` |
| 10 | `10_hybrid_search_rag_raglite/` | Streamlit | Hybrid search + reranking UI (RAGLite) | OpenAI/Anthropic/Cohere keys (per UI) |
| 11 | `11_local_hybrid_search_rag_raglite/` | Streamlit | Local hybrid search with GGUF (llama-cpp) configured via UI | Local models |
| 12 | `12_local_rag_agent_agno_ollama/` | Python | Local RAG agent using Qdrant + Ollama | Qdrant + Ollama |
| 13 | `13_llama_webpage_rag/` | Streamlit | Webpage chat using Ollama + Chroma | Ollama |
| 14 | `14_agentic_rag_gpt5_agno/` | Streamlit | Agentic RAG: index URLs into LanceDB + answer with OpenAI | `OPENAI_API_KEY` |
| 15 | `15_agentic_rag_embedding_gemma/` | Streamlit | Local retrieval with Ollama EmbeddingGemma + LanceDB | Ollama |
| 16 | `16_agentic_rag_with_reasoning/` | Streamlit | Retrieve with OpenAI embeddings + answer with Gemini + show reasoning | `OPENAI_API_KEY`, `GOOGLE_API_KEY` |
| 17 | `17_agentic_rag_math_agent/` | Streamlit | Math agent dashboard: Q&A + feedback + benchmarking | (varies; see folder README) |
| 18 | `18_gemini_agentic_rag/` | Streamlit | Gemini embeddings/LLM + Qdrant retrieval + optional web tooling | Gemini + Qdrant |
| 19 | `19_deepseek_local_rag_agent/` | Streamlit | Local DeepSeek reasoning (Ollama) + Qdrant retrieval | Ollama + Qdrant |
| 20 | `20_qwen_local_rag_agent/` | Streamlit | Local Qwen/Gemma/DeepSeek selection + Qdrant retrieval | Ollama + Qdrant |
| 21 | `21_rag_as_a_service_ragie/` | Streamlit | Hosted RAG provider workflow + Anthropic answering | `RAGIE_API_KEY`, `ANTHROPIC_API_KEY` |
| 22 | `22_rag_agent_cohere_qdrant/` | Streamlit | Cohere embeddings/chat + Qdrant + optional web fallback | Cohere + Qdrant |
| 23 | `23_rag_database_routing/` | Streamlit | Route queries to multiple Qdrant collections (products/support/finance) | OpenAI + Qdrant |
| 24 | `24_knowledge_graph_rag_citations/` | Streamlit | Neo4j knowledge graph + verifiable citations + Ollama | Neo4j + Ollama |
| 25 | `25_contextualai_rag_agent/` | Streamlit | Manage ContextualAI agent/datastore workflow (upload → query) | ContextualAI |
| 26 | `26_ai_blog_search_langgraph/` | Streamlit | Index web pages into Qdrant + agentic LangGraph query flow | Gemini + Qdrant |
| 27 | `27_corrective_rag_langchain/` | Streamlit | Corrective retrieval with optional web search + multi-provider LLMs | OpenAI/Anthropic/Tavily + Qdrant |
| 28 | `28_autonomous_rag_agent/` | Streamlit | Autonomous agent: PDF knowledge base first, then web search | Postgres + OpenAI |
| 29 | `29_rag_failure_diagnostics_clinic/` | CLI | Diagnose common RAG failure modes (P01–P12) + JSON report | Offline |
| 30 | `30_vision_rag_multimodal/` | Streamlit | Multimodal RAG: embed images/PDF pages (Cohere) + answer with Gemini | Cohere + Gemini |

## Environment

See `.env.example` for common environment variables used across tutorials.

## Contributing

See `CONTRIBUTING.md`.

## License

MIT — see `LICENSE`.
