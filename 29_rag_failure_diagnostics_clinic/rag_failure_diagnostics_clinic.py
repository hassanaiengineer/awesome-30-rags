"""
RAG Failure Diagnostics Clinic

Framework-agnostic RAG debugging clinic.
Diagnose LLM + RAG bugs into reusable failure patterns (P01–P12).

Built by Hassan Khan (Hassan Khan RAG Series).
"""

import json
import os
import textwrap
from getpass import getpass

from openai import OpenAI


PATTERNS = [
    {
        "id": "P01",
        "name": "Retrieval hallucination / grounding drift",
        "summary": "Answer confidently contradicts or ignores retrieved documents.",
    },
    {
        "id": "P02",
        "name": "Chunk boundary or segmentation bug",
        "summary": "Relevant facts are split, truncated, or mis-grouped across chunks.",
    },
    {
        "id": "P03",
        "name": "Embedding mismatch / semantic vs vector distance",
        "summary": "Vector similarity does not match true semantic relevance.",
    },
    {
        "id": "P04",
        "name": "Index skew or staleness",
        "summary": "Index returns old or missing data relative to the source of truth.",
    },
    {
        "id": "P05",
        "name": "Query rewriting or router misalignment",
        "summary": "Router or rewriter sends queries to the wrong tool or dataset.",
    },
    {
        "id": "P06",
        "name": "Long-chain reasoning drift",
        "summary": "Multi-step tasks gradually forget earlier constraints or goals.",
    },
    {
        "id": "P07",
        "name": "Tool-call misuse or ungrounded tools",
        "summary": "Tools are called with wrong arguments or without proper grounding.",
    },
    {
        "id": "P08",
        "name": "Session memory leak / missing context",
        "summary": "Conversation loses important facts between turns or sessions.",
    },
    {
        "id": "P09",
        "name": "Evaluation blind spots",
        "summary": "System passes tests but fails on real incidents or edge cases.",
    },
    {
        "id": "P10",
        "name": "Startup ordering / dependency not ready",
        "summary": "Services crash or return 5xx during the first minutes after deploy.",
    },
    {
        "id": "P11",
        "name": "Config or secrets drift across environments",
        "summary": "Works locally but breaks in staging or production because of settings.",
    },
    {
        "id": "P12",
        "name": "Multi-tenant or multi-agent interference",
        "summary": "Requests or agents overwrite each other’s state or resources.",
    },
]


EXAMPLE_1 = """=== Example 1 — retrieval hallucination (P01 style) ===

Context:
You have a simple RAG chatbot that answers questions from a product FAQ.
The FAQ only covers billing rules for your SaaS product and does NOT mention anything about cryptocurrency.

User prompt:
"Can I pay my subscription with Bitcoin?"

Retrieved context (from vector store):
- "We only accept major credit cards and PayPal."
- "All payments are processed in USD."

Model answer:
"Yes, you can pay with Bitcoin. We support several cryptocurrencies through a third-party payment gateway."

Logs:
No errors. Retrieval shows the FAQ chunks above, but the model still confidently invents Bitcoin support.
"""


EXAMPLE_2 = """=== Example 2 — startup ordering / dependency not ready (P10 style) ===

Context:
You have a RAG API with three services: api-gateway, rag-worker, and vector-db (for example Qdrant or FAISS).
In local docker compose everything works.

Deployment:
In production, services are deployed on Kubernetes.

Symptom:
Right after a fresh deploy, api-gateway returns 500 errors for the first few minutes.
Logs show connection timeouts from api-gateway to vector-db.

After a few minutes, the errors disappear and the system behaves normally.
You suspect a startup race between api-gateway and vector-db but are not sure how to fix it properly.
"""


EXAMPLE_3 = """=== Example 3 — config or secrets drift (P11 style) ===

Context:
You added a new environment variable for the RAG pipeline: SECRET_RAG_KEY.
This is required by middleware that signs outgoing requests to an internal search API.

Local:
On developer machines, SECRET_RAG_KEY is defined in .env and everything works.

Production:
You deployed a new version but forgot to add SECRET_RAG_KEY to the production environment.
The first requests after deploy fail with 500 errors and "missing secret" messages in the logs.

After hot-patching the secret into production, the errors stop.
However, similar "first deploy breaks because of missing config" incidents keep happening.
"""


def build_system_prompt() -> str:
    """Build the system prompt that explains the patterns and the task."""
    header = """
You are an assistant that triages failures in LLM + RAG pipelines.

You have a library of reusable failure patterns P01–P12.
For each bug description, you must:

1. Choose exactly ONE primary pattern id from P01–P12.
2. Optionally choose up to TWO secondary candidate pattern ids.
3. Explain your reasoning in clear bullet points.
4. Propose a MINIMAL structural fix:
