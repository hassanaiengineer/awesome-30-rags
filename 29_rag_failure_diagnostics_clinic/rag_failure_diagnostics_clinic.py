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
