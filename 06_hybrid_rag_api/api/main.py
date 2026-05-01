from fastapi import FastAPI
from pydantic import BaseModel

from app.branding import (
    OWNER_EMAIL,
    OWNER_LINKEDIN,
    OWNER_NAME,
    OWNER_TAGLINE,
    fastapi_description_md,
)
from app.evaluate import evaluate_answer
from app.generate import generate_answer


app = FastAPI(
    title="06 - Hybrid RAG API (30 Days of RAG)",
    description=fastapi_description_md(),
    version="0.1.0",
)


class Query(BaseModel):
    query: str


@app.get("/meta")
def meta():
    return {
        "series": "30 Days of RAG",
        "tutorial": "06_hybrid_rag_api",
        "owner": OWNER_NAME,
        "tagline": OWNER_TAGLINE,
        "linkedin": OWNER_LINKEDIN,
        "email": OWNER_EMAIL,
    }


@app.post("/ask")
def ask(q: Query):
    answer, context = generate_answer(q.query)
    score = evaluate_answer(q.query, answer, context)
    return {"answer": answer, "score": score}
