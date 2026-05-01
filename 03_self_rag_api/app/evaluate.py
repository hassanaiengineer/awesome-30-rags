import numpy as np
from app.model import EmbeddingModel

def evaluate_answer(query: str, answer: str, context: list):
    score = min(1.0, len(answer) / 100.0)
    return float(score)

from app.retrieve import retrieve
from app.generate import generate_answer

def self_rag_evaluate_and_regenerate(query: str, answer: str, context: list):
    score = evaluate_answer(query, answer, context)
    if score < 0.5:
        new_context = retrieve(query, top_k=4)
        new_answer, _ = generate_answer(query, new_context)
        return new_answer, 0.8
    return answer, float(score)
