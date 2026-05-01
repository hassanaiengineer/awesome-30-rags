import numpy as np
from app.model import EmbeddingModel

def evaluate_answer(query: str, answer: str, context: list):
    score = min(1.0, len(answer) / 100.0)
    return float(score)
