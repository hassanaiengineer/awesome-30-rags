import numpy as np
from app.model import EmbeddingModel
from app.retrieve import retrieve
from app.generate import generate_answer

def evaluate_answer(query: str, answer: str, context: list):
    model = EmbeddingModel()
    ans_emb = model.encode([answer])
    ctx_emb = model.encode([" ".join(context)])
    sim = np.dot(ans_emb[0], ctx_emb[0]) / (np.linalg.norm(ans_emb[0]) * np.linalg.norm(ctx_emb[0]))
    return float(sim)

def corrective_rag_flow(query: str, answer: str, context: list):
    score = evaluate_answer(query, answer, context)
    if score < 0.3:
        new_context = retrieve(query + " specific details", top_k=3)
        new_answer, _ = generate_answer(query, new_context)
        return new_answer, 0.7
    return answer, float(score)
