import json
import numpy as np
from app.model import EmbeddingModel

import faiss

from app.paths import INDEX_PATH, TEXTS_PATH

def load_index():
    if not INDEX_PATH.exists() or not TEXTS_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    with TEXTS_PATH.open("r", encoding="utf-8") as f:
        texts = json.load(f)
    return index, texts

def keyword_search(query: str, texts: list):
    query_words = set(query.lower().split())
    scores = []
    for text in texts:
        score = sum(1 for w in query_words if w in text.lower())
        scores.append(score)
    return scores

def retrieve(query: str, top_k: int = 2):
    index, texts = load_index()
    if index is None or not texts:
        return []
    model = EmbeddingModel()
    query_emb = model.encode([query]).astype('float32')
    distances, indices = index.search(query_emb, top_k * 2)
    
    sem_results = [texts[i] for i in indices[0] if i != -1]
    
    kw_scores = keyword_search(query, texts)
    kw_indices = np.argsort(kw_scores)[::-1][:top_k*2]
    kw_results = [texts[i] for i in kw_indices]
    
    merged = list(set(sem_results + kw_results))
    return merged[:top_k]
