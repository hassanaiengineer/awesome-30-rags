import json
import numpy as np
from typing import List

import faiss

from app.model import EmbeddingModel
from app.paths import INDEX_PATH, TEXTS_PATH

def load_index():
    if not INDEX_PATH.exists() or not TEXTS_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    with TEXTS_PATH.open("r", encoding="utf-8") as f:
        texts = json.load(f)
    return index, texts

def generate_variations(query: str) -> List[str]:
    return [query, query + " explained", "details about " + query]

def retrieve(query: str, top_k: int = 2):
    index, texts = load_index()
    if index is None or not texts:
        return []
    model = EmbeddingModel()
    variations = generate_variations(query)
    all_results = []
    for var in variations:
        query_emb = model.encode([var]).astype('float32')
        distances, indices = index.search(query_emb, top_k)
        all_results.extend([texts[i] for i in indices[0] if i != -1])
    
    return list(set(all_results))[:top_k]
