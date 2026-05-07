import json

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

def retrieve(query: str, top_k: int = 2):
    index, texts = load_index()
    if index is None or not texts:
        return []
    model = EmbeddingModel()
    query_emb = model.encode([query]).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    return [texts[i] for i in indices[0] if i != -1]
