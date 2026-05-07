import pickle
import networkx as nx
import json
import numpy as np
from app.model import EmbeddingModel

import faiss

from app.paths import GRAPH_PATH, INDEX_PATH, TEXTS_PATH

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
    base_results = [indices[0][i] for i in range(len(indices[0])) if indices[0][i] != -1]
    
    expanded_results = set(base_results)
    if GRAPH_PATH.exists():
        with GRAPH_PATH.open("rb") as f:
            G = pickle.load(f)
        for node in base_results:
            if node in G:
                for neighbor in G.neighbors(node):
                    if isinstance(neighbor, int):
                        expanded_results.add(neighbor)
    
    return [texts[i] for i in list(expanded_results)[:top_k*2]]
