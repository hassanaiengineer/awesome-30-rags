import json

import faiss
import numpy as np
from app.model import EmbeddingModel
from app.paths import DATA_DIR, INDEX_PATH, TEXTS_PATH


def ingest_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial Intelligence is transforming the world.",
        "Python is a versatile programming language.",
        "FAISS is a library for efficient similarity search.",
        "Retrieval-Augmented Generation improves LLM accuracy.",
    ]
    
    model = EmbeddingModel()
    embeddings = model.encode(texts)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    faiss.write_index(index, str(INDEX_PATH))
    with TEXTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
        

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()
