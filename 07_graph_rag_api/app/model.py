import os
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("Loading Embedding Model...")
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance.model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._instance

    def encode(self, texts):
        return self.model.encode(texts)
