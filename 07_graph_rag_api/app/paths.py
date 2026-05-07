from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
TEXTS_PATH = DATA_DIR / "texts.json"
GRAPH_PATH = DATA_DIR / "graph.pkl"

