from __future__ import annotations

import sys
from pathlib import Path


def add_repo_root_to_sys_path(start: Path) -> Path:
    """
    Walk upward until we find the repo root (folder containing `rag_series/`).

    Returns the detected repo root path and ensures it's on `sys.path`.
    """
    current = start.resolve()
    if current.is_file():
        current = current.parent

    while True:
        if (current / "rag_series").is_dir():
            repo_root = current
            break
        if current.parent == current:
            raise RuntimeError("Could not locate repo root containing `rag_series/`.")
        current = current.parent

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root

