"""Resolve project-relative paths to absolute paths so they work regardless of cwd."""
from pathlib import Path
from backend.config.settings import get_settings

# backend/config/paths.py -> backend/config -> backend -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(path_str: str) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = (PROJECT_ROOT / path_str).resolve()
    return str(p)


def chroma_dir() -> str:
    return resolve(get_settings().chroma_persist_dir)


def bm25_path() -> str:
    return resolve(get_settings().bm25_index_path)
