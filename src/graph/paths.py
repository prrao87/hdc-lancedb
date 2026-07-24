from __future__ import annotations

from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
RAW_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_URI = PROJECT_ROOT / "person-location"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
SEMANTIC_HDC_METADATA_FILENAME = "semantic_hdc.json"
SEMANTIC_PROJECTION_FILENAME = "semantic_projection.npy"
