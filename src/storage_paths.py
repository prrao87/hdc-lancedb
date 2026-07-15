from __future__ import annotations

from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
RAW_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_URI = PROJECT_ROOT / "person-location"
DEFAULT_VOCAB_PATH = DEFAULT_DB_URI / "vocabulary.json"
