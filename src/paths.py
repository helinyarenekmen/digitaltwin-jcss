"""
Configurable cache locations for the replication pipeline.

Scripts import CACHE_DIR and PERSONA_DIR from here rather than hard-coding
absolute paths. Both can be overridden by environment variables:

    DT_CACHE_DIR    — raw JSONL completions from every inference run
                      (default: <repo>/cache/calibration/)
    DT_PERSONA_DIR  — rendered persona markdown files, one per config
                      (default: <repo>/data/derived/personas/)

The defaults live inside the repository, are gitignored, and are created on
first use. Original authors ran the pipeline with DT_CACHE_DIR and
DT_PERSONA_DIR pointing at ~/Library/Caches/digitaltwin_* so the caches would
sit outside iCloud sync.
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR   = Path(os.environ.get("DT_CACHE_DIR",   ROOT / "cache" / "calibration"))
PERSONA_DIR = Path(os.environ.get("DT_PERSONA_DIR", ROOT / "data" / "derived" / "personas"))

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PERSONA_DIR.mkdir(parents=True, exist_ok=True)
