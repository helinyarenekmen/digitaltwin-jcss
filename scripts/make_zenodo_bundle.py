"""
Package the raw model completions and persona corpora into `zenodo_bundle/`
for upload to Zenodo.

* Every JSONL cell in `DT_CACHE_DIR` (default: ~/Library/Caches/digitaltwin_calibration
  and the local ./cache/calibration/) is converted to zstd-compressed Parquet.
* Filenames use PAPER config labels (C0-C10, C6-holdout-*). The internal
  legacy id is retained as a `legacy_config_id` column so archived filenames
  stay traceable to the raw JSONL.
* Persona markdown files under `DT_PERSONA_DIR` are archived by config as
  compressed tarballs.
* A MANIFEST.json is written alongside with SHA-256 checksums for every file.

Usage
-----
    python scripts/make_zenodo_bundle.py
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.ablations import CONFIGS_BY_LEGACY_ID
from src.paths import CACHE_DIR as REPO_CACHE_DIR, PERSONA_DIR as REPO_PERSONA_DIR

# Also look at the original ~/Library/Caches locations
EXTRA_CACHES = [
    Path(os.path.expanduser("~/Library/Caches/digitaltwin_calibration")),
]
EXTRA_PERSONAS = [
    Path(os.path.expanduser("~/Library/Caches/digitaltwin_layer_a")),
]

BUNDLE = ROOT / "zenodo_bundle"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _legacy_to_paper(legacy_id: str) -> str:
    return CONFIGS_BY_LEGACY_ID[legacy_id].config_id if legacy_id in CONFIGS_BY_LEGACY_ID else legacy_id


def convert_completions(manifest: list[dict]) -> None:
    """Convert every JSONL cache cell to Parquet under bundle/completions/."""
    total_rows = 0
    total_bytes = 0
    seen = set()
    caches = [REPO_CACHE_DIR] + [c for c in EXTRA_CACHES if c.exists()]
    for cache in caches:
        if not cache.exists():
            continue
        for src in cache.rglob("*.jsonl"):
            key = src.name
            if key in seen: continue
            seen.add(key)
            rows = [json.loads(l) for l in src.open()]
            if not rows: continue
            df = pd.DataFrame(rows)
            # Rewrite config_id column to paper label; keep legacy in a separate column
            if "config_id" in df.columns:
                df["legacy_config_id"] = df["config_id"]
                df["config_id"] = df["config_id"].map(_legacy_to_paper)
            # Compute paper-labeled filename
            legacy_head = src.name.split("_")[0]  # e.g. "C7"
            paper_head = _legacy_to_paper(legacy_head)
            new_name = src.name.replace(legacy_head, paper_head, 1).replace(".jsonl", ".parquet")
            rel = src.relative_to(cache)
            dst = BUNDLE / "completions" / rel.parent / new_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, compression="zstd", index=False)
            manifest.append({
                "kind":         "completions",
                "path":         str(dst.relative_to(BUNDLE)),
                "source":       str(src),
                "rows":         len(df),
                "bytes":        dst.stat().st_size,
                "sha256":       _sha256(dst),
            })
            total_rows += len(df); total_bytes += dst.stat().st_size
    print(f"  completions: {len(seen)} files, {total_rows:,} rows, {total_bytes/1e6:.1f} MB")


def archive_personas(manifest: list[dict]) -> None:
    """Archive each config's persona corpus as a tar.gz under bundle/personas/."""
    persona_dst = BUNDLE / "personas"; persona_dst.mkdir(parents=True, exist_ok=True)
    dirs = []
    for p_root in [REPO_PERSONA_DIR] + EXTRA_PERSONAS:
        if p_root.exists():
            dirs.extend([d for d in p_root.iterdir() if d.is_dir()])
    seen_paper = set()
    for d in dirs:
        legacy_id = d.name
        paper_id = _legacy_to_paper(legacy_id)
        if paper_id in seen_paper: continue
        seen_paper.add(paper_id)
        tar_path = persona_dst / f"{paper_id}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(d, arcname=paper_id)
        manifest.append({
            "kind":         "personas",
            "path":         str(tar_path.relative_to(BUNDLE)),
            "source":       str(d),
            "paper_id":     paper_id,
            "legacy_id":    legacy_id,
            "bytes":        tar_path.stat().st_size,
            "sha256":       _sha256(tar_path),
        })
    print(f"  personas: {len(seen_paper)} tarballs")


def main() -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    print("Converting completions to Parquet...")
    convert_completions(manifest)
    print("Archiving persona corpora as tar.gz...")
    archive_personas(manifest)
    (BUNDLE / "MANIFEST.json").write_text(json.dumps({
        "n_files":    len(manifest),
        "bytes_total": sum(m["bytes"] for m in manifest),
        "files":      manifest,
    }, indent=2))
    total = sum(m["bytes"] for m in manifest) / 1e6
    print(f"\n✓ zenodo_bundle/  ({len(manifest)} files, {total:.1f} MB)")


if __name__ == "__main__":
    main()
