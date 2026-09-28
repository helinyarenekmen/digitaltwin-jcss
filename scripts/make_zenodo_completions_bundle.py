"""
Convert every cached raw-completion JSONL into a compressed Parquet file and
copy them into a single upload-ready bundle for Zenodo.

Reads from:
    ~/Library/Caches/digitaltwin_calibration/screening*/
    ~/Library/Caches/digitaltwin_calibration/screening_vs_cot*/

Writes to (mirroring the source layout):
    outputs/completions/screening*/…/<cell>.parquet
    outputs/completions/screening_vs_cot*/…/<cell>.parquet
    outputs/completions/experiments/exp13/…

Also writes an SHA-256 manifest so integrity is checkable after download.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

import pandas as pd

ROOT   = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "outputs" / "completions"
CACHES = [
    Path.home() / "Library/Caches/digitaltwin_calibration",
]
EXP13_CSVS = ROOT.parent / "outputs" / "experiments" / "exp13_noPC_gpt5.4mini_gpt4omini" / "csv"


def convert_jsonl(src: Path, dst: Path) -> tuple[int, int]:
    """Return (n_rows, output_bytes)."""
    rows = [json.loads(l) for l in src.open()]
    if not rows:
        return 0, 0
    df = pd.DataFrame(rows)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, compression="zstd", index=False)
    return len(df), dst.stat().st_size


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    total_rows = 0
    total_bytes = 0

    for cache in CACHES:
        for src in cache.rglob("*.jsonl"):
            rel = src.relative_to(cache)          # e.g. screening_T08/pacdemons/C7_gpt4omini_...jsonl
            dst = BUNDLE / rel.with_suffix(".parquet")
            n, sz = convert_jsonl(src, dst)
            if n > 0:
                total_rows += n; total_bytes += sz
                manifest.append({
                    "path": str(dst.relative_to(BUNDLE)),
                    "source": str(src),
                    "rows":   n,
                    "bytes":  sz,
                    "sha256": sha256(dst),
                })
                print(f"  ✓ {rel}  ({n} rows, {sz/1024:.0f} KB)")

    # Manifest
    manifest_path = BUNDLE / "MANIFEST.json"
    manifest_path.write_text(json.dumps({
        "n_files": len(manifest),
        "n_rows_total": total_rows,
        "bytes_total": total_bytes,
        "files": manifest,
    }, indent=2))
    print(f"\n✓ {manifest_path}   ({len(manifest)} files, {total_rows:,} rows, {total_bytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
