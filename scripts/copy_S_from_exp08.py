"""
copy_S_from_exp08.py — Seed a new context experiment with exp08's Security records.

Replaces 17b_copy_exp08_S_to_exp10.py and 18b_copy_exp08_S_to_exp11.py.

Because the Security vignette is identical across all Peace-version
experiments in the exp08/exp10/exp11 family, we can re-use exp08's S
records instead of re-running the API. The `experiment_id` field is
rewritten to the target experiment id so the copied records look native.

Only records with `condition == "S"` and `parse_status == "ok"` are copied.
Existing (rid, condition, item_id) triples in the destination are skipped.

Usage
-----
    python scripts/copy_S_from_exp08.py --to exp10_context_peacev3
    python scripts/copy_S_from_exp08.py --to exp11_context_peacev4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "outputs" / "experiments" / "exp08_context_only"
SRC_PREFIX = "exp08_context_only_"


def copy_S(target_exp_id: str) -> None:
    dst_dir = ROOT / "outputs" / "experiments" / target_exp_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for src_fp in sorted(SRC_DIR.glob("*.jsonl")):
        dst_name = src_fp.name.replace(SRC_PREFIX, f"{target_exp_id}_", 1)
        dst_fp = dst_dir / dst_name

        existing: set[tuple[str, str, str]] = set()
        if dst_fp.exists():
            for line in dst_fp.open(encoding="utf-8"):
                try:
                    r = json.loads(line)
                    existing.add((r["respondent_id"], r["condition"], r["item_id"]))
                except Exception:
                    continue

        kept = 0
        with dst_fp.open("a", encoding="utf-8") as out:
            for line in src_fp.open(encoding="utf-8"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("condition") != "S" or r.get("parse_status") != "ok":
                    continue
                key = (r["respondent_id"], r["condition"], r["item_id"])
                if key in existing:
                    continue
                r["experiment_id"] = target_exp_id
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                existing.add(key)
                kept += 1

        print(f"  {src_fp.name}  →  {dst_fp.name}    S copied: {kept}")
        total += kept

    print(f"\n✓ {total} Security records seeded into "
          f"{dst_dir.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--to", required=True,
                   help="Target experiment id, e.g. exp10_context_peacev3")
    args = p.parse_args()
    copy_S(args.to)


if __name__ == "__main__":
    main()
