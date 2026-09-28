"""
12b_analyze_manip_checks.py — Compute manipulation-check accuracy from
12_run_experiment.py output.

For each (condition × item) pair the experimentally-correct answer is known
by design. This script reads the JSONL produced by 12_run_experiment.py,
filters to rows from manip_context and manip_framing, and reports:
  - overall accuracy per item
  - accuracy by condition
  - confusion matrix per item × condition

Expected correct codes
----------------------
manip_context (6a):
  G* (security context) → 1   "Aktif güvenlik gerilimi ..."
  B* (peace context)    → 2   "Anlaşma sağlanmış ..."

manip_framing (6b):
  *C (no framing)       → 1   "Ek bir gerekçe sunulmadan ..."
  *E (universal rights) → 2   "Eşit yurttaşlık, kültürel haklar ..."
  *K (self-determination) → 3 "Halkların kendi kaderini tayin ..."

Code 'Hatırlamıyorum' (3 in 6a, 4 in 6b) is always treated as INCORRECT.

Usage
-----
  python scripts/12b_analyze_manip_checks.py \
      --jsonl outputs/experiments/exp01_kurd_education/exp01_kurd_education_C11_gpt4omini_T08_ben_vs_cot_s0.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


# --- Ground-truth mapping --------------------------------------------------

# (item_id, factor_letter_relevant_for_item) → correct code
CORRECT: dict[tuple[str, str], int] = {
    ("manip_context", "G"): 1,
    ("manip_context", "B"): 2,
    ("manip_framing", "C"): 1,
    ("manip_framing", "E"): 2,
    ("manip_framing", "K"): 3,
}


def correct_for(item_id: str, condition: str) -> int | None:
    """Return the expected correct code for this (item, condition)."""
    if item_id == "manip_context":
        return CORRECT.get((item_id, condition[0]))   # G* or B*
    if item_id == "manip_framing":
        return CORRECT.get((item_id, condition[1]))   # *C, *E, *K
    return None


def load_records(jsonl: Path) -> list[dict]:
    rows = []
    for line in jsonl.open(encoding="utf-8"):
        try:
            r = json.loads(line)
            if r.get("parse_status") == "ok" and r.get("item_id") in {"manip_context", "manip_framing"}:
                rows.append(r)
        except Exception:
            continue
    return rows


def print_overall(rows: list[dict]) -> None:
    print("=" * 70)
    print("  Overall accuracy per manipulation-check item")
    print("=" * 70)
    by_item: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        exp = correct_for(r["item_id"], r["condition"])
        if exp is None:
            continue
        by_item[r["item_id"]].append(r["predicted_value"] == exp)
    for iid, hits in sorted(by_item.items()):
        n, k = len(hits), sum(hits)
        pct = k / n * 100 if n else 0.0
        print(f"  {iid:18s}  n={n:5d}   correct={k:5d}  →  {pct:5.1f}%")


def print_by_condition(rows: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  Accuracy by condition × item")
    print("=" * 70)
    by_cell: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in rows:
        exp = correct_for(r["item_id"], r["condition"])
        if exp is None:
            continue
        by_cell[(r["item_id"], r["condition"])].append(r["predicted_value"] == exp)

    print(f"  {'item':18s} {'cond':6s} {'n':>5s}  {'correct':>8s}  {'%':>7s}  expected")
    print("  " + "-" * 60)
    for (iid, cond), hits in sorted(by_cell.items()):
        n, k = len(hits), sum(hits)
        pct = k / n * 100 if n else 0.0
        exp = correct_for(iid, cond)
        print(f"  {iid:18s} {cond:6s} {n:5d}  {k:>8d}  {pct:6.1f}%  → {exp}")


def print_confusion(rows: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  Confusion: predicted code distribution per condition × item")
    print("=" * 70)
    by_cell: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for r in rows:
        if r.get("predicted_value") is None:
            continue
        by_cell[(r["item_id"], r["condition"])][r["predicted_value"]] += 1

    for (iid, cond), counts in sorted(by_cell.items()):
        total = sum(counts.values())
        exp = correct_for(iid, cond)
        codes = sorted(counts)
        line = ", ".join(
            f"{c}={counts[c]} ({counts[c]/total*100:.0f}%)" +
            ("*" if c == exp else "")
            for c in codes
        )
        print(f"  {iid:18s} {cond:6s}  expected={exp}  →  {line}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, type=Path,
                   help="Path to the experiment JSONL output.")
    args = p.parse_args()

    if not args.jsonl.exists():
        raise SystemExit(f"File not found: {args.jsonl}")

    rows = load_records(args.jsonl)
    if not rows:
        raise SystemExit(f"No manipulation-check rows found in {args.jsonl}")

    print(f"Loaded {len(rows)} manipulation-check records from {args.jsonl}\n")
    print_overall(rows)
    print_by_condition(rows)
    print_confusion(rows)


if __name__ == "__main__":
    main()
