"""
compute_gwet_ac2.py — Compute Gwet's AC2 with quadratic weights for every
calibration cell we have on disk, then produce a phase-by-phase ranking.

Gwet's AC2 (Gwet 2008, 2014) is an inter-rater agreement coefficient that
addresses the paradoxical low values Cohen's kappa can produce when
marginal distributions are skewed (which is exactly the case for
ordinal Likert outcomes such as womenwork). AC2 uses the same weighted-
agreement structure as Cohen's weighted kappa but replaces the chance-
agreement term with Gwet's γ-model, which is more stable under skewed
marginals.

Formula (weighted, 2 raters, k ordinal categories) — disagreement form
----------------------------------------------------------------------
Let n_{ij} be the number of subjects that rater 1 places in category i
and rater 2 places in category j. Let N = Σ n_{ij} and p_{ij} = n_{ij}/N.

Define quadratic AGREEMENT weights on the actual category values
(c_1 < c_2 < … < c_k):

    w_{ij} = 1 − ((c_i − c_j)² / (c_max − c_min)²).

The equivalent DISAGREEMENT weights are

    d_{ij} = 1 − w_{ij}.

Note d_{ii} = 0 and d_{ij} = w-complement off the diagonal (0 on the
diagonal, 1 at the extremes).

Observed weighted disagreement
    Do = Σ_{i,j} d_{ij} · p_{ij}

Marginal probabilities
    π₁(i) = Σ_j p_{ij}         π₂(i) = Σ_j p_{ji}
    π₊(i) = (π₁(i) + π₂(i)) / 2

Chance-expected disagreement under Gwet's γ-model
    De = [Σ_{i,j} d_{ij} / (k (k − 1))] · Σ_i π₊(i) · (1 − π₊(i)).
Since d_{ii} = 0 the double sum in the bracket equals the sum over
i ≠ j; both formulations are equivalent.

Coefficient
    AC2 = 1 − (Do / De).

This disagreement-form derivation is algebraically identical to
(Pa − Pe) / (1 − Pe) when Pa and Pe are defined with the AGREEMENT
weights w_{ij}. The agreement-weight version, however, degenerates in
the binary case: for k = 2 quadratic w has zeros on the off-diagonal
so Tw computed from off-diagonal agreement weights becomes zero and
Pe collapses. The implementation below therefore always uses the
disagreement matrix d_{ij} and never uses w_{ij} directly in the
chance term.

Interpretation is on the same scale as κ: 1 = perfect agreement,
0 = chance, negatives = worse-than-chance.

Inputs
------
Reads every cell JSONL under `~/Library/Caches/digitaltwin_calibration/`
whose outcome is one of {pacdemons, womenwork, neilang} and pairs the
`predicted_value` with the corresponding TGSS ground truth from
`agent-survey/agent-calibration/<outcome>/…_data_output.csv`.

Outputs
-------
outputs/gwet_ac2/
    all_cells.csv            master table (outcome, phase, cell, n, AC2)
    ranking_womenwork.csv    ranking within each Phase for womenwork
    ranking_pacdemons.csv    ranking within each Phase for pacdemons
    ranking_neilang.csv      ranking within each Phase for neilang
    (each ranking table also written as a Markdown for quick inspection)

Also augments
    outputs/per_outcome_scores/<outcome>_all_phases.xlsx
by adding an `ac2_quadratic` column (only for cells whose Source_CSV
row can be matched) — done idempotently.

Usage
-----
    python scripts/compute_gwet_ac2.py
    python scripts/compute_gwet_ac2.py --outcome womenwork --no-excel
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
CALIB = Path.home() / "Library" / "Caches" / "digitaltwin_calibration"
GT_PATHS = {
    "pacdemons": ROOT / "agent-survey/agent-calibration/pacdemons/pacdemons_predictions_20260418_123339.csv",
    "womenwork": ROOT / "agent-survey/agent-calibration/womenwork/womenwork-data/womenwork_data_output.csv",
    "neilang":   ROOT / "agent-survey/agent-calibration/neilang/neilang-data/neilang_verbsampling_20260418_141028.csv",
}
OUT_DIR = ROOT / "outputs" / "gwet_ac2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_RANGES = {"pacdemons": (1, 2), "womenwork": (1, 5), "neilang": (1, 3)}


# ---------------------------------------------------------------------------
# Gwet's AC2 (quadratic weights) — self-contained numpy implementation
# ---------------------------------------------------------------------------
def gwet_ac2_quadratic(rater1: Iterable[int],
                       rater2: Iterable[int],
                       categories: list[int] | None = None) -> float:
    """Gwet's AC2 with quadratic weights, disagreement formulation.

    The agreement weight matrix w_ij is defined internally but is only
    used to derive the disagreement matrix d_ij = 1 − w_ij. All
    subsequent quantities (Do, De, AC2) are computed from d_ij; the
    agreement matrix w_ij is never fed into the chance-agreement term.
    This is what makes the implementation stable in the binary case
    (where w_ij has zeros off the diagonal, which would collapse a
    Tw-based agreement formulation).
    """
    r1 = np.asarray(list(rater1), dtype=int)
    r2 = np.asarray(list(rater2), dtype=int)

    # --- Input validation -------------------------------------------------
    if len(r1) != len(r2):
        return float("nan")
    if len(r1) == 0:
        return float("nan")

    if categories is None:
        categories = sorted(set(r1.tolist()) | set(r2.tolist()))
    k = len(categories)
    if k < 2:
        return float("nan")

    # --- Contingency matrix ----------------------------------------------
    cat_idx = {c: i for i, c in enumerate(categories)}
    n = np.zeros((k, k), dtype=int)
    for a, b in zip(r1, r2):
        if a in cat_idx and b in cat_idx:
            n[cat_idx[a], cat_idx[b]] += 1
    N = int(n.sum())
    if N == 0:
        return float("nan")
    p = n / N

    # --- Disagreement weights (quadratic) --------------------------------
    # Agreement weight: w_ij = 1 − ((c_i − c_j)² / (c_max − c_min)²).
    # Disagreement weight: d_ij = 1 − w_ij = (c_i − c_j)² / (c_max − c_min)².
    # d has zeros on the diagonal by construction.
    cat_vals = np.asarray(categories, dtype=float)
    span_sq = (cat_vals.max() - cat_vals.min()) ** 2
    if span_sq == 0:
        return float("nan")
    d = ((cat_vals[:, None] - cat_vals[None, :]) ** 2) / span_sq

    # --- Observed weighted disagreement ----------------------------------
    Do = float(np.sum(d * p))

    # --- Marginals (Gwet averages the two raters' marginal distributions) --
    pi1 = p.sum(axis=1)   # rater 1 marginal
    pi2 = p.sum(axis=0)   # rater 2 marginal
    pi_plus = (pi1 + pi2) / 2.0

    # --- Chance-expected disagreement under Gwet's γ-model ---------------
    # De = [Σ d_ij / (k(k − 1))] · Σ_i π₊(i) (1 − π₊(i)).
    # Because d_ii = 0, summing over all cells is equivalent to summing
    # over i ≠ j only.
    d_sum = float(np.sum(d))
    denom_pairs = k * (k - 1)
    if denom_pairs == 0:
        return float("nan")
    De = (d_sum / denom_pairs) * float(np.sum(pi_plus * (1.0 - pi_plus)))

    # --- Edge cases in the ratio -----------------------------------------
    # Perfect agreement gives Do = 0. If De is also 0 (e.g. every rating
    # collapses to a single category), the standard convention is AC2 = 1.
    if Do == 0.0:
        return 1.0
    if De == 0.0:
        # Do > 0 and De = 0 → coefficient is undefined.
        return float("nan")

    return 1.0 - (Do / De)


# ---------------------------------------------------------------------------
# Sanity checks — cheap unit tests of the AC2 implementation
# ---------------------------------------------------------------------------
def _sanity() -> None:
    # (1) Perfect agreement on a 5-point ordinal scale → AC2 = 1.
    x5 = [1, 2, 3, 4, 5] * 20
    ac2_perfect_5 = gwet_ac2_quadratic(x5, x5, [1, 2, 3, 4, 5])
    assert abs(ac2_perfect_5 - 1.0) < 1e-12, \
        f"perfect 5-cat agreement should be 1.0, got {ac2_perfect_5}"

    # (2) Perfect agreement on a binary scale → AC2 = 1.
    xb = [1, 2, 1, 2, 1] * 20
    ac2_perfect_bin = gwet_ac2_quadratic(xb, xb, [1, 2])
    assert abs(ac2_perfect_bin - 1.0) < 1e-12, \
        f"perfect binary agreement should be 1.0, got {ac2_perfect_bin}"

    # (3) Balanced binary "coin-flip" agreement pattern:
    # rater1 = [1,1,2,2], rater2 = [1,2,1,2]. Off-diagonal entries are
    # exactly half of the contingency table. AC2 should be ≈ 0.
    r1 = [1, 1, 2, 2]
    r2 = [1, 2, 1, 2]
    ac2_zero = gwet_ac2_quadratic(r1, r2, [1, 2])
    assert abs(ac2_zero) < 1e-9, \
        f"balanced binary disagreement should be ≈0, got {ac2_zero}"

    # (4) Complete binary disagreement → AC2 should be negative.
    r1 = [1, 1, 2, 2]
    r2 = [2, 2, 1, 1]
    ac2_neg = gwet_ac2_quadratic(r1, r2, [1, 2])
    assert ac2_neg < 0.0, \
        f"complete binary disagreement should be negative, got {ac2_neg}"

    # (5) Off-by-one on a 5-point scale → AC2 should still be positive.
    # Quadratic weights penalise a 1-step disagreement only lightly
    # (d = 1/16 = 0.0625), so the coefficient should stay well above 0.
    y1 = [1, 2, 3, 4] * 100
    y2 = [2, 3, 4, 5] * 100  # every response shifted by +1
    ac2_off1 = gwet_ac2_quadratic(y1, y2, [1, 2, 3, 4, 5])
    assert ac2_off1 > 0.0, \
        f"off-by-one on 5-point should be positive, got {ac2_off1}"

    # Extra guardrail: reject mismatched lengths and empty input.
    assert np.isnan(gwet_ac2_quadratic([1, 2], [1], [1, 2]))
    assert np.isnan(gwet_ac2_quadratic([], [], [1, 2]))
    assert np.isnan(gwet_ac2_quadratic([1, 1], [1, 1], [1]))


# ---------------------------------------------------------------------------
# Cell-file discovery + ground-truth join
# ---------------------------------------------------------------------------
def load_gt(outcome: str) -> dict[str, int]:
    df = pd.read_csv(GT_PATHS[outcome])
    df["respondent_id"] = df["persona_id"].astype(str).str.zfill(4).apply(lambda x: f"TGSS_{x}")
    return dict(zip(df["respondent_id"], df[f"gt_{outcome}"].astype(int)))


def outcome_of(path: Path) -> str | None:
    for o in VALID_RANGES:
        if o in path.parts:
            return o
    return None


def discover_cells() -> list[tuple[str, Path]]:
    """Return (outcome, jsonl_path) for every calibration cell we can find."""
    cells: list[tuple[str, Path]] = []
    for outcome in VALID_RANGES:
        for jsonl in CALIB.rglob(f"*/{outcome}/*.jsonl"):
            cells.append((outcome, jsonl))
    return cells


# ---------------------------------------------------------------------------
# Phase inference — read the master catalog once, map cell filenames → phase
# ---------------------------------------------------------------------------
_PHASE_HINTS = [
    # (regex on filename or dir name → phase label)
    (r"screening_vs_cot_T04_sen", "Phase 6+13 (sen-dili)"),
    (r"screening_vs_cot_T04",     "Phase 4 (VS-CoT T=0.4)"),
    (r"screening_vs_cot_T08",     "Phase 3 (VS-CoT T=0.8)"),
    (r"screening_vs_cot(?!/)",    "Phase 2 (VS-CoT T=0)"),
    (r"screening_T04",            "Phase 11 (Direct T=0.4)"),
    (r"screening_T08",            "Phase 11 (Direct T=0.8)"),
    (r"screening(?!_)",           "Phase 1 (Step 2A screening)"),
]

_VENDOR_SUFFIXES = {
    "gpt54mini":         "Phase 7 (GPT-5.4-mini xtier)",
    "geminiflashlite":   "Phase 9 (Gemini xvendor)",
    "gemini2.5flashlite":"Phase 9 (Gemini xvendor)",
    "llama3370b":        "Phase 10 (Llama xvendor)",
    "claudehaiku":       "Phase 8 (Claude Haiku xvendor)",
}


def infer_phase(jsonl: Path) -> str:
    name = jsonl.name
    parent = jsonl.parent.parent.name  # e.g. screening_vs_cot_T08
    # Vendor-specific phases override the plain calibration phase
    for tag, phase in _VENDOR_SUFFIXES.items():
        if f"_{tag}_" in name:
            return phase
    if "_s1" in name or "_s2" in name:
        return "Phase 12 (multi-seed)"
    if "_pc_" in name:
        return "Phase 15 (political context)"
    if "_cot_" in name and "vs_cot" not in name:
        return "Phase 14 (explicit CoT)"
    for pat, phase in _PHASE_HINTS:
        if re.search(pat, str(jsonl)):
            return phase
    return "Unknown"


# ---------------------------------------------------------------------------
# Cell metadata parser (parallels scripts/04's naming convention)
# ---------------------------------------------------------------------------
_MODEL_ALIAS_INV = {
    "gpt4omini":         "gpt-4o-mini",
    "gpt54mini":         "gpt-5.4-mini",
    "gpt5mini":          "gpt-5-mini",
    "geminiflashlite":   "gemini-2.5-flash-lite",
    "gemini2.5flashlite":"gemini-2.5-flash-lite",
    "llama3370b":        "meta-llama/llama-3.3-70b-instruct",
    "claudehaiku":       "claude-haiku-4-5-20251001",
}
_T_MAP = {"T0": 0.0, "T03": 0.3, "T04": 0.4, "T07": 0.7, "T08": 0.8, "T1": 1.0}


def parse_cell_name(name: str) -> dict:
    """Parse filenames like C11_gpt4omini_T08_ben_vs_cot_nocot_s0.jsonl."""
    m = re.match(
        r"(?P<config>C\d+)_(?P<model>[^_]+)_(?P<T>T\d+)_(?P<addr>ben|sen)_"
        r"(?P<sampling>vs_cot|direct)_(?P<cot>cot|nocot)"
        r"(?:_(?P<pc>pc))?_s(?P<seed>\d+)\.jsonl", name)
    if not m:
        return {}
    return {
        "config_id":   m["config"],
        "model_alias": m["model"],
        "model":       _MODEL_ALIAS_INV.get(m["model"], m["model"]),
        "temperature": _T_MAP.get(m["T"], float(m["T"][1:]) / 10),
        "address":     m["addr"],
        "sampling":    m["sampling"],
        "cot":         m["cot"] == "cot",
        "political_context": m["pc"] == "pc",
        "seed":        int(m["seed"]),
    }


# ---------------------------------------------------------------------------
# AC2 computation for one cell
# ---------------------------------------------------------------------------
def compute_cell_ac2(jsonl: Path, outcome: str, gt: dict[str, int]) -> dict:
    lo, hi = VALID_RANGES[outcome]
    categories = list(range(lo, hi + 1))

    preds: dict[str, int] = {}
    n_total, n_ok = 0, 0
    for line in jsonl.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        n_total += 1
        if d.get("parse_status") != "ok" or d.get("predicted_value") is None:
            continue
        rid = d["respondent_id"]
        try:
            v = int(d["predicted_value"])
        except (TypeError, ValueError):
            continue
        if lo <= v <= hi:
            preds[rid] = v
            n_ok += 1

    pairs = [(gt[r], v) for r, v in preds.items() if r in gt]
    n_matched = len(pairs)
    if n_matched == 0:
        ac2 = float("nan")
    else:
        r_gt = [p[0] for p in pairs]
        r_pd = [p[1] for p in pairs]
        ac2 = gwet_ac2_quadratic(r_gt, r_pd, categories)

    return {
        "outcome":   outcome,
        "cell":      jsonl.name,
        "phase":     infer_phase(jsonl),
        "n_total":   n_total,
        "n_ok":      n_ok,
        "n_matched": n_matched,
        "ac2_quadratic": ac2,
        "path":      str(jsonl.relative_to(CALIB.parent) if str(jsonl).startswith(str(CALIB.parent)) else jsonl),
        **parse_cell_name(jsonl.name),
    }


# ---------------------------------------------------------------------------
# Excel augmentation — add AC2 column to per_outcome_scores/*.xlsx
# ---------------------------------------------------------------------------
def augment_excel(outcome: str, ac2_by_cell: dict[str, float]) -> None:
    xlsx = ROOT / "outputs" / "per_outcome_scores" / f"{outcome}_all_phases.xlsx"
    if not xlsx.exists():
        return

    wb = load_workbook(xlsx)
    if "All cells" not in wb.sheetnames:
        return
    ws = wb["All cells"]

    headers = [c.value for c in ws[1]]
    if "ac2_quadratic" in headers:
        col_idx = headers.index("ac2_quadratic") + 1
    else:
        col_idx = len(headers) + 1
        c = ws.cell(row=1, column=col_idx, value="ac2_quadratic")
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="4472C4")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(col_idx)].width = 14

    # Match rows by Source_CSV + Config + Temperature + Sampling + Seed
    src_col = headers.index("Source_CSV") + 1 if "Source_CSV" in headers else None
    for row in range(2, ws.max_row + 1):
        # Locate the source jsonl if we can; the master CSV stores the CSV
        # name, not the JSONL. As a fallback we look for a JSONL that
        # matches (Config, Model, T, Address, Sampling, Seed).
        cfg = ws.cell(row=row, column=headers.index("Config") + 1).value
        model = ws.cell(row=row, column=headers.index("Model") + 1).value if "Model" in headers else None
        T_val = ws.cell(row=row, column=headers.index("Temperature") + 1).value if "Temperature" in headers else None
        addr = ws.cell(row=row, column=headers.index("Address mode") + 1).value if "Address mode" in headers else None
        samp = ws.cell(row=row, column=headers.index("Sampling") + 1).value if "Sampling" in headers else None
        seed = ws.cell(row=row, column=headers.index("Seed") + 1).value if "Seed" in headers else None

        # Search ac2_by_cell for a matching filename
        match_ac2 = None
        for cell_name, ac2 in ac2_by_cell.items():
            meta = parse_cell_name(cell_name)
            if not meta:
                continue
            if (meta["config_id"] == cfg
                and (model is None or meta["model"] == model)
                and (T_val is None or abs(meta["temperature"] - float(T_val)) < 1e-6)
                and (addr is None or meta["address"] == addr)
                and (samp is None or meta["sampling"] == samp)
                and (seed is None or meta["seed"] == int(seed))):
                match_ac2 = ac2
                break
        if match_ac2 is not None and not np.isnan(match_ac2):
            ws.cell(row=row, column=col_idx, value=round(float(match_ac2), 4))

    wb.save(xlsx)


# ---------------------------------------------------------------------------
# Rankings + markdown export
# ---------------------------------------------------------------------------
def write_ranking(rows: list[dict], outcome: str) -> None:
    df = pd.DataFrame([r for r in rows if r["outcome"] == outcome])
    if df.empty:
        return
    df = df.dropna(subset=["ac2_quadratic"]).copy()
    df["ac2_quadratic"] = df["ac2_quadratic"].round(4)

    # Per-phase ranking
    df["rank_in_phase"] = df.groupby("phase")["ac2_quadratic"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values(["phase", "rank_in_phase"])
    out_csv = OUT_DIR / f"ranking_{outcome}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  ✓ {out_csv.relative_to(ROOT)}")

    # Human-readable markdown
    md_lines = [f"# Gwet's AC2 (quadratic) ranking — {outcome}", ""]
    for phase, sub in df.groupby("phase"):
        md_lines.append(f"## {phase}   (n cells = {len(sub)})")
        md_lines.append("")
        md_lines.append("| Rank | AC2 | n matched | Config | Model | T | Sampling | Address | Seed | Cell |")
        md_lines.append("|-----:|----:|----------:|--------|-------|--:|----------|---------|-----:|------|")
        for _, r in sub.iterrows():
            md_lines.append(
                f"| {r['rank_in_phase']} | {r['ac2_quadratic']:.4f} | {int(r['n_matched'])} | "
                f"{r.get('config_id','')} | {r.get('model','')} | {r.get('temperature','')} | "
                f"{r.get('sampling','')} | {r.get('address','')} | {r.get('seed','')} | "
                f"`{r['cell']}` |"
            )
        md_lines.append("")
    out_md = OUT_DIR / f"ranking_{outcome}.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  ✓ {out_md.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outcome", type=str, default=None,
                   help="Restrict to a single outcome (default: all).")
    p.add_argument("--no-excel", action="store_true",
                   help="Skip augmenting outputs/per_outcome_scores/*.xlsx.")
    args = p.parse_args()

    print("Sanity-checking AC2 implementation …")
    _sanity()
    print("  ✓ passed")

    cells = discover_cells()
    if args.outcome:
        cells = [c for c in cells if c[0] == args.outcome]
    print(f"\nDiscovered {len(cells)} cell JSONL files under {CALIB}")

    gts = {o: load_gt(o) for o in VALID_RANGES if not args.outcome or o == args.outcome}

    rows: list[dict] = []
    for outcome, jsonl in cells:
        try:
            rec = compute_cell_ac2(jsonl, outcome, gts[outcome])
            rows.append(rec)
        except Exception as e:
            print(f"  ✗ {jsonl.name}: {e}")

    if not rows:
        print("No cells produced AC2 scores.")
        return

    master = pd.DataFrame(rows)
    master.to_csv(OUT_DIR / "all_cells.csv", index=False)
    print(f"\n✓ Master table: {(OUT_DIR / 'all_cells.csv').relative_to(ROOT)}  "
          f"({len(master)} cells)")

    # Per-outcome rankings + markdown
    print("\nPhase-by-phase rankings:")
    for outcome in sorted({r["outcome"] for r in rows}):
        write_ranking(rows, outcome)

    # Excel augmentation
    if not args.no_excel:
        print("\nAugmenting per_outcome_scores Excel files …")
        for outcome in sorted({r["outcome"] for r in rows}):
            ac2_by_cell = {r["cell"]: r["ac2_quadratic"] for r in rows
                           if r["outcome"] == outcome and not np.isnan(r["ac2_quadratic"])}
            augment_excel(outcome, ac2_by_cell)
            print(f"  ✓ per_outcome_scores/{outcome}_all_phases.xlsx  "
                  f"({len(ac2_by_cell)} AC2 values written)")

    # Concise top-of-list summary
    print("\n" + "=" * 70)
    print("  Top-5 cells per outcome by AC2 (quadratic)")
    print("=" * 70)
    for outcome in sorted({r["outcome"] for r in rows}):
        sub = master[master["outcome"] == outcome].dropna(subset=["ac2_quadratic"])
        top = sub.nlargest(5, "ac2_quadratic")
        print(f"\n  {outcome}:")
        for _, r in top.iterrows():
            print(f"    AC2 = {r['ac2_quadratic']:.4f}   {r['phase']:35s}  {r['cell']}")


if __name__ == "__main__":
    main()
