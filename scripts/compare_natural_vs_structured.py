"""
compare_natural_vs_structured.py — Structured vs natural persona ablation.

Compares calibration metrics between the two persona formats for the
best-setup cells (pacdemons, womenwork), using the same ground truth.

Outputs
-------
outputs/natural_ablation/
    metrics_side_by_side.csv       — per-cell metrics table
    fig_structured_vs_natural.pdf  — comparison plot (bar / dumbbell)
    fig_structured_vs_natural.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, recall_score

ROOT = Path(__file__).resolve().parents[1]
STRUCT_CACHE = Path.home() / "Library" / "Caches" / "digitaltwin_calibration"
NAT_CACHE    = Path.home() / "Library" / "Caches" / "digitaltwin_calibration_natural"
OUT_DIR = ROOT / "outputs" / "natural_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_PATHS = {
    "pacdemons": ROOT / "agent-survey/agent-calibration/pacdemons/pacdemons_predictions_20260418_123339.csv",
    "womenwork": ROOT / "agent-survey/agent-calibration/womenwork/womenwork-data/womenwork_data_output.csv",
}

CELLS = {
    "pacdemons": dict(
        struct_path=STRUCT_CACHE / "screening_T08/pacdemons/C7_gpt4omini_T08_ben_direct_nocot_s0.jsonl",
        nat_path   =NAT_CACHE   / "screening_T08/pacdemons/C7_gpt4omini_T08_ben_direct_nocot_s0.jsonl",
        scale=(1, 2),  scale_type="binary",
        label="pacdemons  (C7 · gpt-4o-mini · Direct · T=0.8)",
    ),
    "womenwork": dict(
        struct_path=STRUCT_CACHE / "screening_vs_cot_T08/womenwork/C7_geminiflashlite_T08_ben_vs_cot_nocot_s0.jsonl",
        nat_path   =NAT_CACHE   / "screening_vs_cot_T08/womenwork/C7_geminiflashlite_T08_ben_vs_cot_nocot_s0.jsonl",
        scale=(1, 5),  scale_type="likert",
        label="womenwork  (C7 · Gemini FL · VS-CoT · T=0.8)",
    ),
}


def load_preds(jsonl: Path) -> dict[str, int]:
    out = {}
    for line in jsonl.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("parse_status") == "ok" and r.get("predicted_value") is not None:
            out[r["respondent_id"]] = int(r["predicted_value"])
    return out


def load_gt(outcome: str) -> dict[str, int]:
    df = pd.read_csv(GT_PATHS[outcome])
    df["rid"] = df["persona_id"].astype(str).str.zfill(4).apply(lambda x: f"TGSS_{x}")
    return dict(zip(df["rid"], df[f"gt_{outcome}"].astype(int)))


def dist(vals: np.ndarray, scale: tuple[int, int]) -> np.ndarray:
    lo, hi = scale
    counts = np.zeros(hi - lo + 1)
    for v in vals:
        if lo <= v <= hi: counts[v - lo] += 1
    return counts / max(counts.sum(), 1)


def metrics(gt: np.ndarray, pred: np.ndarray, scale: tuple[int, int],
             kind: str) -> dict:
    lo, hi = scale
    labels = list(range(lo, hi + 1))
    gt_p = dist(gt, scale); pr_p = dist(pred, scale)
    out = dict(
        n=len(gt),
        jsd = float(jensenshannon(gt_p, pr_p, base=2) ** 2),
        wass= float(wasserstein_distance(np.arange(lo, hi+1), np.arange(lo, hi+1),
                                          gt_p, pr_p)),
        pred_mean = float(pred.mean()),
        gt_mean   = float(gt.mean()),
        cohen_k   = float(cohen_kappa_score(gt, pred, labels=labels)),
    )
    if kind == "likert":
        out["quad_kappa"] = float(cohen_kappa_score(gt, pred, labels=labels,
                                                     weights="quadratic"))
        out["lin_kappa"]  = float(cohen_kappa_score(gt, pred, labels=labels,
                                                     weights="linear"))
    if kind == "binary":
        out["mcc"] = float(matthews_corrcoef(gt, pred))
        # recall of the "minority" (== code 1 in pacdemons)
        out["recall_minority"] = float(recall_score(
            (gt == 1).astype(int), (pred == 1).astype(int), zero_division=0))
    return out


def analyze() -> pd.DataFrame:
    rows = []
    for name, cell in CELLS.items():
        gt_map = load_gt(name)
        for source, jsonl in (("structured", cell["struct_path"]),
                              ("natural",    cell["nat_path"])):
            if not jsonl.exists():
                print(f"  ⚠  {source}: file not found → {jsonl}")
                continue
            preds = load_preds(jsonl)
            pairs = [(gt_map[r], preds[r]) for r in preds if r in gt_map]
            gt   = np.array([p[0] for p in pairs])
            pred = np.array([p[1] for p in pairs])
            m = metrics(gt, pred, cell["scale"], cell["scale_type"])
            rows.append(dict(outcome=name, source=source, **m))
    return pd.DataFrame(rows)


def make_plot(tbl: pd.DataFrame) -> None:
    rcParams["font.family"] = "DejaVu Sans"
    rcParams["pdf.fonttype"] = 42; rcParams["ps.fonttype"] = 42

    INK       = "#1F2937"
    INK_SOFT  = "#4B5563"
    INK_MUTED = "#6B7280"
    STRUCT    = "#1F3A5F"
    NATURAL   = "#B87A2E"

    # Common metrics to plot per outcome
    metrics_by_outcome = {
        "pacdemons": [("jsd","JSD (lower better)"),
                      ("wass","Wasserstein-1 (lower better)"),
                      ("mcc","MCC (higher better)"),
                      ("recall_minority","Recall minority (higher better)")],
        "womenwork": [("wass","Wasserstein-1 (lower better)"),
                      ("jsd","JSD (lower better)"),
                      ("quad_kappa","κ quadratic (higher better)"),
                      ("lin_kappa","κ linear (higher better)")],
    }

    fig, axes = plt.subplots(len(metrics_by_outcome), 4, figsize=(15, 6.5),
                              constrained_layout=True)

    for row_i, (outcome, mets) in enumerate(metrics_by_outcome.items()):
        sub = tbl[tbl["outcome"] == outcome].set_index("source")
        if "structured" not in sub.index or "natural" not in sub.index:
            continue
        for col_j, (key, title) in enumerate(mets):
            ax = axes[row_i, col_j]
            s_val = sub.loc["structured", key]
            n_val = sub.loc["natural",    key]
            bars = ax.bar([0, 1], [s_val, n_val], width=0.55,
                          color=[STRUCT, NATURAL], edgecolor="none", zorder=3)
            top = max(abs(s_val), abs(n_val), 0.01)
            for xi, val in enumerate([s_val, n_val]):
                ax.text(xi, val + (top * 0.05 if val >= 0 else -top * 0.05),
                        f"{val:.3f}", ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=10, color=INK, weight="semibold")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Struct.", "Natural"],
                                                        fontsize=10, color=INK_SOFT)
            ax.set_ylim(min(0, s_val, n_val) - top * 0.15, top * 1.30)
            ax.set_yticks([])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(INK_MUTED); ax.spines["bottom"].set_linewidth(0.7)
            ax.set_title(title, fontsize=10, color=INK, weight="semibold", pad=8, loc="left")

        axes[row_i, 0].set_ylabel(outcome, fontsize=13, color=INK,
                                  weight="bold", labelpad=15)

    fig.suptitle("Structured vs Natural persona rewrite  ·  best-setup calibration",
                 fontsize=14, color=INK, weight="semibold", x=0.01, ha="left")

    for ext in (".pdf", ".png"):
        fp = OUT_DIR / f"fig_structured_vs_natural{ext}"
        fig.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  ✓ {fp.relative_to(ROOT)}")
    plt.close(fig)


def main() -> None:
    tbl = analyze()
    if tbl.empty:
        raise SystemExit("No cells found. Run natural_rewrite_personas.py and "
                         "run_calibration_natural.py first.")
    csv = OUT_DIR / "metrics_side_by_side.csv"
    tbl.to_csv(csv, index=False)
    print(f"  ✓ {csv.relative_to(ROOT)}")
    print()
    print(tbl.to_string(index=False))
    print()
    make_plot(tbl)


if __name__ == "__main__":
    main()
