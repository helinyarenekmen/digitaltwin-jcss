"""
plot_step2a_jsd.py — Distribution plot for Step 2A calibration screening.

Renders one figure per outcome (womenwork, pacdemons, neilang) showing:
  - The ground-truth response distribution (bold reference bars)
  - Each of the 11 configuration predictions (C0, C1, C3–C11),
    sorted by JSD (best fit first)
  - JSD value annotated on each row

Step 2A protocol: Direct sampling, T=0, gpt-4o-mini, ben-dili, seed 0.

Outputs
-------
outputs/plots/
  step2a_configs_vs_gt_womenwork.{pdf,png}
  step2a_configs_vs_gt_pacdemons.{pdf,png}
  step2a_configs_vs_gt_neilang.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
CALIB = Path.home() / "Library" / "Caches" / "digitaltwin_calibration"
MASTER_CSV = CALIB / "master_results.csv"
GT_PATHS = {
    "pacdemons": ROOT / "agent-survey/agent-calibration/pacdemons/pacdemons_predictions_20260418_123339.csv",
    "womenwork": ROOT / "agent-survey/agent-calibration/womenwork/womenwork-data/womenwork_data_output.csv",
    "neilang":   ROOT / "agent-survey/agent-calibration/neilang/neilang-data/neilang_verbsampling_20260418_141028.csv",
}

OUTCOME_META = {
    "womenwork": dict(label="womenwork — 5-pt Likert",
                      subtitle='"Family life suffers when a woman works full time"',
                      scale=(1, 5),
                      cat_labels=["1\nStrongly\ndisagree",
                                  "2\nDisagree",
                                  "3\nNeither",
                                  "4\nAgree",
                                  "5\nStrongly\nagree"],
                      # Primary metric for the ordinal Likert outcome. JSD
                      # ignores category ordering and would be inappropriate.
                      metric="wasserstein",
                      metric_label="Wass"),
    "pacdemons": dict(label="pacdemons — binary participation",
                      subtitle="Participated in protest in the last 12 months",
                      scale=(1, 2),
                      cat_labels=["1\nYes\n(participated)",
                                  "2\nNo\n(did not)"],
                      metric="jsd",
                      metric_label="JSD"),
    "neilang":   dict(label="neilang — 3-pt ordinal",
                     subtitle="Comfort with a neighbour who speaks a different mother tongue",
                     scale=(1, 3),
                     cat_labels=["1\nUncomfortable",
                                 "2\nNeutral",
                                 "3\nComfortable"],
                     # Ordinal, so Wasserstein again.
                     metric="wasserstein",
                     metric_label="Wass"),
}

OUT_DIR = ROOT / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def load_gt(outcome: str) -> dict[str, int]:
    df = pd.read_csv(GT_PATHS[outcome])
    df["respondent_id"] = df["persona_id"].astype(str).str.zfill(4).apply(lambda x: f"TGSS_{x}")
    return dict(zip(df["respondent_id"], df[f"gt_{outcome}"].astype(int)))


def dist_from_jsonl(jsonl: Path, lo: int, hi: int) -> tuple[np.ndarray, int]:
    counts = np.zeros(hi - lo + 1)
    n = 0
    for line in jsonl.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("parse_status") != "ok" or r.get("predicted_value") is None:
            continue
        v = int(r["predicted_value"])
        if lo <= v <= hi:
            counts[v - lo] += 1
            n += 1
    return counts / n if n else counts, n


def gt_dist(outcome: str, gt: dict[str, int]) -> tuple[np.ndarray, int]:
    lo, hi = OUTCOME_META[outcome]["scale"]
    vals = np.array(list(gt.values()))
    counts = np.zeros(hi - lo + 1)
    for v in vals:
        if lo <= v <= hi:
            counts[v - lo] += 1
    return counts / len(vals), len(vals)


# ---------------------------------------------------------------------------
def collect_configs(outcome: str) -> list[tuple[str, np.ndarray, int, float]]:
    """For each Step-2A config, return (config_id, dist, n, metric) sorted by
    the outcome's primary metric (ascending — smaller = closer to GT)."""
    master = pd.read_csv(MASTER_CSV)
    step2a = master[master["outcome"] == outcome].copy()
    metric_col = OUTCOME_META[outcome]["metric"]

    lo, hi = OUTCOME_META[outcome]["scale"]
    cell_dir = CALIB / "screening" / outcome
    rows = []
    for _, r in step2a.iterrows():
        cfg = r["config_id"]
        cell_name = f"{cfg}_gpt4omini_T0_ben_direct_nocot_s0.jsonl"
        jsonl = cell_dir / cell_name
        if not jsonl.exists():
            continue
        dist, n = dist_from_jsonl(jsonl, lo, hi)
        rows.append((cfg, dist, n, float(r[metric_col])))
    rows.sort(key=lambda x: x[3])   # ascending metric (best first)
    return rows


# ---------------------------------------------------------------------------
def plot_outcome(outcome: str) -> None:
    meta = OUTCOME_META[outcome]
    lo, hi = meta["scale"]
    n_cats = hi - lo + 1
    gt = load_gt(outcome)
    gt_p, n_gt = gt_dist(outcome, gt)
    configs = collect_configs(outcome)
    n_configs = len(configs)

    fig, ax = plt.subplots(figsize=(11, 0.55 * (n_configs + 1) + 2.2))
    x = np.arange(n_cats)
    bar_h = 0.36

    # Row layout: config rows on top (best metric at top), GT row at the bottom.
    metric_label = meta["metric_label"]
    for i, (cfg, dist, n, m_val) in enumerate(configs):
        y_center = n_configs - i  # rows go top-to-bottom
        left = 0.0
        cmap = plt.cm.viridis(np.linspace(0.05, 0.95, n_cats))
        for k in range(n_cats):
            w = dist[k] * 100
            ax.barh(y_center, w, bar_h, left=left, color=cmap[k],
                    edgecolor="black", linewidth=0.5)
            if w > 3.5:
                ax.text(left + w / 2, y_center, f"{w:.0f}", ha="center",
                        va="center", fontsize=7.5,
                        color="white" if k < n_cats / 2 else "black")
            left += w
        ax.text(-2, y_center, f"{cfg}   {metric_label}={m_val:.3f}",
                ha="right", va="center", fontsize=9)

    # Ground-truth row (bottom, distinct)
    y_gt = 0
    cmap = plt.cm.viridis(np.linspace(0.05, 0.95, n_cats))
    left = 0.0
    for k in range(n_cats):
        w = gt_p[k] * 100
        ax.barh(y_gt, w, bar_h * 1.1, left=left, color=cmap[k],
                edgecolor="red", linewidth=1.3, hatch="//")
        if w > 3.5:
            ax.text(left + w / 2, y_gt, f"{w:.0f}", ha="center",
                    va="center", fontsize=8,
                    color="white" if k < n_cats / 2 else "black")
        left += w
    ax.text(-2, y_gt, f"GT (TGSS)   n={n_gt}", ha="right", va="center",
            fontsize=10, fontweight="bold", color="red")

    # X axis: percentages 0-100 with category label markers at cumulative
    # centres computed from GT distribution.
    ax.set_xlim(0, 100)
    ax.set_xlabel("% respondents", fontsize=10)
    # Category legend
    handles = [Patch(color=cmap[k], edgecolor="black",
                     label=meta["cat_labels"][k].replace("\n", " "))
               for k in range(n_cats)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, title="Response category", frameon=True)

    ax.set_yticks([])
    ax.set_title(
        f"Step 2A — configuration screening: response distributions vs ground truth\n"
        f"{meta['label']}   ·   {meta['subtitle']}",
        fontsize=11, fontweight="bold", pad=12,
    )
    # Reference line at the "GT band" to make bottom row visually distinct
    ax.axhline(y_gt + bar_h * 1.1 / 2 + 0.15, color="red", linewidth=0.5,
               linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = OUT_DIR / f"step2a_configs_vs_gt_{outcome}"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def main() -> None:
    for outcome in ("womenwork", "pacdemons", "neilang"):
        plot_outcome(outcome)
    print(f"\nAll outputs under: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
