"""
plot_final_picks_vs_gt.py — Ground-truth comparison for the two final picks.

Outputs (no existing files overwritten):
  outputs/plots/final_picks_vs_gt.{pdf,png}        # 2x2 combined
  outputs/plots/final_pacdemons_vs_gt.{pdf,png}    # pacdemons only
  outputs/plots/final_womenwork_vs_gt.{pdf,png}    # womenwork only

Final picks:
  pacdemons  : C7 + Direct + T=0.8 + GPT-4o-mini + ben
  womenwork  : C11 + VS+ T=0.8 + GPT-4o-mini + ben

Usage:
  python scripts/plot_final_picks_vs_gt.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    f1_score,
    recall_score,
    cohen_kappa_score,
)

ROOT = Path(__file__).resolve().parents[1]
CALIB = Path.home() / "Library" / "Caches" / "digitaltwin_calibration"
GT_PD = ROOT / "agent-survey/agent-calibration/pacdemons/pacdemons_predictions_20260418_123339.csv"
GT_WW = ROOT / "agent-survey/agent-calibration/womenwork/womenwork-data/womenwork_data_output.csv"
PRED_PD = CALIB / "screening_T08/pacdemons/C7_gpt4omini_T08_ben_direct_nocot_s0.jsonl"
PRED_WW = CALIB / "screening_vs_cot_T08/womenwork/C11_gpt4omini_T08_ben_vs_cot_nocot_s0.jsonl"
PLOT_DIR = ROOT / "outputs" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COL_GT, COL_PRED = "#2E86AB", "#F18F01"


def load(jsonl: Path):
    rows = []
    for line in open(jsonl):
        d = json.loads(line)
        if d.get("parse_status") == "ok" and d.get("predicted_value") is not None:
            rows.append({"respondent_id": d["respondent_id"], "pred": int(d["predicted_value"])})
    return pd.DataFrame(rows)


def load_gt(csv: Path, col: str):
    g = pd.read_csv(csv)
    g["respondent_id"] = g["persona_id"].astype(str).str.zfill(4).apply(lambda x: f"TGSS_{x}")
    return g[["respondent_id", col]].rename(columns={col: "gt"}).astype({"gt": int})


def merge(jsonl, gt_csv, gt_col):
    p = load(jsonl)
    g = load_gt(gt_csv, gt_col)
    return p.merge(g, on="respondent_id")


# ---------------------------------------------------------------------------
# Panel renderers (take an existing Axes)
# ---------------------------------------------------------------------------
def panel_pacdemons_bars(ax, gt: np.ndarray, pred: np.ndarray):
    labels = ["1\n(Yes — participated)", "2\n(No)"]
    obs = [np.mean(gt == 1) * 100, np.mean(gt == 2) * 100]
    sim = [np.mean(pred == 1) * 100, np.mean(pred == 2) * 100]
    x = np.arange(2); w = 0.38
    b1 = ax.bar(x - w/2, obs, w, color=COL_GT, edgecolor="black",
                linewidth=0.7, label="Ground Truth (TGSS)")
    b2 = ax.bar(x + w/2, sim, w, color=COL_PRED, edgecolor="black",
                linewidth=0.7, label="Simulated (C7)")
    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2,
                    f"{b.get_height():.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% respondents", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(obs), max(sim)) * 1.18)
    mcc = matthews_corrcoef(gt, pred)
    rec = recall_score(gt, pred, pos_label=1, zero_division=0)
    f1 = f1_score(gt, pred, pos_label=1, zero_division=0)
    ax.set_title(f"pacdemons (n={len(gt)})  ·  MCC={mcc:.3f}  ·  Rec(+)={rec:.3f}  ·  F1(+)={f1:.3f}",
                 fontsize=10)


def panel_pacdemons_confusion(ax, gt: np.ndarray, pred: np.ndarray):
    cm = confusion_matrix(gt, pred, labels=[1, 2])
    cm_pct = cm / cm.sum() * 100
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred = 1\n(Yes)", "Pred = 2\n(No)"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["GT = 1\n(Yes)", "GT = 2\n(No)"], fontsize=9)
    ax.set_title("pacdemons — confusion matrix", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.65, label="count")


def panel_womenwork_bars(ax, gt: np.ndarray, pred: np.ndarray):
    labels = ["1\nStrongly\ndisagree", "2\nDisagree", "3\nNeither", "4\nAgree", "5\nStrongly\nagree"]
    bins = np.arange(1, 7) - 0.5
    obs = np.histogram(gt, bins=bins)[0] / len(gt) * 100
    sim = np.histogram(pred, bins=bins)[0] / len(pred) * 100
    x = np.arange(5); w = 0.38
    b1 = ax.bar(x - w/2, obs, w, color=COL_GT, edgecolor="black",
                linewidth=0.7, label="Ground Truth (TGSS)")
    b2 = ax.bar(x + w/2, sim, w, color=COL_PRED, edgecolor="black",
                linewidth=0.7, label="Simulated (C11)")
    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
                    f"{b.get_height():.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% respondents", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(obs), max(sim)) * 1.18)
    wd = wasserstein_distance(gt, pred)
    kw = cohen_kappa_score(gt, pred, weights="quadratic")
    ax.set_title(f"womenwork (n={len(gt)})  ·  Wass={wd:.3f}  ·  $\\kappa_w$={kw:.3f}  ·  "
                 f"mean: obs={gt.mean():.2f} / sim={pred.mean():.2f}", fontsize=10)


def panel_womenwork_cdf(ax, gt: np.ndarray, pred: np.ndarray):
    xx = np.arange(1, 6)
    obs_p = np.histogram(gt, bins=np.arange(1, 7)-0.5)[0] / len(gt)
    sim_p = np.histogram(pred, bins=np.arange(1, 7)-0.5)[0] / len(pred)
    gt_cdf, pred_cdf = np.cumsum(obs_p), np.cumsum(sim_p)
    ax.plot(xx, gt_cdf, "-o", color=COL_GT, linewidth=2.5, markersize=8, label="Ground Truth (TGSS)")
    ax.plot(xx, pred_cdf, "-s", color=COL_PRED, linewidth=2.5, markersize=8, label="Simulated (C11)")
    ax.fill_between(xx, gt_cdf, pred_cdf, alpha=0.18, color="gray")
    ax.set_xticks(xx); ax.set_xticklabels(["1","2","3","4","5"], fontsize=10)
    ax.set_xlabel("Likert response", fontsize=10)
    ax.set_ylabel("Cumulative proportion", fontsize=10)
    wd = wasserstein_distance(gt, pred)
    ax.set_title(f"womenwork — empirical CDFs (Wass-1 = {wd:.3f})", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)


# ---------------------------------------------------------------------------
# Figure renderers
# ---------------------------------------------------------------------------
def render_pacdemons(gt, pred, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})
    panel_pacdemons_bars(axes[0], gt, pred)
    panel_pacdemons_confusion(axes[1], gt, pred)
    plt.suptitle("pacdemons final pick vs ground truth\n"
                 "C7 + Direct + T=0.8 + GPT-4o-mini + first-person address",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def render_womenwork(gt, pred, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})
    panel_womenwork_bars(axes[0], gt, pred)
    panel_womenwork_cdf(axes[1], gt, pred)
    plt.suptitle("womenwork final pick vs ground truth\n"
                 "C11 + VS + T=0.8 + GPT-4o-mini + first-person address",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def render_combined(pd_gt, pd_pred, ww_gt, ww_pred, out: Path):
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.20,
                          width_ratios=[1.35, 1])
    axes = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]]
    panel_pacdemons_bars(axes[0][0], pd_gt, pd_pred)
    panel_pacdemons_confusion(axes[0][1], pd_gt, pd_pred)
    panel_womenwork_bars(axes[1][0], ww_gt, ww_pred)
    panel_womenwork_cdf(axes[1][1], ww_gt, ww_pred)
    # Row sub-titles
    fig.text(0.02, 0.96,
             "(a) pacdemons — binary protest participation\n"
             "    C7 + Direct + T=0.8 + GPT-4o-mini + first-person address",
             fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.49,
             "(b) womenwork — 5-pt Likert (family life suffers when a woman works full time)\n"
             "    C11 + VS + T=0.8 + GPT-4o-mini + first-person address",
             fontsize=11, fontweight="bold", ha="left", va="top")
    plt.suptitle("Final protocol distributions vs. ground truth",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


# ---------------------------------------------------------------------------
def main() -> None:
    pd_m = merge(PRED_PD, GT_PD, "gt_pacdemons")
    ww_m = merge(PRED_WW, GT_WW, "gt_womenwork")
    pd_gt, pd_pred = pd_m["gt"].values, pd_m["pred"].values
    ww_gt, ww_pred = ww_m["gt"].values, ww_m["pred"].values

    print("Plots:")
    render_pacdemons(pd_gt, pd_pred, PLOT_DIR / "final_pacdemons_vs_gt")
    render_womenwork(ww_gt, ww_pred, PLOT_DIR / "final_womenwork_vs_gt")
    render_combined(pd_gt, pd_pred, ww_gt, ww_pred, PLOT_DIR / "final_picks_vs_gt")

    # Summary stats for the report
    print("\nSummary:")
    print(f"  pacdemons (n={len(pd_gt)}):")
    print(f"    MCC          = {matthews_corrcoef(pd_gt, pd_pred):.4f}")
    print(f"    Recall(+)    = {recall_score(pd_gt, pd_pred, pos_label=1, zero_division=0):.4f}")
    print(f"    macro-F1     = {f1_score(pd_gt, pd_pred, average='macro', zero_division=0):.4f}")
    print(f"    Cohen κ      = {cohen_kappa_score(pd_gt, pd_pred):.4f}")
    print(f"    Obs +rate    = {np.mean(pd_gt == 1):.4f}")
    print(f"    Sim +rate    = {np.mean(pd_pred == 1):.4f}")
    print(f"  womenwork (n={len(ww_gt)}):")
    print(f"    Wasserstein  = {wasserstein_distance(ww_gt, ww_pred):.4f}")
    print(f"    weighted κ   = {cohen_kappa_score(ww_gt, ww_pred, weights='quadratic'):.4f}")
    print(f"    Obs mean     = {ww_gt.mean():.4f}")
    print(f"    Sim mean     = {ww_pred.mean():.4f}")


if __name__ == "__main__":
    main()
