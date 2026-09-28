"""
12d_descriptive_plots.py — Descriptive figures for Stage 3 (2×2 DV run).

Paper-ready plots produced from
    outputs/experiments/exp02_kurd_education_2x2/
    exp02_kurd_education_2x2_C11_gpt4omini_T08_ben_vs_cot_s0.jsonl

Outputs (no existing files overwritten) under
    outputs/experiments/exp02_kurd_education_2x2/descriptive_plots/

Figures produced
----------------
fig01_cell_means.{pdf,png}
    Four panels (one per DV). Within each panel: cell means for the four
    conditions (GE, GK, BE, BK) with 95 % CIs. Best paper-headline figure.

fig02_response_distributions.{pdf,png}
    Four panels (one per DV). Stacked bar chart per cell showing the
    response-category percentages — visualises *where* the distribution
    is and how it shifts across cells.

fig03_context_effect.{pdf,png}
    Single panel. Context main effect (mean(security) − mean(peace)) per
    DV with 95 % CI from paired persona-level differences. Quick
    summary of the size and direction of the headline finding.

fig04_g_vs_b_overlay.{pdf,png}
    Four panels (one per DV). Side-by-side density / histogram of the
    response distribution under security vs peace context, collapsed
    across framing.

Usage
-----
    python scripts/12d_descriptive_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "outputs" / "experiments" / "exp02_kurd_education_2x2"
JSONL = EXP_DIR / "exp02_kurd_education_2x2_C11_gpt4omini_T08_ben_vs_cot_s0.jsonl"
OUT_DIR = EXP_DIR / "descriptive_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


DV_INFO = {
    "dv_legitimacy": dict(
        label="Movement legitimacy",
        subtitle="“Bu grup meşru bir hak mücadelesi yürütüyor.”",
        scale=(1, 5),
        cat_labels=["1\nStrongly\ndisagree", "2\nDisagree",
                    "3\nNeither", "4\nAgree", "5\nStrongly\nagree"],
    ),
    "dv_threat": dict(
        label="Threat perception",
        subtitle="“Bu grubun talepleri toplumsal huzuru ve ulusal birliği tehdit edici nitelik taşıyor.”",
        scale=(1, 5),
        cat_labels=["1\nStrongly\ndisagree", "2\nDisagree",
                    "3\nNeither", "4\nAgree", "5\nStrongly\nagree"],
    ),
    "dv_policy_support": dict(
        label="Policy support",
        subtitle="“Devlet okullarında Kürtçe anadilde eğitime yasal olarak izin verilmelidir.”",
        scale=(1, 5),
        cat_labels=["1\nStrongly\ndisagree", "2\nDisagree",
                    "3\nNeither", "4\nAgree", "5\nStrongly\nagree"],
    ),
    "dv_behavioral_intent": dict(
        label="Behavioral intent",
        subtitle="“Böyle bir yürüyüş düzenlendiğinde aşağıdakilerden hangisini yapardınız?”",
        scale=(1, 6),
        cat_labels=["1\nAttend\nin person", "2\nShare on\nsocial media",
                    "3\nSupport,\nnot publicly", "4\nOppose,\nnot publicly",
                    "5\nPost counter\non SM", "6\nAttend counter\nin person"],
    ),
}
DV_LIST = list(DV_INFO.keys())

CELL_ORDER = ["GE", "GK", "BE", "BK"]
CELL_LABEL = {
    "GE": "GE\nsecurity ×\nuniv. rights",
    "GK": "GK\nsecurity ×\nself-deter.",
    "BE": "BE\npeace ×\nuniv. rights",
    "BK": "BK\npeace ×\nself-deter.",
}
CELL_COLOR = {"GE": "#C6373A", "GK": "#E07B3F",
              "BE": "#3A82B5", "BK": "#5DA6C6"}
CTX_COLOR = {"G": "#C6373A", "B": "#3A82B5"}


def load() -> pd.DataFrame:
    rows = []
    for line in JSONL.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("parse_status") != "ok" or r.get("predicted_value") is None:
            continue
        rows.append({
            "respondent_id": r["respondent_id"],
            "condition":     r["condition"],
            "context":       r["factor_a"],
            "framing":       r["factor_b"],
            "item_id":       r["item_id"],
            "y":             int(r["predicted_value"]),
        })
    return pd.DataFrame(rows)


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, low_95CI, high_95CI) using normal-approx SE."""
    m = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


def paired_context_effect(df: pd.DataFrame, item: str) -> tuple[float, float, float, int]:
    """Within-subject (paired) context effect = mean(G) − mean(B) per persona."""
    sub = df[df["item_id"] == item]
    # mean per (rid, context) over the two framings
    p = (sub.groupby(["respondent_id", "context"])["y"]
            .mean().unstack("context"))
    p = p.dropna(subset=["G", "B"])
    d = (p["G"] - p["B"]).values
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se, len(d)


# ---------------------------------------------------------------------------
# fig01 — cell means with CIs (four panels)
# ---------------------------------------------------------------------------
def fig01_cell_means(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for ax, item in zip(axes, DV_LIST):
        info = DV_INFO[item]
        sub = df[df["item_id"] == item]
        xs = np.arange(4)
        means, los, his = [], [], []
        for cell in CELL_ORDER:
            vals = sub.loc[sub["condition"] == cell, "y"].values
            m, lo, hi = mean_ci(vals)
            means.append(m); los.append(lo); his.append(hi)
        colors = [CELL_COLOR[c] for c in CELL_ORDER]
        ax.bar(xs, means, color=colors, edgecolor="black", linewidth=0.6,
               yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
               capsize=4, alpha=0.9)
        for x, m in zip(xs, means):
            ax.text(x, m + 0.05, f"{m:.2f}", ha="center", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([CELL_LABEL[c] for c in CELL_ORDER], fontsize=8)
        ax.set_ylim(info["scale"][0] - 0.4, info["scale"][1] + 0.4)
        ax.set_title(info["label"], fontsize=11, fontweight="bold")
        ax.set_ylabel("mean response", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        # Add subtitle
        ax.text(0.5, -0.32, info["subtitle"], transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, style="italic", color="#555",
                wrap=True)
    plt.suptitle(
        "Cell means across the 2×2 design (n = 2,615 per cell, 95 % CI)\n"
        "C11 + VS-CoT + T = 0.8 + GPT-4o-mini + ben",
        fontsize=12, fontweight="bold", y=1.04
    )
    plt.tight_layout()
    out = OUT_DIR / "fig01_cell_means"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# fig02 — response distributions per cell, grouped bar
# ---------------------------------------------------------------------------
def fig02_response_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, item in zip(axes.flat, DV_LIST):
        info = DV_INFO[item]
        sub = df[df["item_id"] == item]
        lo, hi = info["scale"]
        bins = list(range(lo, hi + 1))
        x = np.arange(len(bins))
        width = 0.20
        offsets = [-1.5, -0.5, 0.5, 1.5]
        for off, cell in zip(offsets, CELL_ORDER):
            v = sub.loc[sub["condition"] == cell, "y"].values
            pct = [(v == k).mean() * 100 for k in bins]
            bars = ax.bar(x + off * width, pct, width, color=CELL_COLOR[cell],
                          edgecolor="black", linewidth=0.4, label=cell)
            for b, p in zip(bars, pct):
                if p > 3.5:
                    ax.text(b.get_x() + b.get_width() / 2, p + 0.5, f"{p:.0f}",
                            ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(info["cat_labels"], fontsize=7.5)
        ax.set_ylabel("% respondents", fontsize=9)
        ax.set_title(info["label"], fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncols=4, columnspacing=0.8)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Response distributions per cell (n = 2,615 per cell)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    out = OUT_DIR / "fig02_response_distributions"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# fig03 — context main effect across DVs
# ---------------------------------------------------------------------------
def fig03_context_effect(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels, effs, los, his = [], [], [], []
    for item in DV_LIST:
        m, lo, hi, n = paired_context_effect(df, item)
        labels.append(DV_INFO[item]["label"])
        effs.append(m); los.append(lo); his.append(hi)
    ys = np.arange(len(labels))[::-1]
    colors = ["#3A82B5" if e < 0 else "#C6373A" for e in effs]
    ax.hlines(ys, los, his, color="black", linewidth=1.4)
    ax.scatter(effs, ys, s=110, color=colors, edgecolor="black",
               linewidth=0.7, zorder=3)
    for y, m in zip(ys, effs):
        ax.text(m, y + 0.16, f"{m:+.3f}", ha="center", fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Context effect: mean(security) − mean(peace) [scale units]",
                  fontsize=10)
    ax.set_title("Context main effect by dependent variable\n(paired persona-level difference, 95 % CI)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    # Helper text
    ax.text(0.01, 0.02,
            "Negative ⇒ peace context raises the score    Positive ⇒ security context raises the score",
            transform=ax.transAxes, fontsize=8, style="italic", color="#444")
    plt.tight_layout()
    out = OUT_DIR / "fig03_context_effect"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# fig04 — G vs B side-by-side distribution overlays
# ---------------------------------------------------------------------------
def fig04_g_vs_b_overlay(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, item in zip(axes.flat, DV_LIST):
        info = DV_INFO[item]
        sub = df[df["item_id"] == item]
        lo, hi = info["scale"]
        bins = list(range(lo, hi + 1))
        x = np.arange(len(bins))
        width = 0.40
        for off, ctx, lbl in [(-0.5, "G", "Security context"),
                              (+0.5, "B", "Peace context")]:
            v = sub.loc[sub["context"] == ctx, "y"].values
            pct = [(v == k).mean() * 100 for k in bins]
            ax.bar(x + off * width, pct, width, color=CTX_COLOR[ctx],
                   edgecolor="black", linewidth=0.6, alpha=0.92, label=lbl)
            for xx, p in zip(x + off * width, pct):
                if p > 3:
                    ax.text(xx, p + 0.6, f"{p:.0f}", ha="center", fontsize=7.5)
        m_g = sub.loc[sub["context"] == "G", "y"].mean()
        m_b = sub.loc[sub["context"] == "B", "y"].mean()
        ax.set_xticks(x); ax.set_xticklabels(info["cat_labels"], fontsize=7.5)
        ax.set_ylabel("% respondents", fontsize=9)
        ax.set_title(f"{info['label']}  (mean: security {m_g:.2f} | peace {m_b:.2f})",
                     fontsize=10.5, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Distribution comparison: security vs peace context "
                 "(collapsed across framing)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    out = OUT_DIR / "fig04_g_vs_b_overlay"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Companion table — descriptive summary CSV
# ---------------------------------------------------------------------------
def descriptive_table(df: pd.DataFrame) -> None:
    rows = []
    for item in DV_LIST:
        sub = df[df["item_id"] == item]
        for cell in CELL_ORDER:
            v = sub.loc[sub["condition"] == cell, "y"].values
            m, lo, hi = mean_ci(v)
            rows.append({
                "dv": item, "cell": cell, "n": len(v),
                "mean": round(m, 3), "sd": round(v.std(ddof=1), 3),
                "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                "median": int(np.median(v)),
            })
    df_out = pd.DataFrame(rows)
    out = OUT_DIR / "descriptive_summary_table.csv"
    df_out.to_csv(out, index=False)
    print(f"  ✓ {out.relative_to(ROOT)}")


def main() -> None:
    print(f"Loading {JSONL} ...")
    df = load()
    print(f"  {len(df):,} OK records")

    print("\nProducing figures:")
    fig01_cell_means(df)
    fig02_response_distributions(df)
    fig03_context_effect(df)
    fig04_g_vs_b_overlay(df)
    descriptive_table(df)

    print(f"\nAll outputs under: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
