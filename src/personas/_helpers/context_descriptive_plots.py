"""
context_descriptive_plots.py — Descriptive figures for the S-vs-P context
experiment family (exp08 / exp10 / exp11).

Replaces 14d_context_plots.py.

Design: 2 conditions (S = security, P = peace), 3 DVs (2 Likert + 1 binary).

Outputs (under <exp-dir>/descriptive_plots/):
  fig01_cell_means.{pdf,png}
      Three panels (one per DV). Bars = condition means with 95 % CI.
  fig02_response_distributions.{pdf,png}
      Side-by-side response-category percentages for each DV.
  fig03_effect_summary.{pdf,png}
      Single-panel forest of the S − P contrast for each DV
      (paper-headline candidate).
  fig04_cdf_comparison.{pdf,png}
      Empirical CDFs for the two Likert DVs (Wasserstein visualisation).
  fig05_attendance_split.{pdf,png}
      Binary DV split as % "Katılırdım" per condition.
  descriptive_summary_table.csv
      Numeric summary for the paper appendix.

Usage
-----
    python scripts/context_descriptive_plots.py --exp exp08_context_only
    python scripts/context_descriptive_plots.py --exp exp10_context_peacev3
    python scripts/context_descriptive_plots.py --exp exp11_context_peacev4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

VALID_EXPS = {
    "exp08_context_only",
    "exp10_context_peacev3",
    "exp11_context_peacev4",
}

_p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
_p.add_argument("--exp", required=True, choices=sorted(VALID_EXPS),
                help="Experiment id (folder under outputs/experiments/).")
_args = _p.parse_args()

EXP_ID = _args.exp
EXP_DIR = ROOT / "outputs" / "experiments" / EXP_ID
JSONL_LIKERT = EXP_DIR / f"{EXP_ID}_C7_gemini25flashlite_T08_ben_vs_cot_s0.jsonl"
JSONL_BINARY = EXP_DIR / f"{EXP_ID}_C7_gpt4omini_T0_ben_direct_s0.jsonl"
OUT_DIR = EXP_DIR / "descriptive_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_ORDER = ["S", "P"]
CONDITION_LABEL = {"S": "S\nSecurity\ncontext", "P": "P\nPeace\ncontext"}
CONDITION_COLOR = {"S": "#C6373A", "P": "#3A82B5"}

DV_INFO = {
    "dv_legitimacy": dict(
        label="Movement legitimacy",
        subtitle='"Bu grup meşru bir hak mücadelesi yürütüyor"',
        scale=(1, 5),
        scale_type="likert",
        cat_labels=["1\nStrongly\ndisagree", "2\nDisagree",
                    "3\nNeither", "4\nAgree", "5\nStrongly\nagree"],
        source=JSONL_LIKERT,
        y_direction="↑ = agrees group is legitimate",
        protocol="C7 + Gemini 2.5 Flash Lite + VS-CoT + T=0.8",
    ),
    "dv_policy_support": dict(
        label="Policy support",
        subtitle='"Devlet okullarında Kürtçe anadilde eğitime izin verilmelidir"',
        scale=(1, 5),
        scale_type="likert",
        cat_labels=["1\nStrongly\ndisagree", "2\nDisagree",
                    "3\nNeither", "4\nAgree", "5\nStrongly\nagree"],
        source=JSONL_LIKERT,
        y_direction="↑ = supports policy",
        protocol="C7 + Gemini 2.5 Flash Lite + VS-CoT + T=0.8",
    ),
    "dv_behavioral_intent": dict(
        label="Behavioral intent",
        subtitle='"Böyle bir yürüyüş düzenlense yürüyüşe bizzat katılır mıydınız?"',
        scale=(1, 2),
        scale_type="binary",
        cat_labels=["1\nKatılırdım\n(would attend)",
                    "2\nKatılmazdım\n(would not)"],
        source=JSONL_BINARY,
        y_direction="↓ = would attend (lower = supportive)",
        protocol="C7 + GPT-4o-mini + Direct + T=0",
    ),
}
DV_LIST = list(DV_INFO)


# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    rows = []
    for fp in (JSONL_LIKERT, JSONL_BINARY):
        for line in fp.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("parse_status") != "ok" or r.get("predicted_value") is None:
                continue
            rows.append({
                "respondent_id": r["respondent_id"],
                "condition":     r["condition"],
                "item_id":       r["item_id"],
                "y":             int(r["predicted_value"]),
            })
    return pd.DataFrame(rows)


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    m = float(values.mean())
    if len(values) > 1:
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        return m, m - 1.96 * se, m + 1.96 * se
    return m, m, m


def paired_effect(df: pd.DataFrame, item: str) -> tuple[float, float, float, int]:
    """Paired within-persona effect S − P for a single item."""
    sub = df[df["item_id"] == item]
    wide = (sub.pivot_table(index="respondent_id", columns="condition",
                            values="y", aggfunc="first")
                .dropna(subset=["S", "P"]))
    d = (wide["S"] - wide["P"]).values
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se, n


# ---------------------------------------------------------------------------
def fig01_cell_means(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, item in zip(axes, DV_LIST):
        info = DV_INFO[item]
        sub = df[df["item_id"] == item]
        xs = np.arange(len(CONDITION_ORDER))
        means, los, his = [], [], []
        for cell in CONDITION_ORDER:
            vals = sub.loc[sub["condition"] == cell, "y"].values
            m, lo, hi = mean_ci(vals)
            means.append(m); los.append(lo); his.append(hi)
        colors = [CONDITION_COLOR[c] for c in CONDITION_ORDER]
        ax.bar(xs, means, color=colors, edgecolor="black", linewidth=0.6,
               yerr=[np.array(means) - np.array(los),
                     np.array(his) - np.array(means)],
               capsize=6, alpha=0.9, width=0.55)
        offset = 0.05 if info["scale"][1] > 2 else 0.02
        for x, m in zip(xs, means):
            ax.text(x, m + offset, f"{m:.3f}", ha="center", fontsize=10,
                    fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER],
                           fontsize=9.5)
        ax.set_ylim(info["scale"][0] - 0.4, info["scale"][1] + 0.4)
        ax.set_title(info["label"], fontsize=12, fontweight="bold")
        ax.set_ylabel("mean response", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.5, -0.25, info["subtitle"], transform=ax.transAxes,
                ha="center", va="top", fontsize=8, style="italic", color="#555")
        ax.text(0.5, -0.34, info["y_direction"], transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color="#333")
        ax.text(0.5, -0.42, info["protocol"], transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color="#888")
    plt.suptitle(
        "exp08 — Cell means for Security vs Peace context (n = 2,615 per cell, 95 % CI)",
        fontsize=13, fontweight="bold", y=1.06,
    )
    plt.tight_layout()
    out = OUT_DIR / "fig01_cell_means"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def fig02_response_distributions(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22)
    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])]
    for ax, item in zip(axes, DV_LIST):
        info = DV_INFO[item]
        sub = df[df["item_id"] == item]
        lo, hi = info["scale"]
        bins = list(range(lo, hi + 1))
        x = np.arange(len(bins))
        width = 0.38
        offsets = [-0.5, 0.5]
        for off, cell in zip(offsets, CONDITION_ORDER):
            v = sub.loc[sub["condition"] == cell, "y"].values
            pct = [(v == k).mean() * 100 for k in bins]
            bars = ax.bar(x + off * width, pct, width,
                          color=CONDITION_COLOR[cell],
                          edgecolor="black", linewidth=0.5,
                          label=f"{cell}  (n = {len(v):,})")
            for b, p in zip(bars, pct):
                if p > 1.5:
                    ax.text(b.get_x() + b.get_width() / 2, p + 0.6, f"{p:.0f}",
                            ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(info["cat_labels"], fontsize=8.5)
        ax.set_ylabel("% respondents", fontsize=10)
        ax.set_title(f"{info['label']}  ({info['scale_type']})",
                     fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.5, -0.16, info["subtitle"], transform=ax.transAxes,
                ha="center", va="top", fontsize=8, style="italic", color="#555")
    plt.suptitle("exp08 — Response distributions, Security vs Peace context",
                 fontsize=13, fontweight="bold", y=0.995)
    out = OUT_DIR / "fig02_response_distributions"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def fig03_effect_summary(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    labels, effs, los, his = [], [], [], []
    for item in DV_LIST:
        m, lo, hi, n = paired_effect(df, item)
        labels.append(DV_INFO[item]["label"])
        effs.append(m); los.append(lo); his.append(hi)
    ys = np.arange(len(labels))[::-1]
    colors = ["#3A82B5" if e < 0 else "#C6373A" for e in effs]
    ax.hlines(ys, los, his, color="black", linewidth=1.4)
    ax.scatter(effs, ys, s=120, color=colors, edgecolor="black",
               linewidth=0.7, zorder=3)
    for y, m in zip(ys, effs):
        ax.text(m, y + 0.18, f"{m:+.3f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Paired context effect: mean(Security) − mean(Peace)   [scale units]",
                  fontsize=10)
    ax.set_title("exp08 — Context main effect (paired persona-level difference, 95 % CI)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.text(0.01, 0.02,
            "Negative ⇒ peace context raises the score.    "
            "Positive ⇒ security context raises the score.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#444")
    plt.tight_layout()
    out = OUT_DIR / "fig03_effect_summary"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def fig04_cdf_comparison(df: pd.DataFrame) -> None:
    likert_items = [it for it in DV_LIST if DV_INFO[it]["scale_type"] == "likert"]
    fig, axes = plt.subplots(1, len(likert_items), figsize=(6.5 * len(likert_items), 5))
    if len(likert_items) == 1:
        axes = [axes]
    for ax, item in zip(axes, likert_items):
        info = DV_INFO[item]
        sub = df[df["item_id"] == item]
        lo, hi = info["scale"]
        xx = np.arange(lo, hi + 1)
        for cell in CONDITION_ORDER:
            v = sub.loc[sub["condition"] == cell, "y"].values
            counts = np.array([(v == k).sum() for k in xx], dtype=float)
            cdf = np.cumsum(counts / counts.sum())
            ax.plot(xx, cdf, "-o", linewidth=2.5, markersize=8,
                    color=CONDITION_COLOR[cell], label=f"{cell}")
        # Shade area between CDFs → Wasserstein visualisation
        v_S = sub.loc[sub["condition"] == "S", "y"].values
        v_P = sub.loc[sub["condition"] == "P", "y"].values
        cdf_S = np.cumsum([(v_S == k).sum() for k in xx]) / len(v_S)
        cdf_P = np.cumsum([(v_P == k).sum() for k in xx]) / len(v_P)
        ax.fill_between(xx, cdf_S, cdf_P, alpha=0.15, color="gray")
        ax.set_xticks(xx)
        ax.set_xlabel("Likert response", fontsize=10)
        ax.set_ylabel("Cumulative proportion", fontsize=10)
        ax.set_title(info["label"], fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9,
                  title="Context", frameon=True)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.text(0.5, -0.15, "Shaded area ≈ Wasserstein-1 distance",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, style="italic", color="#555")
    plt.suptitle("exp08 — Empirical CDFs, Security vs Peace (Likert DVs)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "fig04_cdf_comparison"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def fig05_attendance_split(df: pd.DataFrame) -> None:
    sub = df[df["item_id"] == "dv_behavioral_intent"]
    xs = np.arange(len(CONDITION_ORDER))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cell in enumerate(CONDITION_ORDER):
        v = sub.loc[sub["condition"] == cell, "y"].values
        p_att = (v == 1).mean() * 100
        p_not = (v == 2).mean() * 100
        ax.bar(i, p_att, width=0.55, color=CONDITION_COLOR[cell],
               edgecolor="black", linewidth=0.6, label="Katılırdım" if i == 0 else None)
        ax.bar(i, p_not, bottom=p_att, width=0.55,
               color=CONDITION_COLOR[cell], edgecolor="black",
               linewidth=0.6, alpha=0.35,
               label="Katılmazdım" if i == 0 else None)
        ax.text(i, p_att / 2, f"{p_att:.1f}%\nKatılırdım", ha="center",
                va="center", fontsize=11, color="white", fontweight="bold")
        ax.text(i, p_att + p_not / 2, f"{p_not:.1f}%\nKatılmazdım",
                ha="center", va="center", fontsize=10,
                color="#333")
    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER], fontsize=10)
    ax.set_ylabel("% respondents", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title(
        "exp08 — Behavioral intent split by context\n"
        '"Böyle bir yürüyüş düzenlense yürüyüşe bizzat katılır mıydınız?"',
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "fig05_attendance_split"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def descriptive_table(df: pd.DataFrame) -> None:
    rows = []
    for item in DV_LIST:
        sub = df[df["item_id"] == item]
        for cond in CONDITION_ORDER:
            v = sub.loc[sub["condition"] == cond, "y"].values
            m, lo, hi = mean_ci(v)
            rows.append({
                "dv": item, "condition": cond, "n": len(v),
                "mean": round(m, 3),
                "sd": round(v.std(ddof=1), 3) if len(v) > 1 else 0.0,
                "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                "median": int(np.median(v)),
            })
        m, lo, hi, n = paired_effect(df, item)
        pooled = sub["y"].values
        sd_pooled = pooled.std(ddof=1) if len(pooled) > 1 else 0.0
        rows.append({
            "dv": item, "condition": "S−P (paired)", "n": n,
            "mean": round(m, 3),
            "sd": round(sd_pooled, 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            "median": "",
        })
    out = OUT_DIR / "descriptive_summary_table.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  ✓ {out.relative_to(ROOT)}")


def main() -> None:
    print("Loading data …")
    df = load()
    print(f"  {len(df):,} OK records")

    print("\nProducing figures:")
    fig01_cell_means(df)
    fig02_response_distributions(df)
    fig03_effect_summary(df)
    fig04_cdf_comparison(df)
    fig05_attendance_split(df)
    descriptive_table(df)
    print(f"\nAll outputs under: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
