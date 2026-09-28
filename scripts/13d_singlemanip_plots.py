"""
13d_singlemanip_plots.py — Descriptive figures for the single-manipulation
experiment (exp07_kurd_education_singlemanip).

Design: 4 single-manipulation conditions (S/P/R/SD), 3 DVs (2 Likert + 1 binary).

Outputs (no existing files overwritten) under
  outputs/experiments/exp07_kurd_education_singlemanip/descriptive_plots/

Figures
-------
fig01_condition_means.{pdf,png}
    Three panels (one per DV). Bars = condition means with 95% CI.

fig02_response_distributions.{pdf,png}
    One panel per DV. Grouped bars showing % per response category
    across the 4 conditions.

fig03_context_vs_framing.{pdf,png}
    Compare average of context conditions (S+P) with framing conditions
    (R+SD). Shows how much the two manipulation "types" move DV means.

fig04_pairwise_contrasts.{pdf,png}
    For each DV: forest plot of pairwise contrasts between conditions
    (S−P, R−SD, S−R, P−SD, S−SD, P−R) with 95% CI. Diagnostic.

Usage
-----
    python scripts/13d_singlemanip_plots.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "outputs" / "experiments" / "exp07_kurd_education_singlemanip"
JSONL_LIKERT = EXP_DIR / "exp07_kurd_education_singlemanip_C11_gpt4omini_T08_ben_vs_cot_s0.jsonl"
JSONL_BINARY = EXP_DIR / "exp07_kurd_education_singlemanip_C7_gpt4omini_T0_ben_direct_s0.jsonl"
OUT_DIR = EXP_DIR / "descriptive_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CONDITION_ORDER = ["S", "P", "R", "SD"]
CONDITION_LABEL = {
    "S":  "S\nSecurity\ncontext",
    "P":  "P\nPeace\ncontext",
    "R":  "R\nReligious\nframe",
    "SD": "SD\nSelf-det.\nframe",
}
CONDITION_COLOR = {
    "S":  "#C6373A",   # security = red
    "P":  "#3A82B5",   # peace = blue
    "R":  "#7A4E9C",   # religious = purple
    "SD": "#2A9D8F",   # self-determination = teal
}

DV_INFO = {
    "dv_legitimacy": dict(
        label="Movement legitimacy",
        subtitle='"Bu grup meşru bir hak mücadelesi yürütüyor."',
        scale_type="likert",
        scale=(1, 5),
        cat_labels=["1\nKesinlikle\nkatılmıyorum", "2\nKatılmıyorum",
                    "3\nNe katılıyorum\nne katılmıyorum",
                    "4\nKatılıyorum", "5\nKesinlikle\nkatılıyorum"],
        source=JSONL_LIKERT,
        y_direction="↑ = agrees group is legitimate",
    ),
    "dv_policy_support": dict(
        label="Policy support",
        subtitle='"Devlet okullarında Kürtçe anadilde eğitime izin verilmelidir."',
        scale_type="likert",
        scale=(1, 5),
        cat_labels=["1\nKesinlikle\nkatılmıyorum", "2\nKatılmıyorum",
                    "3\nNe katılıyorum\nne katılmıyorum",
                    "4\nKatılıyorum", "5\nKesinlikle\nkatılıyorum"],
        source=JSONL_LIKERT,
        y_direction="↑ = supports policy",
    ),
    "dv_behavioral_intent": dict(
        label="Behavioral intent",
        subtitle='"Böyle bir yürüyüş düzenlense yürüyüşe bizzat katılır mıydınız?"',
        scale_type="binary",
        scale=(1, 2),
        cat_labels=["1\nKatılırdım", "2\nKatılmazdım"],
        source=JSONL_BINARY,
        y_direction="↓ = would attend (lower = supportive)",
    ),
}
DV_LIST = list(DV_INFO.keys())


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


# ---------------------------------------------------------------------------
def fig01_condition_means(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, item in zip(axes, DV_LIST):
        info = DV_INFO[item]
        sub = df[df["item_id"] == item]
        xs = np.arange(4)
        means, los, his = [], [], []
        for cell in CONDITION_ORDER:
            vals = sub.loc[sub["condition"] == cell, "y"].values
            m, lo, hi = mean_ci(vals)
            means.append(m); los.append(lo); his.append(hi)
        colors = [CONDITION_COLOR[c] for c in CONDITION_ORDER]
        ax.bar(xs, means, color=colors, edgecolor="black", linewidth=0.6,
               yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
               capsize=4, alpha=0.9)
        for x, m in zip(xs, means):
            offset = 0.05 if info["scale"][1] > 2 else 0.02
            ax.text(x, m + offset, f"{m:.3f}", ha="center", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER], fontsize=8)
        ax.set_ylim(info["scale"][0] - 0.4, info["scale"][1] + 0.4)
        ax.set_title(info["label"], fontsize=11, fontweight="bold")
        ax.set_ylabel("mean response", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.5, -0.28, info["subtitle"], transform=ax.transAxes,
                ha="center", va="top", fontsize=7, style="italic", color="#555")
        ax.text(0.5, -0.42, info["y_direction"], transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color="#333")
    plt.suptitle(
        "Condition means across the 4 single manipulations (n = 2,615 per cell, 95% CI)\n"
        "S/P: C11 + VS-CoT + T=0.8  ·  behavioral_intent: C7 + Direct + T=0",
        fontsize=12, fontweight="bold", y=1.05,
    )
    plt.tight_layout()
    out = OUT_DIR / "fig01_condition_means"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def fig02_response_distributions(df: pd.DataFrame) -> None:
    # One panel per DV. Grouped bars, 4 conditions per response category.
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])]
    for ax, item in zip(axes, DV_LIST):
        info = DV_INFO[item]
        sub = df[df["item_id"] == item]
        lo, hi = info["scale"]
        bins = list(range(lo, hi + 1))
        x = np.arange(len(bins))
        width = 0.20
        offsets = [-1.5, -0.5, 0.5, 1.5]
        for off, cell in zip(offsets, CONDITION_ORDER):
            v = sub.loc[sub["condition"] == cell, "y"].values
            pct = [(v == k).mean() * 100 for k in bins]
            bars = ax.bar(x + off * width, pct, width, color=CONDITION_COLOR[cell],
                          edgecolor="black", linewidth=0.4, label=cell)
            for b, p in zip(bars, pct):
                if p > 3:
                    ax.text(b.get_x() + b.get_width() / 2, p + 0.6, f"{p:.0f}",
                            ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(info["cat_labels"], fontsize=7.5)
        ax.set_ylabel("% respondents", fontsize=9)
        ax.set_title(f"{info['label']}  —  {info['scale_type']}", fontsize=10.5, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncols=4, columnspacing=0.8)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Response distributions per condition (n = 2,615)",
                 fontsize=13, fontweight="bold", y=1.00)
    out = OUT_DIR / "fig02_response_distributions"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def fig03_context_vs_framing(df: pd.DataFrame) -> None:
    # Group S+P (contexts) vs R+SD (framings) — how much do the manipulation
    # families move DV means? Bar chart with paired points per DV.
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(DV_LIST))
    width = 0.28
    ctx_means, ctx_ci = [], []
    frm_means, frm_ci = [], []
    for item in DV_LIST:
        sub = df[df["item_id"] == item]
        ctx = sub[sub["condition"].isin(["S", "P"])]["y"].values
        frm = sub[sub["condition"].isin(["R", "SD"])]["y"].values
        m1, lo1, hi1 = mean_ci(ctx)
        m2, lo2, hi2 = mean_ci(frm)
        ctx_means.append(m1); ctx_ci.append((m1 - lo1, hi1 - m1))
        frm_means.append(m2); frm_ci.append((m2 - lo2, hi2 - m2))
    ax.bar(xs - width/2, ctx_means, width, color="#5B7DB1", edgecolor="black",
           linewidth=0.6, label="Context (S+P)",
           yerr=list(zip(*ctx_ci)), capsize=4)
    ax.bar(xs + width/2, frm_means, width, color="#B67C4F", edgecolor="black",
           linewidth=0.6, label="Framing (R+SD)",
           yerr=list(zip(*frm_ci)), capsize=4)
    for x, m in zip(xs - width/2, ctx_means):
        ax.text(x, m + 0.03, f"{m:.3f}", ha="center", fontsize=8)
    for x, m in zip(xs + width/2, frm_means):
        ax.text(x, m + 0.03, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([DV_INFO[dv]["label"] for dv in DV_LIST], fontsize=10)
    ax.set_ylabel("mean response", fontsize=10)
    ax.set_title("Context conditions (S+P) vs Framing conditions (R+SD) — mean DV response",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "fig03_context_vs_framing"
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def fig04_pairwise_contrasts(df: pd.DataFrame) -> None:
    # For each DV, all 6 pairwise contrasts with 95% CI (paired persona-level
    # differences).
    pairs = list(combinations(CONDITION_ORDER, 2))  # 6 pairs
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
    for ax, item in zip(axes, DV_LIST):
        info = DV_INFO[item]
        sub = df[df["item_id"] == item]
        wide = (sub.pivot_table(index="respondent_id", columns="condition",
                                values="y", aggfunc="first")
                    .dropna(subset=CONDITION_ORDER))
        labels = []
        effs, los, his = [], [], []
        for a, b in pairs:
            d = (wide[a] - wide[b]).values
            m = float(d.mean())
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            ci = 1.96 * se
            labels.append(f"{a} − {b}")
            effs.append(m); los.append(m - ci); his.append(m + ci)
        ys = np.arange(len(labels))[::-1]
        colors = ["#1f77b4" if (lo > 0 or hi < 0) else "#888"
                  for lo, hi in zip(los, his)]
        ax.hlines(ys, los, his, color="black", linewidth=1.2)
        ax.scatter(effs, ys, s=90, color=colors, edgecolor="black",
                   linewidth=0.6, zorder=3)
        for y, m in zip(ys, effs):
            ax.text(m, y + 0.16, f"{m:+.3f}", ha="center", fontsize=8)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("paired mean difference", fontsize=10)
        ax.set_title(info["label"], fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("All pairwise contrasts (paired persona-level differences, 95% CI)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "fig04_pairwise_contrasts"
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
                "mean": round(m, 3), "sd": round(v.std(ddof=1), 3) if len(v) > 1 else 0.0,
                "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                "median": int(np.median(v)),
            })
    out = OUT_DIR / "descriptive_summary_table.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  ✓ {out.relative_to(ROOT)}")


def main() -> None:
    print(f"Loading data …")
    df = load()
    print(f"  {len(df):,} OK records")

    print("\nProducing figures:")
    fig01_condition_means(df)
    fig02_response_distributions(df)
    fig03_context_vs_framing(df)
    fig04_pairwise_contrasts(df)
    descriptive_table(df)

    print(f"\nAll outputs under: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
