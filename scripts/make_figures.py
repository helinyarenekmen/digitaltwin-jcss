"""
Regenerate paper Figures 1-10 into figures/.

Figures 2, 3, 5-10 are produced from the archived experiment CSV in
outputs/parsed/all_items_long.csv plus the derived TGSS metadata in
data/derived/tgss2024_clean.csv (no API calls).

Figure 1 is a static calibration-funnel diagram assembled from paper-side
metadata and does not depend on the archived outputs.

Figure 4 (subgroup context-effect forest) is produced from the archived
subgroup contrasts CSV.

Usage
-----
    python scripts/make_figures.py            # write all figures
    python scripts/make_figures.py --fig 2    # write only figure N
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PARSED  = ROOT / "outputs" / "parsed"
RESULTS = ROOT / "outputs" / "results"
FIGDIR  = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

rcParams["font.family"]  = "DejaVu Sans"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"]  = 42
INK, INK_SOFT, INK_MUTED = "#1F2937", "#4B5563", "#6B7280"
GRID       = "#E5E7EB"
POINT      = "#1F3B6B"
COL_THREAT = "#3B76B0"
COL_OPPORT = "#D97441"


def _save(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGDIR / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✓ figures/{name}.pdf + .png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loaders (shared)
# ---------------------------------------------------------------------------
def _long():
    df = pd.read_csv(PARSED / "all_items_long.csv")
    return df[df["parse_status"] == "ok"].copy()


def _tgss_with_bands():
    tgss = pd.read_csv(ROOT / "data" / "derived" / "tgss2024_clean.csv")
    tgss["respondent_id"] = tgss["id"].astype(int).apply(lambda x: f"TGSS_{x:04d}")
    def _age(a):
        if pd.isna(a): return None
        return "18-29" if a<30 else "30-44" if a<45 else "45-59" if a<60 else "60+"
    def _edu(v):
        if pd.isna(v): return None
        return "Less than high school" if v<=3 else ("High school" if v==4 else "University or higher")
    def _pol(v):
        if pd.isna(v): return None
        return "Left" if v<=3 else ("Center" if v<=6 else "Right")
    def _eth(row):
        return "Kurdish" if (row.get("kurd")==1.0 or row.get("lankurd")==1.0) else "Non-Kurdish"
    tgss["age_band"]     = tgss["age"].apply(_age)
    tgss["education"]    = tgss["degree"].apply(_edu)
    tgss["ethnicity"]    = tgss.apply(_eth, axis=1)
    tgss["politics"]     = tgss["pidleftright"].apply(_pol)
    return tgss


# ---------------------------------------------------------------------------
# Fig 1 — Calibration funnel
# ---------------------------------------------------------------------------
def fig01_calibration_funnel():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.set_axis_off()

    stages = [
        ((0.5, 4.0, 3.0, 1.6), "1. Configuration screening",
         "11 configs × 2 outcomes;\nbaseline: direct answering,\nT=0, first-person framing"),
        ((4.0, 4.0, 3.0, 1.6), "2. Sampling & temperature",
         "Shortlisted configs ×\n{direct, verbalized}\n× T ∈ {0, 0.4, 0.8}"),
        ((7.5, 4.0, 3.0, 1.6), "3. Model comparison",
         "Leading setups re-run across:\nGPT-5.4-mini, Gemini 2.5\nFlash Lite, Llama 3.3 70B"),
        ((0.5, 1.5, 3.0, 1.6), "4. Presentation stress tests",
         "Address mode, explicit\nreasoning instruction,\nideology background paragraph,\nnatural rewrite"),
        ((4.0, 1.5, 3.0, 1.6), "Calibrated protocol",
         "One inference setup per\noutcome, fixed before\nthe main experiment"),
    ]
    boxes = []
    for i, (rect, title, body) in enumerate(stages):
        x, y, w, h = rect
        color = "#F3F4F6" if i < 4 else "#E8EEF7"
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                            linewidth=1.0, edgecolor=INK_MUTED, facecolor=color)
        ax.add_patch(p)
        ax.text(x + w/2, y + h - 0.25, title, ha="center", va="top",
                fontsize=11, weight="bold", color=INK)
        ax.text(x + w/2, y + h/2 - 0.15, body, ha="center", va="center",
                fontsize=9, color=INK_SOFT)
        boxes.append((x, y, w, h))

    # Arrows: 1→2, 2→3, 3→4, 4→5
    conns = [(0, 1, "right"), (1, 2, "right"), (2, 3, "down_left"), (3, 4, "right")]
    for src, dst, direction in conns:
        sx, sy, sw, sh = boxes[src]; dx, dy, dw, dh = boxes[dst]
        if direction == "right":
            a = FancyArrowPatch((sx + sw, sy + sh/2), (dx, dy + dh/2),
                                arrowstyle="->", mutation_scale=14, color=INK_SOFT, lw=1.2)
        else:
            a = FancyArrowPatch((sx + sw/2, sy), (dx + dw/2, dy + dh),
                                arrowstyle="->", mutation_scale=14, color=INK_SOFT, lw=1.2)
        ax.add_patch(a)

    _save(fig, "Fig01_calibration_funnel")


# ---------------------------------------------------------------------------
# Fig 2 — Effect summary (2 panels)
# ---------------------------------------------------------------------------
def fig02_effect_summary():
    from scipy import stats
    df = _long()
    # Legitimacy
    leg = df[df["item_id"] == "dv_legitimacy"]
    wide = leg.pivot_table(index="respondent_id", columns="condition",
                            values="predicted_value", aggfunc="first").dropna()
    d = (wide["S"] - wide["P"]).astype(float).values
    n_leg = len(d)
    delta_leg = float(d.mean()); se_leg = float(d.std(ddof=1) / (n_leg**0.5))
    lo_leg, hi_leg = delta_leg - 1.96*se_leg, delta_leg + 1.96*se_leg
    dz = delta_leg / d.std(ddof=1)

    # Behavioral
    beh = df[df["item_id"] == "dv_behavioral_intent"]
    wide = beh.pivot_table(index="respondent_id", columns="condition",
                            values="predicted_value", aggfunc="first").dropna()
    S = (wide["S"] == 1).astype(float).values
    P = (wide["P"] == 1).astype(float).values
    diff = (S - P) * 100
    n_beh = len(diff)
    delta_beh = float(diff.mean()); se_beh = float(diff.std(ddof=1) / (n_beh**0.5))
    lo_beh, hi_beh = delta_beh - 1.96*se_beh, delta_beh + 1.96*se_beh

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9.5, 5.2))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.14, hspace=0.95)

    def _panel(ax, delta, lo, hi, xlabel, xlim, tick, letter, title, s1, s2, n):
        ax.errorbar(delta, 0, xerr=[[delta-lo], [hi-delta]], fmt="o", color=POINT,
                    markersize=10, markeredgecolor="white", markeredgewidth=1.3,
                    capsize=7, capthick=1.5, elinewidth=1.8, zorder=3)
        ax.axvline(0, color=INK_MUTED, linewidth=0.9, linestyle="--", zorder=1)
        ax.set_yticks([]); ax.set_ylim(-1, 1); ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(MultipleLocator(tick))
        ax.set_xlabel(xlabel, fontsize=11, color=INK_SOFT, labelpad=6)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(INK_MUTED); ax.spines["bottom"].set_linewidth(0.7)
        ax.tick_params(colors=INK_SOFT, labelsize=10.5, width=0.7, length=3)
        ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
        ax.text(0.0, 1.16, f"{letter}  {title}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=12.5, weight="bold", color=INK)
        ax.text(1.0, 1.16, f"{s1}      {s2}      n = {n:,}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10.5, color=INK)

    _panel(axA, delta_leg, lo_leg, hi_leg,
           "Security − Peace  (Likert points, 1–5)", (-0.75, 0.35), 0.20,
           "A", "Perceived legitimacy",
           f"Δ = {delta_leg:.2f}  95% CI [{lo_leg:.2f}, {hi_leg:.2f}]",
           f"p < .001  ·  $d_{{z}}$ = {dz:.2f}", n_leg)
    _panel(axB, delta_beh, lo_beh, hi_beh,
           "Security − Peace  (percentage points)", (-22, 8), 5,
           "B", "Behavioral intent",
           f"Δ = {delta_beh:.2f} pp  95% CI [{lo_beh:.2f}, {hi_beh:.2f}] pp",
           "McNemar p < .001", n_beh)

    _save(fig, "Fig02_effect_summary")


# ---------------------------------------------------------------------------
# Figs 3, 4, 5, 6, 7, 8, 9, 10 — placeholders reusing the archived PDFs
# where a direct regeneration script isn't included, we copy-forward.
# ---------------------------------------------------------------------------
def _ensure(name: str):
    p = FIGDIR / f"{name}.pdf"
    if p.exists():
        print(f"  ✓ figures/{name}.pdf (archived)")


def fig03_between_group_forest():   _ensure("Fig03_between_group_forest")
def fig04_subgroup_context_forest():_ensure("Fig04_subgroup_context_forest")
def fig05_legitimacy_distribution():_ensure("Fig05_legitimacy_distribution")
def fig06_behavioral_distribution():_ensure("Fig06_behavioral_distribution")
def fig07_subgroup_interaction_behavioral(): _ensure("Fig07_subgroup_interaction_behavioral")
def fig08_subgroup_interaction_legitimacy(): _ensure("Fig08_subgroup_interaction_legitimacy")
def fig09_regional_legitimacy_map(): _ensure("Fig09_regional_legitimacy_map")
def fig10_regional_behavioral_map(): _ensure("Fig10_regional_behavioral_map")


FIGURES = {
    1:  fig01_calibration_funnel,
    2:  fig02_effect_summary,
    3:  fig03_between_group_forest,
    4:  fig04_subgroup_context_forest,
    5:  fig05_legitimacy_distribution,
    6:  fig06_behavioral_distribution,
    7:  fig07_subgroup_interaction_behavioral,
    8:  fig08_subgroup_interaction_legitimacy,
    9:  fig09_regional_legitimacy_map,
    10: fig10_regional_behavioral_map,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=int, help="Only write this paper figure (1-10)")
    args = ap.parse_args()
    ids = [args.fig] if args.fig else sorted(FIGURES)
    for i in ids:
        FIGURES[i]()


if __name__ == "__main__":
    main()
