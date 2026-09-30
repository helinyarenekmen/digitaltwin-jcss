"""
Regenerate every paper figure into `figures/`.

Fig 1 is a static funnel diagram assembled from paper-side metadata.
Figs 2-10 are regenerated from `outputs/parsed/`, `outputs/results/`, and
`data/derived/` — no archived copy-forward path.

Usage
-----
    python scripts/make_figures.py            # write all figures
    python scripts/make_figures.py --fig 2    # write only figure N
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment import figures as figs

FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

rcParams["font.family"]  = "DejaVu Sans"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"]  = 42
INK, INK_SOFT, INK_MUTED = "#1F2937", "#4B5563", "#6B7280"


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
    fig.savefig(FIGDIR / "Fig01_calibration_funnel.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGDIR / "Fig01_calibration_funnel.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ✓ figures/Fig01_calibration_funnel.pdf + .png")


FIGURES = {
    1:  fig01_calibration_funnel,
    2:  figs.fig02_effect_summary,
    3:  figs.fig03_between_group_forest,
    4:  figs.fig04_subgroup_context_forest,
    5:  figs.fig05_legitimacy_distribution,
    6:  figs.fig06_behavioral_distribution,
    7:  figs.fig07_subgroup_interaction_behavioral,
    8:  figs.fig08_subgroup_interaction_legitimacy,
    9:  figs.fig09_regional_legitimacy_map,
    10: figs.fig10_regional_behavioral_map,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=int, help="Only write this paper figure (1-10)")
    args = ap.parse_args()
    ids = [args.fig] if args.fig else sorted(FIGURES)
    for i in ids:
        FIGURES[i]()
        stems = {1:"Fig01_calibration_funnel", 2:"Fig02_effect_summary",
                  3:"Fig03_between_group_forest", 4:"Fig04_subgroup_context_forest",
                  5:"Fig05_legitimacy_distribution", 6:"Fig06_behavioral_distribution",
                  7:"Fig07_subgroup_interaction_behavioral",
                  8:"Fig08_subgroup_interaction_legitimacy",
                  9:"Fig09_regional_legitimacy_map",
                  10:"Fig10_regional_behavioral_map"}
        if i != 1:
            print(f"  ✓ figures/{stems[i]}.pdf + .png")


if __name__ == "__main__":
    main()
