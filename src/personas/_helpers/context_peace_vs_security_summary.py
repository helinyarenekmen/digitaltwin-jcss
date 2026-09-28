"""
context_peace_vs_security_summary.py — Two-panel Peace vs Security summary.

Left  panel: Dumbbell of Peace / Security means for the two Likert DVs
             (Movement legitimacy, Policy support), 1–5 scale.
Right panel: Bar chart of "would participate" percentage, Peace vs Security.

Reads exp08 CSVs and writes:
  outputs/figures/peace_vs_security_summary.png
  outputs/figures/peace_vs_security_summary.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

import argparse

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = {
    "exp08_context_only":     ("outputs/experiments/exp08_context_only",    "Peace"),
    "exp10_context_peacev3":  ("outputs/experiments/exp10_context_peacev3", "Peace (v3)"),
    "exp11_context_peacev4":  ("outputs/experiments/exp11_context_peacev4", "Peace (v4)"),
}
# short aliases for convenience
_ALIASES = {"exp08": "exp08_context_only",
            "exp10": "exp10_context_peacev3",
            "exp11": "exp11_context_peacev4"}

_ap = argparse.ArgumentParser()
_ap.add_argument("--exp", required=True,
                 choices=sorted(list(EXPERIMENTS) + list(_ALIASES)))
_args = _ap.parse_args()
_exp_id = _ALIASES.get(_args.exp, _args.exp)
_EXP_REL, PEACE_LABEL = EXPERIMENTS[_exp_id]
CSV_DIR = ROOT / _EXP_REL / "csv"
OUT_DIR = ROOT / _EXP_REL
OUT_DIR.mkdir(parents=True, exist_ok=True)
_OUT_STEM = "fig08_peace_vs_security_summary"

# Palette matching the reference image
PEACE     = "#1F8A7C"   # teal
SECURITY  = "#B87A2E"   # ochre
INK       = "#1F2937"
INK_SOFT  = "#4B5563"
INK_MUTED = "#6B7280"
GRID      = "#E5E7EB"
LINE      = "#C7CBD1"


def paired(dv: str) -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / f"{dv}.csv")
    return df.pivot(index="respondent_id", columns="condition",
                    values=f"{dv}_pred").dropna()


def compute() -> dict:
    leg = paired("dv_legitimacy")
    pol = paired("dv_policy_support")
    beh = paired("dv_behavioral_intent")
    return dict(
        leg_p=leg["P"].mean(), leg_s=leg["S"].mean(),
        pol_p=pol["P"].mean(), pol_s=pol["S"].mean(),
        beh_p=(beh["P"] == 1).mean() * 100,
        beh_s=(beh["S"] == 1).mean() * 100,
    )


def render() -> None:
    rcParams["font.family"] = "DejaVu Sans"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42

    s = compute()

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(15, 4.5),
        gridspec_kw=dict(width_ratios=[2.1, 1.0], wspace=0.28),
        constrained_layout=True,
    )

    # ---- LEFT PANEL: dumbbell -------------------------------------------
    labels = ["Movement\nlegitimacy", "Policy\nsupport"]
    ys = [1, 0]
    p_vals = [s["leg_p"], s["pol_p"]]
    s_vals = [s["leg_s"], s["pol_s"]]

    for y, pv, sv in zip(ys, p_vals, s_vals):
        lo, hi = sorted([pv, sv])
        ax_l.plot([lo, hi], [y, y], color=LINE, linewidth=4.5,
                  solid_capstyle="round", zorder=2)
        ax_l.plot(sv, y, "o", color=SECURITY, markersize=13,
                  markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        ax_l.plot(pv, y, "o", color=PEACE, markersize=13,
                  markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        ax_l.text(sv, y + 0.22, f"{sv:.2f}", ha="center", va="bottom",
                  color=SECURITY, fontsize=13, weight="semibold")
        ax_l.text(pv, y + 0.22, f"{pv:.2f}", ha="center", va="bottom",
                  color=PEACE, fontsize=13, weight="semibold")

    ax_l.set_yticks(ys)
    ax_l.set_yticklabels(labels, fontsize=12, color=INK)
    ax_l.set_ylim(-0.7, 1.7)
    ax_l.set_xlim(1.0, 3.0)
    ax_l.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])

    for side in ("top", "right", "left"):
        ax_l.spines[side].set_visible(False)
    ax_l.spines["bottom"].set_color(INK_MUTED)
    ax_l.spines["bottom"].set_linewidth(0.8)
    ax_l.tick_params(axis="x", colors=INK_SOFT, labelsize=11, length=3, width=0.7)
    ax_l.tick_params(axis="y", length=0, labelsize=12)
    ax_l.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax_l.set_axisbelow(True)

    ax_l.set_title("Likert outcomes  ·  mean on 1–5",
                   loc="left", fontsize=13.5, color=INK, weight="semibold", pad=12)

    # legend (top-right of left panel)
    ax_l.plot([], [], "o", color=PEACE,    markersize=10, label=PEACE_LABEL)
    ax_l.plot([], [], "o", color=SECURITY, markersize=10, label="Security")
    leg = ax_l.legend(loc="upper right", frameon=False, fontsize=11.5,
                      handletextpad=0.5, borderpad=0.2, labelspacing=0.4,
                      ncol=2, columnspacing=1.3)
    for t, c in zip(leg.get_texts(), [PEACE, SECURITY]):
        t.set_color(INK)

    # ---- RIGHT PANEL: bar chart -----------------------------------------
    xs = [0, 1]
    vals = [s["beh_p"], s["beh_s"]]
    colors = [PEACE, SECURITY]
    ax_r.bar(xs, vals, width=0.55, color=colors, edgecolor="none", zorder=3)
    ymax = max(vals) * 1.25
    for x, v in zip(xs, vals):
        ax_r.text(x, v + ymax * 0.03, f"{v:.1f}%", ha="center", va="bottom",
                  fontsize=14, color=INK, weight="semibold")
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels([PEACE_LABEL, "Security"], fontsize=12, color=INK)
    ax_r.set_ylim(0, ymax)
    ax_r.set_yticks([])

    for side in ("top", "right", "left"):
        ax_r.spines[side].set_visible(False)
    ax_r.spines["bottom"].set_color(INK_MUTED)
    ax_r.spines["bottom"].set_linewidth(0.8)
    ax_r.tick_params(axis="x", length=0, labelsize=12)

    ax_r.set_title("Would participate  ·  % of respondents",
                   loc="left", fontsize=13.5, color=INK, weight="semibold", pad=12)

    png = OUT_DIR / f"{_OUT_STEM}.png"
    pdf = OUT_DIR / f"{_OUT_STEM}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf,           bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {png.relative_to(ROOT)}")
    print(f"  ✓ {pdf.relative_to(ROOT)}")
    print()
    print(f"  Movement legitimacy   Peace={s['leg_p']:.2f}   Security={s['leg_s']:.2f}")
    print(f"  Policy support        Peace={s['pol_p']:.2f}   Security={s['pol_s']:.2f}")
    print(f"  Would participate     Peace={s['beh_p']:.1f}%   Security={s['beh_s']:.1f}%")


if __name__ == "__main__":
    render()
