"""
context_mean_diff.py — Paired mean-diff summary + forest plots for the
S-vs-P context experiment family (exp08 / exp10 / exp11).

Replaces:
    14e_context_mean_diff.py       (exp08, basic forest only)
    14f_forest_academic.py         (exp08, academic forest w/ hardcoded numbers)
    17c_exp10_mean_diff.py         (exp10)
    18c_exp11_mean_diff.py         (exp11)

For each of the three DVs, computes the paired mean difference
(Security − Peace) with a 95 % CI and a paired significance test, then
produces:

    <exp_dir>/mean_diff_table.csv       — table of {Δ, CI, p, d} per DV
    <exp_dir>/fig06_mean_diff_forest.{pdf,png}
    <exp_dir>/fig07_forest_academic.{pdf,png}

The two figures are the same layout; fig06 is retained for legacy
compatibility. Both feature two panels:
    Panel A — attitudinal outcomes (Likert scale points, 1–5)
    Panel B — behavioral outcome  (percentage points)

Usage
-----
    python scripts/context_mean_diff.py --exp exp08_context_only
    python scripts/context_mean_diff.py --exp exp10_context_peacev3
    python scripts/context_mean_diff.py --exp exp11_context_peacev4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

# Peace label shown on the x-axis and in the caption for each experiment.
PEACE_LABEL_BY_EXP = {
    "exp08_context_only":     "Peace",
    "exp10_context_peacev3":  "Peace (v3)",
    "exp11_context_peacev4":  "Peace (v4)",
}

DVS = [
    ("dv_legitimacy",       "Perceived legitimacy",         "likert",
     "C7 · Gemini 2.5 FL · VS-CoT · T = 0.8"),
    ("dv_policy_support",   "Policy support",               "likert",
     "C7 · Gemini 2.5 FL · VS-CoT · T = 0.8"),
    ("dv_behavioral_intent","Behavioral intent (would act)", "binary",
     "C7 · GPT-4o-mini · Direct · T = 0"),
]

# --- palette (matches earlier calibration/forest plots) --------------------
INK       = "#1F2937"
INK_SOFT  = "#4B5563"
INK_MUTED = "#6B7280"
GRID_LINE = "#E5E7EB"
ZERO_LINE = "#9CA3AF"
POINT     = "#1F3A5F"


# ---------------------------------------------------------------------------
def paired(csv_dir: Path, dv: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_dir / f"{dv}.csv")
    piv = df.pivot(index="respondent_id", columns="condition",
                   values=f"{dv}_pred").dropna()
    return piv["S"].to_numpy(), piv["P"].to_numpy()


def summarise(csv_dir: Path, dv: str, scale: str) -> dict:
    s, p = paired(csv_dir, dv)
    if scale == "binary":
        s01, p01 = (s == 1).astype(int), (p == 1).astype(int)
        diff = s01 - p01
        n = len(diff)
        mean_diff = diff.mean()
        se = diff.std(ddof=1) / np.sqrt(n)
        b = int(((s01 == 1) & (p01 == 0)).sum())
        c = int(((s01 == 0) & (p01 == 1)).sum())
        p_val = (stats.binomtest(min(b, c), b + c, 0.5).pvalue
                 if (b + c) > 0 else 1.0)
        test = "McNemar"
        s_mean, p_mean = s01.mean(), p01.mean()
    else:
        diff = s.astype(float) - p.astype(float)
        n = len(diff)
        mean_diff = diff.mean()
        se = diff.std(ddof=1) / np.sqrt(n)
        p_val = stats.ttest_rel(s, p).pvalue
        test = "paired t"
        s_mean, p_mean = s.mean(), p.mean()
    ci = 1.96 * se
    d = mean_diff / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
    return dict(dv=dv, scale=scale, n=n,
                s_mean=float(s_mean), p_mean=float(p_mean),
                mean_diff=float(mean_diff),
                lo=float(mean_diff - ci), hi=float(mean_diff + ci),
                p_val=float(p_val), d=float(d), test=test)


def fmt_p(pv: float) -> str:
    if pv < 1e-4: return "p < 0.0001"
    if pv < 1e-3: return f"p = {pv:.4f}"
    return f"p = {pv:.3f}"


# ---------------------------------------------------------------------------
def draw_panel(ax, rows, xlabel, xlim, xticks, *, letter, title):
    ys = list(range(len(rows)))[::-1]
    ax.axvline(0, color=ZERO_LINE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    for y, r in zip(ys, rows):
        # Binary DV lives in proportion space; panel is in percentage points.
        scale = 100.0 if r["scale"] == "binary" else 1.0
        ax.plot([r["lo"] * scale, r["hi"] * scale], [y, y],
                color=POINT, linewidth=3.2,
                solid_capstyle="round", zorder=3)
        ax.plot(r["mean_diff"] * scale, y, "o", color=POINT, markersize=11,
                markeredgecolor="white", markeredgewidth=1.4, zorder=4)
    ax.set_xlim(*xlim); ax.set_xticks(xticks)
    ax.set_yticks(ys); ax.set_yticklabels([""] * len(rows))
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED); ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=11, width=0.8,
                   length=4, pad=6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel, fontsize=12, color=INK_SOFT, labelpad=10)
    ax.grid(axis="x", color=GRID_LINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.0, 1.02, letter, transform=ax.transAxes,
            fontsize=14, color=INK, weight="bold", ha="left", va="bottom")
    ax.text(0.035, 1.02, title, transform=ax.transAxes,
            fontsize=12.5, color=INK, weight="semibold", ha="left", va="bottom")
    for y, r in zip(ys, rows):
        ax.text(-0.02, y + 0.14, r["label"], transform=ax.get_yaxis_transform(),
                fontsize=12, color=INK, ha="right", va="center", weight="semibold")
        ax.text(-0.02, y - 0.20, f"{r['config']}   ·   n = {r['n']:,}",
                transform=ax.get_yaxis_transform(),
                fontsize=9, color=INK_MUTED, ha="right", va="center", style="italic")
        p_txt = fmt_p(r["p_val"])
        if r["scale"] == "binary": p_txt = "McNemar " + p_txt
        scale = 100 if r["scale"] == "binary" else 1
        est = (f"Δ = {r['mean_diff']*scale:+.2f}   "
               f"95% CI [{r['lo']*scale:+.2f}, {r['hi']*scale:+.2f}]\n"
               f"{p_txt}   ·   Cohen's d = {r['d']:+.2f}")
        ax.text(1.02, y, est, transform=ax.get_yaxis_transform(),
                fontsize=10, color=INK, ha="left", va="center",
                family="DejaVu Sans")


def render(exp_id: str) -> None:
    rcParams["font.family"] = "DejaVu Sans"
    rcParams["pdf.fonttype"] = 42; rcParams["ps.fonttype"] = 42

    exp_dir = ROOT / "outputs" / "experiments" / exp_id
    csv_dir = exp_dir / "csv"
    if not csv_dir.exists():
        raise SystemExit(f"CSV dir not found: {csv_dir}. "
                         "Run run_context_experiment.py --export-csvs first.")

    peace_label = PEACE_LABEL_BY_EXP.get(exp_id, "Peace")

    rows_all = []
    for dv, label, scale, proto in DVS:
        r = summarise(csv_dir, dv, scale)
        r["label"] = label; r["config"] = proto
        rows_all.append(r)

    tbl = pd.DataFrame(rows_all)[
        ["dv","label","scale","config","n","s_mean","p_mean",
         "mean_diff","lo","hi","d","test","p_val"]
    ]
    tbl.to_csv(exp_dir / "mean_diff_table.csv", index=False)
    print(f"  ✓ {(exp_dir / 'mean_diff_table.csv').relative_to(ROOT)}")

    attitudinal = [r for r in rows_all if r["scale"] == "likert"]
    behavioral  = [r for r in rows_all if r["scale"] == "binary"]

    fig = plt.figure(figsize=(13, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.15], hspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[1, 0])

    draw_panel(ax_a, attitudinal,
               xlabel=f"Security − {peace_label}   (Likert scale points, 1–5)",
               xlim=(-0.55, 0.55),
               xticks=[-0.5, -0.25, 0, 0.25, 0.5],
               letter="A", title="Attitudinal outcomes")
    draw_panel(ax_b, behavioral,
               xlabel=f"Security − {peace_label}   (percentage points)",
               xlim=(-25, 25),
               xticks=[-25, -15, -5, 0, 5, 15, 25],
               letter="B", title="Behavioral outcome")

    fig.suptitle(f"Effect of Security Framing Relative to {peace_label} Framing",
                 fontsize=15, color=INK, weight="semibold", x=0.02, ha="left")
    fig.text(0.02, -0.02,
             f"Negative values indicate that {peace_label} framing elicits higher "
             "responses than Security framing.   Point = paired mean "
             "difference; whiskers = 95% CI.",
             fontsize=9.5, color=INK_MUTED, ha="left", va="top")

    for stem in ("fig06_mean_diff_forest", "fig07_forest_academic"):
        for ext in (".png", ".pdf"):
            fp = exp_dir / f"{stem}{ext}"
            fig.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  ✓ {(exp_dir / f'{stem}.png').relative_to(ROOT)}")
    plt.close(fig)
    print()
    print(tbl.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--exp", required=True,
                   choices=list(PEACE_LABEL_BY_EXP),
                   help="Experiment id (folder under outputs/experiments/)")
    args = p.parse_args()
    render(args.exp)


if __name__ == "__main__":
    main()
