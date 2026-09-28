"""
context_subgroup_means_peace.py — 2x2 subgroup means under the Peace
condition, for the S-vs-P context experiment family (exp08 / exp10 / exp11).

For each of the three DVs, produces a four-panel figure that shows the
mean response in the Peace condition by Ethnicity / Politics / Education
/ Age. Each panel is titled with the expected sociological gradient so
unexpected patterns are easy to spot.

Uses the current Kurdish definition (`kurd == 1 OR lankurd == 1`) that
matches the R-code's `kurd_all`.

Outputs (under <exp-dir>/subgroup_analysis/plots/):
    subgroup_means_peace__dv_legitimacy.{pdf,png}
    subgroup_means_peace__dv_policy_support.{pdf,png}
    subgroup_means_peace__dv_behavioral_intent.{pdf,png}

Usage
-----
    python scripts/context_subgroup_means_peace.py --exp exp08_context_only
    python scripts/context_subgroup_means_peace.py --exp exp10_context_peacev3
    python scripts/context_subgroup_means_peace.py --exp exp11_context_peacev4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[2]
TGSS_PATH = ROOT / "data" / "tgss2024_clean.csv"

VALID_EXPS = {
    "exp08_context_only":     "peace",
    "exp10_context_peacev3":  "peace (v3)",
    "exp11_context_peacev4":  "peace (v4)",
}

_p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
_p.add_argument("--exp", required=True, choices=sorted(VALID_EXPS),
                help="Experiment id (folder under outputs/experiments/).")
_args = _p.parse_args()

EXP_ID = _args.exp
PEACE_LABEL = VALID_EXPS[EXP_ID]
EXP_DIR = ROOT / "outputs" / "experiments" / EXP_ID
JSONL_LIKERT = EXP_DIR / f"{EXP_ID}_C7_gemini25flashlite_T08_ben_vs_cot_s0.jsonl"
JSONL_BINARY = EXP_DIR / f"{EXP_ID}_C7_gpt4omini_T0_ben_direct_s0.jsonl"
PLOT_DIR = EXP_DIR / "subgroup_analysis" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- palette ---------------------------------------------------------------
INK       = "#1F2937"
INK_SOFT  = "#4B5563"
INK_MUTED = "#6B7280"
GRID      = "#E5E7EB"
BAR       = "#1F8A7C"    # teal

DV_INFO = {
    "dv_legitimacy":        ("movement legitimacy",  "likert", (1, 5)),
    "dv_policy_support":    ("policy support",       "likert", (1, 5)),
    "dv_behavioral_intent": ("would participate (%)", "binary_pct", (0, 100)),
}


# --- data helpers ----------------------------------------------------------
def load_records() -> pd.DataFrame:
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


def age_band(a):
    if pd.isna(a): return None
    if a < 30: return "18-29"
    if a < 45: return "30-44"
    if a < 60: return "45-59"
    return "60+"


DEGREE_TO_ED = {
    1.0: "< high school", 2.0: "< high school", 3.0: "< high school",
    4.0: "High school",
    5.0: "University", 6.0: "University", 7.0: "University", 8.0: "University",
}


def politics_band(v):
    if pd.isna(v): return None
    if v <= 3: return "Left"
    if v <= 6: return "Center"
    return "Right"


def load_subgroups() -> pd.DataFrame:
    df = pd.read_csv(TGSS_PATH, low_memory=False,
                     usecols=["id", "age", "degree", "kurd", "lankurd",
                              "pidleftright"])
    df["respondent_id"] = df["id"].astype(int).apply(lambda x: f"TGSS_{x:04d}")
    df["ethnicity"]  = np.where(
        (df["kurd"] == 1.0) | (df["lankurd"] == 1.0), "Kurdish", "Non-Kurdish")
    df["age_grp"]    = df["age"].apply(age_band)
    df["education"]  = df["degree"].map(DEGREE_TO_ED)
    df["politics"]   = df["pidleftright"].apply(politics_band)
    return df[["respondent_id", "ethnicity", "age_grp", "education", "politics"]]


# --- plot ------------------------------------------------------------------
DIMENSIONS = [
    ("ethnicity",  ["Kurdish", "Non-Kurdish"],
     "Ethnicity",  "expected  Kurdish > Non-Kurdish"),
    ("politics",   ["Left", "Center", "Right"],
     "Politics",   "expected  Left > Right"),
    ("education",  ["University", "High school", "< high school"],
     "Education",  "expected  more schooling > less"),
    ("age_grp",    ["18-29", "30-44", "45-59", "60+"],
     "Age",        "expected  younger > older"),
]


def stat(y: pd.Series, kind: str) -> float:
    if kind == "binary_pct":
        return (y == 1).mean() * 100
    return y.mean()


def draw(fig, ax, values: list[float], labels: list[str], *,
         title: str, gradient: str, xmin: float, xmax: float,
         fmt: str) -> None:
    ys = list(range(len(labels)))[::-1]
    ax.barh(ys, values, color=BAR, edgecolor="none",
            height=0.55, zorder=3)
    for y, v in zip(ys, values):
        ax.text(v + (xmax - xmin) * 0.015, y, fmt.format(v),
                va="center", ha="left",
                fontsize=13, color=INK, weight="semibold")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=12, color=INK)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{title}  ·  {gradient}",
                 loc="left", fontsize=13, color=INK,
                 weight="semibold", pad=12)


def render(dv: str) -> None:
    rcParams["font.family"] = "DejaVu Sans"
    rcParams["pdf.fonttype"] = 42; rcParams["ps.fonttype"] = 42

    label, kind, (lo, hi) = DV_INFO[dv]

    exp = load_records()
    sub = load_subgroups()
    df = exp.merge(sub, on="respondent_id", how="inner")
    peace = df[(df["condition"] == "P") & (df["item_id"] == dv)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 8),
                             constrained_layout=True)

    for ax, (dim, order, title, gradient) in zip(axes.flat, DIMENSIONS):
        vals = []
        for level in order:
            slice_ = peace[peace[dim] == level]["y"]
            vals.append(stat(slice_, kind) if len(slice_) > 0 else np.nan)
        vmax = max(vals) * 1.20
        if kind == "binary_pct":
            xmin, xmax_ax = 0, max(vmax, 10)
            fmt = "{:.1f}%"
        else:
            xmin, xmax_ax = 1.0, min(hi, max(vmax, 2.0))
            fmt = "{:.2f}"
        draw(fig, ax, vals, order,
             title=title, gradient=gradient,
             xmin=xmin, xmax=xmax_ax, fmt=fmt)

    if kind == "binary_pct":
        caption = (f"Mean {label} under {PEACE_LABEL}, by subgroup. "
                   "Expected sociological gradients appear before estimating "
                   "treatment effects.")
    else:
        caption = (f"Mean {label} (1–5 Likert) under {PEACE_LABEL}, by subgroup. "
                   "Expected sociological gradients appear before estimating "
                   "treatment effects.")
    fig.text(0.02, -0.02, caption, fontsize=10.5, color=INK_MUTED,
             ha="left", va="top", style="italic")

    stem = f"subgroup_means_peace__{dv}"
    for ext in (".png", ".pdf"):
        fp = PLOT_DIR / f"{stem}{ext}"
        fig.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  ✓ {fp.relative_to(ROOT)}")
    plt.close(fig)


def main() -> None:
    for dv in DV_INFO:
        render(dv)


if __name__ == "__main__":
    main()
