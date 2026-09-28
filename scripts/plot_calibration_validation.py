"""
plot_calibration_validation.py — Two calibration validation bar plots.

Reproduces just the two charts from the "Calibrated protocols against TGSS
ground truth" validation slide:

  fig01_pacdemons_positive_rate.pdf/png
      Observed vs simulated positive rate for
      pacdemons · C7 · Direct · T=0.8 · GPT-4o-mini · ben.

  fig02_womenwork_response_share.pdf/png
      Observed vs simulated response share across categories 1–5 for
      womenwork · C7 · VS-CoT · T=0.8 · Gemini 2.5 Flash Lite · ben.

Numbers are computed live from the calibration cell JSONLs so the plots
stay in sync with the underlying data.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CALIB = Path.home() / "Library" / "Caches" / "digitaltwin_calibration"

PAC_JSONL = CALIB / "screening_T08/pacdemons/C7_gpt4omini_T08_ben_direct_nocot_s0.jsonl"
PAC_GT    = ROOT / "agent-survey/agent-calibration/pacdemons/pacdemons_predictions_20260418_123339.csv"
WW_JSONL  = CALIB / "screening_vs_cot_T08/womenwork/C7_geminiflashlite_T08_ben_vs_cot_nocot_s0.jsonl"
WW_GT     = ROOT / "agent-survey/agent-calibration/womenwork/womenwork-data/womenwork_data_output.csv"

OUT_DIR = ROOT / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- colors matching the validation-slide palette -------------------------
INK        = "#1B2432"
INK_SOFT   = "#5A6674"
INK_LIGHT  = "#8A96A5"
GRID_LINE  = "#EEEAE1"
OBS_BAR    = "#4E6079"     # muted slate for observed
PAC_ACC    = "#1F8A7C"     # teal for pacdemons simulated
WW_ACC     = "#B87A2E"     # warm ochre for womenwork simulated


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_predictions(jsonl: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in jsonl.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("parse_status") == "ok" and r.get("predicted_value") is not None:
            out[r["respondent_id"]] = int(r["predicted_value"])
    return out


def load_gt(csv: Path, gt_col: str) -> dict[str, int]:
    g = pd.read_csv(csv)
    g["rid"] = g["persona_id"].astype(str).str.zfill(4).apply(lambda x: f"TGSS_{x}")
    return dict(zip(g["rid"], g[gt_col].astype(int)))


def pacdemons_rates() -> tuple[float, float]:
    preds = load_predictions(PAC_JSONL)
    gts   = load_gt(PAC_GT, "gt_pacdemons")
    pairs = [(gts[r], preds[r]) for r in preds if r in gts]
    gt = np.array([p[0] for p in pairs])
    pr = np.array([p[1] for p in pairs])
    return float((gt == 1).mean()) * 100, float((pr == 1).mean()) * 100


def womenwork_distributions() -> tuple[list[float], list[float]]:
    preds = load_predictions(WW_JSONL)
    gts   = load_gt(WW_GT, "gt_womenwork")
    pairs = [(gts[r], preds[r]) for r in preds if r in gts]
    gt = np.array([p[0] for p in pairs])
    pr = np.array([p[1] for p in pairs])
    obs = [float((gt == k).mean()) * 100 for k in range(1, 6)]
    sim = [float((pr == k).mean()) * 100 for k in range(1, 6)]
    return obs, sim


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
def style_axes(ax, *, ylabel: str, ylim: tuple[float, float],
               yticks: list[int]) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_LIGHT)
        ax.spines[side].set_linewidth(0.7)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.tick_params(axis="both", colors=INK_SOFT, labelsize=9.5,
                   width=0.7, length=3)
    ax.grid(axis="y", color=GRID_LINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10, labelpad=8)


# ---------------------------------------------------------------------------
# fig01 — pacdemons positive rate
# ---------------------------------------------------------------------------
def plot_pacdemons(out: Path) -> None:
    obs, sim = pacdemons_rates()

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([0, 1], [obs, sim], width=0.55,
                  color=[OBS_BAR, PAC_ACC], edgecolor="none", zorder=3)
    ymax = max(10.0, max(obs, sim) * 1.5)
    for bar, val in zip(bars, [obs, sim]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + ymax * 0.02,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=10.5, color=INK, weight="semibold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Observed", "Simulated"],
                       fontsize=11, color=INK)
    ax.set_title("Positive rate (%)", fontsize=11.5,
                 color=INK_SOFT, pad=14, loc="center")
    style_axes(ax, ylabel="",
               ylim=(0, ymax),
               yticks=list(range(0, int(ymax) + 1, 2)))
    fig.subplots_adjust(left=0.14, right=0.94, top=0.90, bottom=0.12)
    for ext in (".pdf", ".png"):
        p = out.with_suffix(ext)
        fig.savefig(p, dpi=250, bbox_inches="tight",
                    facecolor="white")
        print(f"  ✓ {p.relative_to(ROOT)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig02 — womenwork response share
# ---------------------------------------------------------------------------
def plot_womenwork(out: Path) -> None:
    obs, sim = womenwork_distributions()

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(1, 6)
    width = 0.36
    ax.bar(xs - width / 2, obs, width=width, color=OBS_BAR,
           edgecolor="none", label="Observed", zorder=3)
    ax.bar(xs + width / 2, sim, width=width, color=WW_ACC,
           edgecolor="none", label="Simulated", zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=11, color=INK)
    ax.set_title("Response share (%)", fontsize=11.5,
                 color=INK_SOFT, pad=14, loc="center")
    ymax = max(max(obs), max(sim)) * 1.25
    style_axes(ax, ylabel="",
               ylim=(0, ymax),
               yticks=list(range(0, int(ymax) + 1, 5)))
    leg = ax.legend(loc="upper left", fontsize=10, frameon=False,
                    handlelength=1.1, handleheight=0.9,
                    borderpad=0.2)
    for text in leg.get_texts():
        text.set_color(INK_SOFT)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.12)
    for ext in (".pdf", ".png"):
        p = out.with_suffix(ext)
        fig.savefig(p, dpi=250, bbox_inches="tight",
                    facecolor="white")
        print(f"  ✓ {p.relative_to(ROOT)}")
    plt.close(fig)


def main() -> None:
    plot_pacdemons(OUT_DIR / "fig01_pacdemons_positive_rate")
    plot_womenwork(OUT_DIR / "fig02_womenwork_response_share")


if __name__ == "__main__":
    main()
