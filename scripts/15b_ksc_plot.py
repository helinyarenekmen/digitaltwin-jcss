"""15b_ksc_plot.py — Response distribution plot for exp_ksc."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
JSONL = (ROOT / "outputs/experiments/exp_ksc"
              / "exp_ksc_C7_gemini25flashlite_T08_ben_vs_cot_s0.jsonl")
OUT_DIR = ROOT / "outputs/experiments/exp_ksc"

OPTIONS = {
    1: "Eğitim iki dilli olmalı,\nhem Türkçe hem ana dilde eğitim verilmeli",
    2: "Eğitim dili Türkçe olmalı,\nana dil okulda ayrıca öğretilmeli",
    3: "Eğitim dili Türkçe olmalı,\nokulda ana dili öğretmeye gerek yok",
    4: "Eğitim dili sadece\nana dil (Kurmanci/Zazaki) olmalı",
}

INK       = "#1F2937"
INK_SOFT  = "#4B5563"
INK_MUTED = "#6B7280"
GRID_LINE = "#E5E7EB"
BAR       = "#1F3A5F"
BAR_HI    = "#B87A2E"


def load_counts() -> tuple[Counter, int]:
    c: Counter = Counter()
    n = 0
    for line in JSONL.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("parse_status") == "ok":
            c[int(r["predicted_value"])] += 1
            n += 1
    return c, n


def render() -> None:
    rcParams["font.family"] = "DejaVu Sans"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"]  = 42

    counts, n = load_counts()
    ks = [1, 2, 3, 4]
    pcts = [counts[k] / n * 100 for k in ks]
    top = max(pcts)

    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    xs = list(range(len(ks)))
    colors = [BAR_HI if p == top else BAR for p in pcts]
    bars = ax.bar(xs, pcts, width=0.55, color=colors,
                  edgecolor="none", zorder=3)

    for x, p, k in zip(xs, pcts, ks):
        ax.text(x, p + top * 0.03, f"{p:.1f}%",
                ha="center", va="bottom", fontsize=13,
                color=INK, weight="semibold")
        ax.text(x, -top * 0.06, f"Seçenek {k}",
                ha="center", va="top", fontsize=11,
                color=INK, weight="semibold")
        ax.text(x, -top * 0.14, OPTIONS[k],
                ha="center", va="top", fontsize=9.5, color=INK_SOFT)

    ax.set_ylim(0, top * 1.18)
    ax.set_xlim(-0.55, len(ks) - 0.45)
    ax.set_xticks([])
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_ylabel("% simulated respondents", fontsize=11.5,
                  color=INK_SOFT, labelpad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(axis="y", colors=INK_SOFT, labelsize=10,
                   width=0.7, length=3)
    ax.grid(axis="y", color=GRID_LINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(
        "Ana dili eğitim politikası — simüle edilmiş tercih dağılımı",
        fontsize=14, color=INK, weight="semibold", loc="left", pad=12,
    )
    fig.text(0.01, -0.02,
             f"Soru: Sizce ana dili Türkçe olmayanlar için, ana dili "
             f"Kurmanci/Zazaki olanlar için okullarda eğitim dili nasıl olmalı?\n"
             f"Protocol: C7 · Gemini flash-lite (latest) · VS-CoT · T = 0.8 · ben   "
             f"·   n = {n:,}   ·   parse rate = 100%",
             fontsize=9.5, color=INK_MUTED, ha="left", va="top")

    png = OUT_DIR / "fig01_ksc_distribution.png"
    pdf = OUT_DIR / "fig01_ksc_distribution.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf,           bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {png.relative_to(ROOT)}")
    print(f"  ✓ {pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    render()
