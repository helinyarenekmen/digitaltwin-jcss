"""
06_plot_screening.py — Step 2B visualization: predicted vs. ground-truth distributions.

Produces one figure per outcome (three total):
  outputs/plots/screening_pacdemons.pdf  / .png
  outputs/plots/screening_womenwork.pdf  / .png
  outputs/plots/screening_neilang.pdf    / .png

And a combined summary figure:
  outputs/plots/screening_summary.pdf / .png

Run after 05_compute_metrics.py (uses master_results.csv + screening_ranking.csv).
Can also run while screening is still in progress — skips missing cells gracefully.

Usage:
  python scripts/06_plot_screening.py
  python scripts/06_plot_screening.py --dpi 200   # lower res for quick preview
"""

import argparse
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import CACHE_DIR as _CACHE, PERSONA_DIR as _PERSONA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV_PATH        = ROOT / "data" / "tgss2024_clean.csv"
CALIB_DIR       = Path(str(_CACHE))
PLOTS_DIR       = ROOT / "outputs" / "plots"

# ---------------------------------------------------------------------------
# Outcome metadata
# ---------------------------------------------------------------------------

OUTCOME_META = {
    "pacdemons": {
        "title": "Sokak Eylemi Katılımı (pacdemons)",
        "categories": {1: "Evet", 2: "Hayır"},
        # Diverging: minority (Evet) in red, majority (Hayır) in gray-blue
        "colors": {1: "#c0392b", 2: "#7f8c8d"},
        "valid_range": (1, 2),
        "rank_metric": "composite_score",
        "metric_label": "Composite (↓ better)",
    },
    "womenwork": {
        "title": "Kadın Çalışması Tutumu (womenwork)",
        "categories": {
            1: "Hiç katılmıyorum",
            2: "Katılmıyorum",
            3: "Ne/ne",
            4: "Katılıyorum",
            5: "Tamamen katılıyorum",
        },
        # RdYlGn diverging: red=disagree, yellow=neutral, green=agree
        "colors": {1: "#d73027", 2: "#fc8d59", 3: "#fee08b", 4: "#91cf60", 5: "#1a9850"},
        "valid_range": (1, 5),
        "rank_metric": "wasserstein",
        "metric_label": "Wasserstein (↓ better)",
    },
    "neilang": {
        "title": "Komşu Dil Toleransı (neilang)",
        "categories": {
            1: "Hiç rahatsız olmazdım",
            2: "Biraz rahatsız olurdum",
            3: "Çok rahatsız olurdum",
        },
        # Green → amber → red
        "colors": {1: "#27ae60", 2: "#f39c12", 3: "#c0392b"},
        "valid_range": (1, 3),
        "rank_metric": "wasserstein",
        "metric_label": "Wasserstein (↓ better)",
    },
}

# Config display order (C0 first as baseline, C1 second as benchmark, rest C3–C11)
CONFIG_ORDER = ["C0", "C1", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl_dir(outcome_dir: Path, temp_str: str = "T0",
                   seed: int = 0, address_mode: str = "ben",
                   model_alias: str = "gpt4omini",
                   cot: bool = False,
                   political_context: bool = False) -> dict[str, pd.DataFrame]:
    """Load JSONL files matching the given temperature + seed + address + model + cot."""
    result = {}
    cot_token = "cot" if cot else "nocot"
    pc_token = "_pc" if political_context else ""
    pattern = f"*_{model_alias}_{temp_str}_{address_mode}_*_{cot_token}{pc_token}_s{seed}.jsonl"
    for jsonl_path in sorted(outcome_dir.glob(pattern)):
        config_id = jsonl_path.stem.split("_")[0]
        rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        import json
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        if rows:
            result[config_id] = pd.DataFrame(rows)
    return result


def compute_dist(vals: np.ndarray, lo: int, hi: int) -> dict[int, float]:
    """Proportion of each category in [lo, hi]."""
    cats = range(lo, hi + 1)
    total = len(vals)
    if total == 0:
        return {c: 0.0 for c in cats}
    return {c: float((vals == c).sum()) / total for c in cats}


def load_ground_truth_dists(outcome: str) -> dict[int, float]:
    meta = OUTCOME_META[outcome]
    lo, hi = meta["valid_range"]
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    vals = df[outcome].dropna()
    vals = vals[vals.between(lo, hi)].astype(int).values
    return compute_dist(vals, lo, hi)


def build_dist_table(
    outcome: str,
    jsonl_map: dict[str, pd.DataFrame],
    gt_dist: dict[int, float],
    ranking: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Returns:
      dist_df — rows=configs+GT, cols=categories, values=proportions
      metric_map — {config_id: primary_metric_value}
    """
    meta = OUTCOME_META[outcome]
    lo, hi = meta["valid_range"]
    cats = list(range(lo, hi + 1))
    rank_metric = meta["rank_metric"]

    rows = {}
    metric_map: dict[str, float] = {}

    available_configs = [c for c in CONFIG_ORDER if c in jsonl_map]

    for config_id in available_configs:
        df = jsonl_map[config_id]
        ok = df[df["parse_status"] == "ok"]["predicted_value"].astype(int).values
        rows[config_id] = compute_dist(ok, lo, hi)
        if ranking is not None and config_id in ranking["config_id"].values:
            rv = ranking.loc[ranking["config_id"] == config_id, f"{outcome}_metric"]
            metric_map[config_id] = float(rv.iloc[0]) if len(rv) else np.nan
        else:
            metric_map[config_id] = np.nan

    rows["GT"] = gt_dist
    metric_map["GT"] = np.nan

    dist_df = pd.DataFrame(rows, index=cats).T  # shape: (n_configs+1, n_cats)
    return dist_df, metric_map


# ---------------------------------------------------------------------------
# Per-outcome plot
# ---------------------------------------------------------------------------

def plot_outcome(
    outcome: str,
    dist_df: pd.DataFrame,
    metric_map: dict[str, float],
    ranking: pd.DataFrame | None,
    dpi: int = 150,
) -> plt.Figure:
    meta = OUTCOME_META[outcome]
    cats = list(meta["categories"].keys())
    cat_labels = [meta["categories"][c] for c in cats]
    colors = [meta["colors"][c] for c in cats]

    bar_labels = list(dist_df.index)  # e.g. [C0, C1, C3, ..., GT]
    n_bars = len(bar_labels)
    x = np.arange(n_bars)

    # Build rank labels for X axis ticks
    def rank_label(cfg):
        if cfg == "GT":
            return "GT ★"
        m = metric_map.get(cfg, np.nan)
        if ranking is not None and cfg in ranking["config_id"].values:
            rk = ranking.loc[ranking["config_id"] == cfg, f"{outcome}_rank"]
            r_val = int(rk.iloc[0]) if len(rk) and not rk.isna().all() else "?"
            return f"{cfg}\n(#{r_val})"
        return cfg

    tick_labels = [rank_label(c) for c in bar_labels]

    fig, ax = plt.subplots(figsize=(max(10, n_bars * 0.85), 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bottoms = np.zeros(n_bars)
    bar_width = 0.72

    for i, (cat, col, lbl) in enumerate(zip(cats, colors, cat_labels)):
        heights = dist_df[cat].values
        bars = ax.bar(
            x, heights, bar_width,
            bottom=bottoms,
            color=col,
            label=lbl,
            zorder=3,
        )
        # Annotate proportions inside bars if segment is wide enough
        for j, (h, b) in enumerate(zip(heights, bottoms)):
            if h > 0.07:
                ax.text(
                    x[j], b + h / 2, f"{h:.0%}",
                    ha="center", va="center",
                    fontsize=7.5, color="white" if i in (0, len(cats) - 1) else "#333333",
                    fontweight="bold", zorder=5,
                )
        bottoms += heights

    # Highlight GT bar with thick black border
    gt_idx = bar_labels.index("GT")
    ax.add_patch(mpatches.FancyBboxPatch(
        (x[gt_idx] - bar_width / 2, 0), bar_width, 1.0,
        boxstyle="square,pad=0",
        linewidth=2.5, edgecolor="#222222", facecolor="none", zorder=6,
    ))

    # Highlight shortlist configs with dashed border
    if ranking is not None and "in_shortlist" in ranking.columns:
        sl_set = set(ranking.loc[ranking["in_shortlist"], "config_id"])
        for j, cfg in enumerate(bar_labels):
            if cfg in sl_set:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x[j] - bar_width / 2, 0), bar_width, 1.0,
                    boxstyle="square,pad=0",
                    linewidth=1.8, edgecolor="#2980b9",
                    linestyle="--", facecolor="none", zorder=6,
                ))

    ax.set_xlim(-0.6, n_bars - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Oran (%)", fontsize=11)
    ax.set_title(meta["title"], fontsize=13, fontweight="bold", pad=10)

    # Grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=colors[i], label=cat_labels[i])
        for i in range(len(cats))
    ]
    legend_handles += [
        mpatches.Patch(facecolor="none", edgecolor="#222222", linewidth=2, label="Ground Truth"),
        mpatches.Patch(facecolor="none", edgecolor="#2980b9", linewidth=1.8,
                       linestyle="dashed", label="Shortlist (Step 2C)"),
    ]
    # Lejant grafik alanının dışına (sağ tarafa) yerleştirilir
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8.5,
        framealpha=0.95,
        ncol=1,
        borderaxespad=0,
    )

    # Caption with metric values
    metric_vals = ", ".join(
        f"{c}={metric_map[c]:.3f}"
        for c in bar_labels if c != "GT" and not np.isnan(metric_map.get(c, np.nan))
    )
    if metric_vals:
        fig.text(
            0.5, -0.04,
            f"{meta['metric_label']}: {metric_vals}",
            ha="center", fontsize=8, color="#555555",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary 3-panel figure
# ---------------------------------------------------------------------------

def plot_summary(
    figs_data: list[tuple[str, pd.DataFrame, dict, pd.DataFrame | None]],
    dpi: int = 150,
) -> plt.Figure:
    """Compact panel figure (one column per outcome) for paper appendix."""
    n = len(figs_data)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5))
    fig.patch.set_facecolor("white")
    # matplotlib n=1'de tek Axes döner, n>1'de array — normalleştir
    axes_list = [axes] if n == 1 else list(axes)

    for ax, (outcome, dist_df, metric_map, ranking) in zip(axes_list, figs_data):
        meta = OUTCOME_META[outcome]
        cats = list(meta["categories"].keys())
        cat_labels_short = [meta["categories"][c] for c in cats]
        colors = [meta["colors"][c] for c in cats]

        bar_labels = list(dist_df.index)
        n_bars = len(bar_labels)
        x = np.arange(n_bars)
        bottoms = np.zeros(n_bars)

        for cat, col in zip(cats, colors):
            heights = dist_df[cat].values
            ax.bar(x, heights, 0.72, bottom=bottoms, color=col, zorder=3)
            bottoms += heights

        gt_idx = bar_labels.index("GT")
        ax.add_patch(mpatches.FancyBboxPatch(
            (x[gt_idx] - 0.36, 0), 0.72, 1.0,
            boxstyle="square,pad=0",
            linewidth=2.5, edgecolor="#222222", facecolor="none", zorder=6,
        ))

        ax.set_xlim(-0.5, n_bars - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=7.5, rotation=45, ha="right")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(meta["title"].split("(")[0].strip(), fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Compact legend below the subplot (grafik dışında)
        handles = [mpatches.Patch(facecolor=colors[i], label=cat_labels_short[i])
                   for i in range(len(cats))]
        ax.legend(
            handles=handles, fontsize=7,
            loc="upper center", bbox_to_anchor=(0.5, -0.15),
            ncol=min(len(cats), 3), framealpha=0.95,
        )

    fig.suptitle("Step 2A Screening: Tahmin vs. Gerçek Dağılımlar",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_TEMP_LABELS = {0.0: "T0", 0.3: "T03", 0.4: "T04", 0.7: "T07", 0.8: "T08", 1.0: "T1"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["direct", "vs_cot"], default="direct",
                        help="'direct' (Step 2A) or 'vs_cot' (verbalized sampling).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature used in filenames (default 0.0).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed used in filenames (default 0).")
    parser.add_argument("--address_mode", choices=["ben", "sen"], default="ben",
                        help="Address mode used in filenames (default ben).")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model used in filenames (default gpt-4o-mini).")
    parser.add_argument("--cot", action="store_true",
                        help="Plot from explicit-CoT JSONL files (with _cot_ suffix).")
    parser.add_argument("--political_context", action="store_true",
                        help="Plot from political-context JSONL files (with _pc_ suffix).")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI (default 150)")
    args = parser.parse_args()

    _MODEL_ALIASES = {"gpt-4o-mini": "gpt4omini", "gpt-5-mini": "gpt5mini",
                      "gpt-5.4-mini": "gpt54mini", "claude-haiku": "claudehaiku",
                      "claude-haiku-4-5-20251001": "claudehaiku",
                      "gemini-2.5-flash-lite": "geminiflashlite",
                      "gemini-2.5-flash": "geminiflash",
                      "gemini-2.5-pro": "geminipro",
                      "meta-llama/llama-3.3-70b-instruct": "llama3370b",
                      "meta-llama/llama-3.1-70b-instruct": "llama3170b",
                      "meta-llama/llama-3.1-8b-instruct":  "llama318b"}
    model_alias = _MODEL_ALIASES.get(args.model, args.model.replace("-", "").replace(".", ""))
    model_suffix = "" if args.model == "gpt-4o-mini" else f"_{model_alias}"

    temp_str = _TEMP_LABELS.get(args.temperature, f"T{args.temperature}".replace(".", ""))
    seed_suffix = "" if args.seed == 0 else f"_s{args.seed}"
    addr_suffix = "" if args.address_mode == "ben" else f"_{args.address_mode}"
    cot_suffix = "_cot" if args.cot else ""
    pc_suffix = "_pc" if args.political_context else ""
    if args.variant == "direct" and args.temperature == 0.0:
        screening_dir = CALIB_DIR / "screening"
        suffix = addr_suffix + model_suffix + cot_suffix + pc_suffix + seed_suffix
    elif args.variant == "direct":
        screening_dir = CALIB_DIR / f"screening_{temp_str}"
        suffix = f"_{temp_str}" + addr_suffix + model_suffix + cot_suffix + pc_suffix + seed_suffix
    elif args.temperature == 0.0:
        screening_dir = CALIB_DIR / "screening_vs_cot"
        suffix = "_vs_cot" + addr_suffix + model_suffix + cot_suffix + pc_suffix + seed_suffix
    else:
        screening_dir = CALIB_DIR / f"screening_vs_cot_{temp_str}"
        suffix = f"_vs_cot_{temp_str}" + addr_suffix + model_suffix + cot_suffix + pc_suffix + seed_suffix
    ranking_csv   = CALIB_DIR / f"screening_ranking{suffix}.csv"

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load ranking if available
    ranking = None
    if ranking_csv.exists():
        ranking = pd.read_csv(ranking_csv)
        print(f"Loaded ranking from {ranking_csv}")
    else:
        print(f"No {ranking_csv.name} found — run 05_compute_metrics.py first for ranks.")
        print("Plotting distributions without rank annotations.\n")

    # Load ground truth once
    df_gt_raw = pd.read_csv(CSV_PATH, encoding="utf-8")

    summary_data = []

    for outcome in ["pacdemons", "womenwork", "neilang"]:
        meta = OUTCOME_META[outcome]
        lo, hi = meta["valid_range"]

        # Ground truth distribution
        gt_vals = df_gt_raw[outcome].dropna()
        gt_vals = gt_vals[gt_vals.between(lo, hi)].astype(int).values
        gt_dist = compute_dist(gt_vals, lo, hi)
        print(f"\n[{outcome}] Ground truth: {len(gt_vals)} respondents")
        print(f"  GT dist: { {k: f'{v:.3f}' for k,v in gt_dist.items()} }")

        # Load JSONL predictions
        outcome_dir = screening_dir / outcome
        if not outcome_dir.exists():
            print(f"  No JSONL directory found — skipping {outcome}")
            continue

        jsonl_map = load_jsonl_dir(outcome_dir, temp_str=temp_str, seed=args.seed,
                                   address_mode=args.address_mode,
                                   model_alias=model_alias, cot=args.cot,
                                   political_context=args.political_context)
        if not jsonl_map:
            print(f"  No JSONL files found — skipping {outcome}")
            continue

        print(f"  Found configs: {sorted(jsonl_map.keys())}")

        dist_df, metric_map = build_dist_table(outcome, jsonl_map, gt_dist, ranking)

        # Per-outcome figure
        fig = plot_outcome(outcome, dist_df, metric_map, ranking, dpi=args.dpi)
        for ext in ("pdf", "png"):
            out_path = PLOTS_DIR / f"screening{suffix}_{outcome}.{ext}"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            print(f"  Saved: {out_path}")
        plt.close(fig)

        summary_data.append((outcome, dist_df, metric_map, ranking))

    # Combined summary figure
    if len(summary_data) > 0:
        fig_sum = plot_summary(summary_data, dpi=args.dpi)
        for ext in ("pdf", "png"):
            out_path = PLOTS_DIR / f"screening{suffix}_summary.{ext}"
            fig_sum.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                            facecolor="white", edgecolor="none")
            print(f"\nSaved summary: {out_path}")
        plt.close(fig_sum)

    print("\nDone.")


if __name__ == "__main__":
    main()
