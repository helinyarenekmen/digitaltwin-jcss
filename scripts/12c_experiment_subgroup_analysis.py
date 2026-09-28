"""
12c_experiment_subgroup_analysis.py — Subgroup heterogeneity in the 2×2
experiment results.

Examines how the Stage-3 context main effect (security vs peace) on each
DV varies across substantively important respondent subgroups available in
the TGSS metadata: gender, age band, education (collapsed), urbanization,
ethnic origin, political orientation, and NUTS-1 region.

Inputs
------
- outputs/experiments/exp02_kurd_education_2x2/
    exp02_kurd_education_2x2_C11_gpt4omini_T08_ben_vs_cot_s0.jsonl
- data/tgss2024_clean.csv   (subgroup metadata)
- data/geo/nuts1_eu_2021.geojson  (Eurostat NUTS-1; for the region map)

Outputs
-------
outputs/experiments/exp02_kurd_education_2x2/subgroup_analysis/
  csv/
    subgroup_means_<dimension>.csv          (per dimension × cell means)
    context_effect_<dimension>.csv          (context main effect + 95% CI)
  plots/
    forest_context_effect_<DV>.{pdf,png}    (per DV; one forest plot per DV)
    forest_context_effect_all_DVs.{pdf,png} (small multiples; key visual)
    heatmap_subgroup_x_cell_<DV>.{pdf,png}  (per DV)
    interaction_<dimension>_<DV>.{pdf,png}  (politics × context, ethnic × context, ...)
    regional_context_effect_<DV>.{pdf,png}  (NUTS-1 choropleth)

Subgroup rules
--------------
- Cells with n<50 inside a (subgroup × condition) are flagged but not
  hidden (forest plot's whiskers expand to reflect imprecision).
- Sign convention: context effect = mean(security) − mean(peace).
  For dv_legitimacy / dv_policy_support / dv_threat (5-pt Likert, 5=agree):
    negative effect = peace lifts the construct.
  For dv_behavioral_intent (1=attend ... 6=counter-protest):
    positive effect = security shifts intent toward counter-protest.

Usage
-----
  python scripts/12c_experiment_subgroup_analysis.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
TGSS_PATH = ROOT / "data" / "tgss2024_clean.csv"
GEO_PATH = ROOT / "data" / "geo" / "nuts1_eu_2021.geojson"

# Parameterized paths (added 2026-06-30 so the same script can analyse V1 / V4).
# Defaults preserve the original Stage-4 behaviour (V1 / exp02 dataset).
import argparse as _argparse_for_paths
_ap = _argparse_for_paths.ArgumentParser(add_help=False)
_ap.add_argument("--jsonl", type=Path, default=None)
_ap.add_argument("--out-dir", type=Path, default=None)
_known, _ = _ap.parse_known_args()

if _known.jsonl is not None:
    JSONL = _known.jsonl.resolve()
    EXP_DIR = JSONL.parent
else:
    EXP_DIR = ROOT / "outputs" / "experiments" / "exp02_kurd_education_2x2"
    JSONL = EXP_DIR / "exp02_kurd_education_2x2_C11_gpt4omini_T08_ben_vs_cot_s0.jsonl"

OUT_DIR = _known.out_dir.resolve() if _known.out_dir is not None else EXP_DIR / "subgroup_analysis"
CSV_DIR = OUT_DIR / "csv"
PLOT_DIR = OUT_DIR / "plots"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Subgroup recoding (same scheme as scripts/appendix_subgroup_analysis.py)
# ---------------------------------------------------------------------------

GENDER_LABEL = {1.0: "Male", 2.0: "Female"}

DEGREE_COLLAPSE = {
    1.0: "Less than high school", 2.0: "Less than high school", 3.0: "Less than high school",
    4.0: "High school",
    5.0: "University or higher", 6.0: "University or higher",
    7.0: "University or higher", 8.0: "University or higher",
}
DEGURBA_LABEL = {1.0: "Rural", 2.0: "Intermediate", 3.0: "Dense urban"}
ETHNIC_COLLAPSE = {1.0: "Turkish", 2.0: "Kurdish"}

NUTS1_LABEL = {
    1:  "İstanbul",            2: "Western Marmara",   3: "Aegean",
    4:  "Eastern Marmara",     5: "Western Anatolia",  6: "Mediterranean",
    7:  "Central Anatolia",    8: "Western Black Sea", 9: "Eastern Black Sea",
    10: "Northeast Anatolia", 11: "East-Central Anatolia", 12: "Southeast Anatolia",
}
NUTS1_TO_EUROSTAT = {i: f"TR{c}" for i, c in enumerate(
    ["1","2","3","4","5","6","7","8","9","A","B","C"], start=1)}


def age_band(a):
    if pd.isna(a): return "Not reported"
    if a < 30: return "18-29"
    if a < 45: return "30-44"
    if a < 60: return "45-59"
    return "60+"


def politics_band(v):
    if pd.isna(v): return "Not reported"
    if v <= 3: return "Left (0-3)"
    if v <= 6: return "Center (4-6)"
    return "Right (7-10)"


DIM_ORDER = {
    "gender":        ["Female", "Male"],
    "age_band":      ["18-29", "30-44", "45-59", "60+"],
    "education":     ["Less than high school", "High school", "University or higher"],
    "urbanization":  ["Rural", "Intermediate", "Dense urban"],
    "ethnic_origin": ["Turkish", "Kurdish", "Other / Not reported"],
    "politics":      ["Left (0-3)", "Center (4-6)", "Right (7-10)", "Not reported"],
}

DIM_PRETTY = {
    "gender":        "Gender",
    "age_band":      "Age",
    "education":     "Education",
    "urbanization":  "Urbanization",
    "ethnic_origin": "Ethnic origin",
    "politics":      "Political orientation",
}

DV_LABEL = {
    "dv_legitimacy":        "Movement legitimacy\n(5-pt, ↑=more legitimate)",
    "dv_threat":            "Threat perception\n(5-pt, ↑=more threatening)",
    "dv_policy_support":    "Policy support\n(5-pt, ↑=more support)",
    "dv_behavioral_intent": "Behavioral intent\n(6-cat, ↓=attend, ↑=counter)",
}
DV_LIST = list(DV_LABEL.keys())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_experiment() -> pd.DataFrame:
    rows = []
    for line in JSONL.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("parse_status") != "ok" or r.get("predicted_value") is None:
            continue
        rows.append({
            "respondent_id": r["respondent_id"],
            "condition":     r["condition"],
            "factor_a":      r["factor_a"],     # G / B
            "factor_b":      r["factor_b"],     # E / K
            "item_id":       r["item_id"],
            "y":             int(r["predicted_value"]),
        })
    return pd.DataFrame(rows)


def load_subgroups() -> pd.DataFrame:
    df = pd.read_csv(TGSS_PATH, encoding="utf-8",
                     usecols=["id", "age", "gender", "degree", "degurba",
                              "eidfinal", "pidleftright", "nuts1"])
    df["respondent_id"] = df["id"].astype(int).apply(lambda x: f"TGSS_{x:04d}")
    df["gender"]        = df["gender"].map(GENDER_LABEL).fillna("Not reported")
    df["age_band"]      = df["age"].apply(age_band)
    df["education"]     = df["degree"].map(DEGREE_COLLAPSE).fillna("Not reported")
    df["urbanization"]  = df["degurba"].map(DEGURBA_LABEL).fillna("Not reported")
    df["ethnic_origin"] = df["eidfinal"].map(ETHNIC_COLLAPSE).fillna("Other / Not reported")
    df["politics"]      = df["pidleftright"].apply(politics_band)
    df["region"]        = df["nuts1"].astype(int).map(NUTS1_LABEL)
    df["nuts_id"]       = df["nuts1"].astype(int).map(NUTS1_TO_EUROSTAT)
    return df[["respondent_id", "gender", "age_band", "education", "urbanization",
               "ethnic_origin", "politics", "region", "nuts_id"]]


# ---------------------------------------------------------------------------
# Subgroup × condition table  (means + SE per cell)
# ---------------------------------------------------------------------------
def subgroup_means(df: pd.DataFrame, dim: str, item: str) -> pd.DataFrame:
    sub = df[df["item_id"] == item]
    g = sub.groupby([dim, "condition"])["y"].agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    return g


def context_effect_table(df: pd.DataFrame, dim: str, item: str) -> pd.DataFrame:
    """Compute mean(security) − mean(peace) per subgroup level with 95 % CI."""
    sub = df[df["item_id"] == item].copy()
    sub["context"] = sub["factor_a"]  # 'G' or 'B'
    rows = []
    for level, g in sub.groupby(dim):
        sec = g.loc[g["context"] == "G", "y"].values
        peace = g.loc[g["context"] == "B", "y"].values
        if len(sec) == 0 or len(peace) == 0:
            continue
        m_s, m_p = sec.mean(), peace.mean()
        diff = m_s - m_p
        # SE of the difference (Welch-style; obs are paired only loosely
        # because each persona contributes both — within-subject. We use the
        # unpooled SD of the difference at the persona level.)
        # Build paired differences per respondent (averaged across framings).
        pair = (
            g.groupby(["respondent_id", "context"])["y"]
              .mean()
              .unstack("context")
              .dropna()
        )
        if pair.empty or "G" not in pair or "B" not in pair:
            continue
        d = (pair["G"] - pair["B"]).values
        n = len(d)
        sd = d.std(ddof=1) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        ci = 1.96 * se if se == se else np.nan
        rows.append({
            "level":         level,
            "n_respondents": int(n),
            "mean_security": m_s,
            "mean_peace":    m_p,
            "context_effect": diff,
            "se":             se,
            "ci_lo":          diff - ci,
            "ci_hi":          diff + ci,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _order(dim, levels_in_data):
    order = DIM_ORDER.get(dim, sorted(levels_in_data))
    rest = [l for l in levels_in_data if l not in order]
    return order + rest


def plot_forest_one_dv(df: pd.DataFrame, item: str, out: Path) -> None:
    """Forest plot — context effect on one DV across all subgroup dimensions."""
    dims = list(DIM_PRETTY.keys())
    blocks = []
    for dim in dims:
        eff = context_effect_table(df, dim, item)
        if eff.empty:
            continue
        ord_levels = _order(dim, eff["level"].tolist())
        eff = eff.set_index("level").reindex(ord_levels).reset_index().dropna(subset=["context_effect"])
        eff["dim"] = dim
        blocks.append(eff)
    full = pd.concat(blocks, ignore_index=True)
    # Insert a blank row between dimensions for visual spacing.
    rows = []
    prev_dim = None
    for _, r in full.iterrows():
        if prev_dim is not None and r["dim"] != prev_dim:
            rows.append(None)
        rows.append(r.to_dict())
        prev_dim = r["dim"]

    fig, ax = plt.subplots(figsize=(9, 0.30 * len(rows) + 2))
    yticks, ylabels = [], []
    for i, row in enumerate(rows):
        if row is None:
            continue
        y = -i
        yticks.append(y)
        lbl = (DIM_PRETTY[row["dim"]] + " — " if i == 0 or rows[i-1] is None else "") + str(row["level"]) + f"  (n={int(row['n_respondents'])})"
        ylabels.append(lbl)
        eff, lo, hi = row["context_effect"], row["ci_lo"], row["ci_hi"]
        col = "#1f77b4" if (lo > 0 or hi < 0) else "#888"
        ax.plot([lo, hi], [y, y], color=col, linewidth=2)
        ax.scatter([eff], [y], s=42, color=col, edgecolor="black", linewidth=0.6, zorder=3)

    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Context effect: mean(security) − mean(peace)  [Likert units]", fontsize=10)
    ax.set_title(f"Subgroup context-effect forest plot\n{DV_LABEL[item]}", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def plot_forest_all_dvs(df: pd.DataFrame, out: Path) -> None:
    """Small multiples — context effect across subgroup dimensions, one panel per DV."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 9), sharey=True)
    dims = list(DIM_PRETTY.keys())

    # Build a stable row index across panels.
    blocks = []
    for dim in dims:
        eff = context_effect_table(df, dim, "dv_legitimacy")  # any item, just for ordering
        ord_levels = _order(dim, eff["level"].tolist())
        for lvl in ord_levels:
            blocks.append((dim, lvl))
    yticks = list(range(len(blocks)))[::-1]

    for ax_idx, item in enumerate(DV_LIST):
        ax = axes[ax_idx]
        # Build effects table for this item; reindex against the canonical block order.
        all_eff = pd.concat([context_effect_table(df, d, item).assign(dim=d) for d in dims],
                            ignore_index=True)
        all_eff["_key"] = all_eff["dim"].astype(str) + "||" + all_eff["level"].astype(str)
        all_eff = all_eff.set_index("_key")
        for i, (dim, lvl) in enumerate(blocks):
            y = yticks[i]
            key = f"{dim}||{lvl}"
            if key not in all_eff.index:
                continue
            r = all_eff.loc[key]
            eff, lo, hi = r["context_effect"], r["ci_lo"], r["ci_hi"]
            col = "#1f77b4" if (lo > 0 or hi < 0) else "#888"
            ax.plot([lo, hi], [y, y], color=col, linewidth=2)
            ax.scatter([eff], [y], s=36, color=col, edgecolor="black", linewidth=0.5, zorder=3)
        ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.set_title(DV_LABEL[item], fontsize=10)
        ax.set_xlabel("Sec − Peace", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    # y-axis labels with dimension headers
    yticklabels = []
    prev_dim = None
    for (dim, lvl) in blocks:
        head = f"{DIM_PRETTY[dim]}: " if dim != prev_dim else ""
        yticklabels.append(head + lvl)
        prev_dim = dim
    axes[0].set_yticks(yticks); axes[0].set_yticklabels(yticklabels, fontsize=8)
    plt.suptitle("Context effect (security − peace) by subgroup × DV",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def plot_heatmap_subgroup_cell(df: pd.DataFrame, item: str, dim: str, out: Path) -> None:
    """Heatmap of subgroup × cell means for one DV."""
    sub = df[df["item_id"] == item]
    grand_mean = sub["y"].mean()
    pivot = (sub.groupby([dim, "condition"])["y"].mean()
                .unstack("condition")
                .reindex(_order(dim, sub[dim].unique()))
                [["GE", "GK", "BE", "BK"]])
    ns = (sub.groupby([dim, "condition"])["y"].count()
              .unstack("condition")
              .reindex(pivot.index)[["GE", "GK", "BE", "BK"]])

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(pivot) + 2))
    vmax = float(np.nanmax(np.abs(pivot.values - grand_mean)))
    norm = TwoSlopeNorm(vmin=grand_mean - vmax, vcenter=grand_mean, vmax=grand_mean + vmax)
    im = ax.imshow(pivot.values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["GE\nsec×rights", "GK\nsec×self-det",
                        "BE\npeace×rights", "BK\npeace×self-det"], fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([f"{lvl}  (n={int(ns.loc[lvl].iloc[0])})"
                        for lvl in pivot.index], fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v - grand_mean) > vmax * 0.6 else "black")
    ax.set_title(f"{DIM_PRETTY[dim]} × condition — mean of {item}\n(grand mean = {grand_mean:.2f})",
                 fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.7, label=f"{item} mean (centered)")
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def plot_interaction(df: pd.DataFrame, dim: str, item: str, out: Path) -> None:
    """Line-plot: x = context (G,B); one line per subgroup level."""
    sub = df[df["item_id"] == item]
    pivot = sub.groupby([dim, "factor_a"])["y"].mean().unstack("factor_a")
    pivot = pivot.reindex(_order(dim, sub[dim].unique()))
    if "G" not in pivot or "B" not in pivot:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for lvl, row in pivot.iterrows():
        ax.plot(["Security", "Peace"], [row["G"], row["B"]],
                "-o", linewidth=2, markersize=7, label=str(lvl))
    ax.set_ylabel(f"{item} mean", fontsize=10)
    ax.set_title(f"{DIM_PRETTY[dim]} × context interaction — {item}",
                 fontsize=11)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def plot_regional_choropleth(df: pd.DataFrame, item: str, out: Path) -> None:
    """NUTS-1 choropleth — context effect on item by region."""
    sub = df[df["item_id"] == item].copy()
    pair = (sub.groupby(["respondent_id", "factor_a", "region", "nuts_id"])["y"]
                 .mean().reset_index())
    paired = pair.pivot_table(index=["respondent_id", "region", "nuts_id"],
                              columns="factor_a", values="y").reset_index()
    paired["effect"] = paired["G"] - paired["B"]
    per_region = paired.groupby(["region", "nuts_id"]).agg(
        n=("effect", "count"),
        effect=("effect", "mean"),
        se=("effect", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
    ).reset_index()

    gdf = gpd.read_file(GEO_PATH)
    gdf = gdf[gdf["CNTR_CODE"] == "TR"][["NUTS_ID", "geometry"]]
    gdf = gdf.merge(per_region, left_on="NUTS_ID", right_on="nuts_id", how="left")

    fig, ax = plt.subplots(figsize=(13, 6.5))
    vlim = float(np.nanmax(np.abs(gdf["effect"].values)))
    gdf.plot(column="effect", cmap="RdBu_r",
             norm=TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim),
             ax=ax, legend=True, edgecolor="black", linewidth=0.6,
             missing_kwds={"color": "#cccccc"},
             legend_kwds={"label": "context effect (Sec − Peace)", "shrink": 0.6})
    for _, r in gdf.iterrows():
        if r["geometry"] is None or r["geometry"].is_empty: continue
        c = r["geometry"].representative_point()
        if pd.isna(r["effect"]):
            txt = f"{r['region']}\n(no data)" if "region" in r and not pd.isna(r["region"]) else r["NUTS_ID"]
        else:
            txt = f"{r['region']}\nΔ={r['effect']:+.2f}\nn={int(r['n'])}"
        ax.annotate(txt, xy=(c.x, c.y), fontsize=6.5, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1))
    ax.set_title(f"Regional context effect on {item}\n(positive = security raises score; negative = peace raises score)",
                 fontsize=11, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading experiment data from {JSONL} ...")
    exp = load_experiment()
    print(f"  {len(exp):,} OK records")

    sub = load_subgroups()
    print(f"  {len(sub):,} TGSS subgroup rows")

    df = exp.merge(sub, on="respondent_id", how="left")
    n_missing = df["gender"].isna().sum()
    print(f"  merge: {len(df):,} rows; missing subgroup = {n_missing}")

    df = df.dropna(subset=["gender"])  # drop rows without subgroup meta

    # ---------- CSV: per-dimension means + context effects ----------
    print("\nCSV exports:")
    for dim in DIM_PRETTY:
        for item in DV_LIST:
            sm = subgroup_means(df, dim, item)
            sm.insert(0, "item", item)
            ce = context_effect_table(df, dim, item)
            ce.insert(0, "item", item)
            sm.to_csv(CSV_DIR / f"subgroup_means_{dim}__{item}.csv", index=False)
            ce.to_csv(CSV_DIR / f"context_effect_{dim}__{item}.csv", index=False)
    # Aggregate context-effect tables
    agg = []
    for dim in DIM_PRETTY:
        for item in DV_LIST:
            ce = context_effect_table(df, dim, item)
            ce.insert(0, "dim", dim); ce.insert(0, "item", item)
            agg.append(ce)
    agg_df = pd.concat(agg, ignore_index=True)
    agg_df.to_csv(CSV_DIR / "context_effect_all.csv", index=False)
    print(f"  ✓ {(CSV_DIR / 'context_effect_all.csv').relative_to(ROOT)}")

    # ---------- Forest plots ----------
    print("\nForest plots:")
    for item in DV_LIST:
        plot_forest_one_dv(df, item, PLOT_DIR / f"forest_context_effect_{item}")
    plot_forest_all_dvs(df, PLOT_DIR / "forest_context_effect_all_DVs")

    # ---------- Heatmaps ----------
    print("\nHeatmaps:")
    for item in DV_LIST:
        for dim in ["politics", "ethnic_origin", "education", "age_band"]:
            plot_heatmap_subgroup_cell(df, item, dim,
                                       PLOT_DIR / f"heatmap_{dim}_x_cell__{item}")

    # ---------- Interaction plots ----------
    print("\nInteraction plots:")
    for item in DV_LIST:
        for dim in ["politics", "ethnic_origin"]:
            plot_interaction(df, dim, item,
                             PLOT_DIR / f"interaction_{dim}__{item}")

    # ---------- Regional choropleths ----------
    print("\nRegional choropleths:")
    for item in DV_LIST:
        plot_regional_choropleth(df, item,
                                 PLOT_DIR / f"regional_context_effect__{item}")

    # ---------- Concise text summary ----------
    print("\n" + "=" * 72)
    print("  Summary — context effect (Sec − Peace) by subgroup × DV")
    print("=" * 72)
    for item in DV_LIST:
        print(f"\n  {item}:")
        ce_all = pd.concat(
            [context_effect_table(df, d, item).assign(dim=d) for d in DIM_PRETTY],
            ignore_index=True,
        )
        ce_all = ce_all.sort_values("context_effect")
        for _, r in ce_all.iterrows():
            sig = "★" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else " "
            print(f"    {sig} {r['dim']:14s} {str(r['level']):28s} "
                  f"Δ={r['context_effect']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
                  f"n={int(r['n_respondents'])}")

    print(f"\nAll outputs under: {OUT_DIR.relative_to(ROOT)}")

    # ---------- Consolidate 49 per-dim × item CSVs into a single Excel ----
    print("\n" + "=" * 72)
    print("  Consolidating per-(dim × item) CSVs into subgroup_analysis.xlsx …")
    print("=" * 72)
    import subprocess, sys
    helper = Path(__file__).resolve().parent / "_helpers" / "consolidate_subgroup_csvs.py"
    subprocess.run([sys.executable, str(helper)], check=True)


if __name__ == "__main__":
    main()
