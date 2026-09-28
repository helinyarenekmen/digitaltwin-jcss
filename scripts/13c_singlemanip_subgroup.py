"""
13c_singlemanip_subgroup.py — Subgroup analysis for the single-manipulation
experiment (exp07_kurd_education_singlemanip).

Design: single-factor with 4 conditions (S/P/R/SD), 3 DVs (2 Likert + 1 binary).

For each of six subgroup dimensions (gender, age band, education, urbanization,
ethnic origin, political orientation) plus NUTS-1 region we compute two
paired-within-persona contrasts per DV:

  Context contrast:  Security − Peace    (S − P)
  Framing contrast:  Religious − Self-determination  (R − SD)

Positive S − P means the security context raises the DV; positive R − SD means
the religious frame raises the DV.

Outputs (no existing files overwritten) under
  outputs/experiments/exp07_kurd_education_singlemanip/subgroup_analysis/

  csv/
    subgroup_means.csv       — mean/n per (dim × level × DV × condition)
    contrasts.csv            — paired contrasts + 95% CI per (dim × level × DV × contrast)
  plots/
    forest_S_vs_P__<DV>.{pdf,png}    — one forest per DV (context contrast)
    forest_R_vs_SD__<DV>.{pdf,png}   — one forest per DV (framing contrast)
    forest_all_DVs__S_vs_P.{pdf,png} — small multiples
    forest_all_DVs__R_vs_SD.{pdf,png}
    regional_S_vs_P__<DV>.{pdf,png}  — NUTS-1 choropleth of the context contrast
    interaction_politics__<DV>.{pdf,png} — 4 lines per DV (one per condition)

Usage
-----
    python scripts/13c_singlemanip_subgroup.py
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
EXP_DIR = ROOT / "outputs" / "experiments" / "exp07_kurd_education_singlemanip"
JSONL_LIKERT = EXP_DIR / "exp07_kurd_education_singlemanip_C11_gpt4omini_T08_ben_vs_cot_s0.jsonl"
JSONL_BINARY = EXP_DIR / "exp07_kurd_education_singlemanip_C7_gpt4omini_T0_ben_direct_s0.jsonl"
OUT_DIR = EXP_DIR / "subgroup_analysis"
CSV_DIR = OUT_DIR / "csv"
PLOT_DIR = OUT_DIR / "plots"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


CONDITIONS = ["S", "P", "R", "SD"]
CONTRASTS = [("S", "P"), ("R", "SD")]

DV_INFO = {
    "dv_legitimacy":        dict(label="Movement legitimacy (1-5; ↑=legitimate)"),
    "dv_policy_support":    dict(label="Policy support (1-5; ↑=support)"),
    "dv_behavioral_intent": dict(label="Behavioral intent (1=would attend, 2=would not)"),
}
DV_LIST = list(DV_INFO)


# ---------------------------------------------------------------------------
# Subgroup recoding — same scheme as 12c_experiment_subgroup_analysis.py
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


# ---------------------------------------------------------------------------
def load_experiment() -> pd.DataFrame:
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
def subgroup_means(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    rows = []
    for item in DV_LIST:
        sub = df[df["item_id"] == item]
        g = sub.groupby([dim, "condition"])["y"].agg(["mean", "std", "count"]).reset_index()
        g["item"] = item
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def paired_contrast(df: pd.DataFrame, dim: str, item: str,
                    a: str, b: str) -> pd.DataFrame:
    """For each subgroup level: mean(a) − mean(b), paired within-persona."""
    sub = df[df["item_id"] == item].copy()
    wide = (sub.pivot_table(index=["respondent_id", dim], columns="condition",
                            values="y", aggfunc="first")
                .reset_index())
    if a not in wide or b not in wide:
        return pd.DataFrame()
    wide = wide.dropna(subset=[a, b])
    rows = []
    for level, g in wide.groupby(dim):
        d = (g[a] - g[b]).values
        if len(d) == 0:
            continue
        m = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        ci = 1.96 * se
        rows.append({
            "dim": dim, "level": level, "item": item,
            "contrast": f"{a}-{b}",
            "n_respondents": int(len(d)),
            "mean_a": float(g[a].mean()),
            "mean_b": float(g[b].mean()),
            "contrast_effect": m,
            "se": se,
            "ci_lo": m - ci,
            "ci_hi": m + ci,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def _order(dim, levels_in_data):
    order = DIM_ORDER.get(dim, sorted(levels_in_data))
    rest = [l for l in levels_in_data if l not in order]
    return order + rest


def plot_forest_one_dv(df: pd.DataFrame, item: str, a: str, b: str,
                       out: Path) -> None:
    dims = list(DIM_PRETTY)
    blocks = []
    for dim in dims:
        eff = paired_contrast(df, dim, item, a, b)
        if eff.empty:
            continue
        ord_levels = _order(dim, eff["level"].tolist())
        eff = eff.set_index("level").reindex(ord_levels).reset_index().dropna(subset=["contrast_effect"])
        blocks.append(eff)
    full = pd.concat(blocks, ignore_index=True)

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
        head = (DIM_PRETTY[row["dim"]] + " — "
                if i == 0 or rows[i - 1] is None else "")
        ylabels.append(head + str(row["level"]) + f"  (n={int(row['n_respondents'])})")
        eff, lo, hi = row["contrast_effect"], row["ci_lo"], row["ci_hi"]
        col = "#1f77b4" if (lo > 0 or hi < 0) else "#888"
        ax.plot([lo, hi], [y, y], color=col, linewidth=2)
        ax.scatter([eff], [y], s=42, color=col, edgecolor="black", linewidth=0.6, zorder=3)

    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel(f"Contrast: mean({a}) − mean({b})   [scale units]", fontsize=10)
    ax.set_title(f"Subgroup forest — {a} − {b}\n{DV_INFO[item]['label']}", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def plot_forest_all_dvs(df: pd.DataFrame, a: str, b: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 10), sharey=True)
    dims = list(DIM_PRETTY)

    ref = paired_contrast(df, dims[0], "dv_legitimacy", a, b)
    blocks = []
    for dim in dims:
        eff = paired_contrast(df, dim, "dv_legitimacy", a, b)
        for lvl in _order(dim, eff["level"].tolist()):
            blocks.append((dim, lvl))
    yticks = list(range(len(blocks)))[::-1]

    for ax_idx, item in enumerate(DV_LIST):
        ax = axes[ax_idx]
        all_eff = pd.concat(
            [paired_contrast(df, d, item, a, b).assign(dim=d) for d in dims],
            ignore_index=True,
        )
        all_eff["_key"] = all_eff["dim"].astype(str) + "||" + all_eff["level"].astype(str)
        all_eff = all_eff.set_index("_key")
        for i, (dim, lvl) in enumerate(blocks):
            y = yticks[i]
            key = f"{dim}||{lvl}"
            if key not in all_eff.index:
                continue
            r = all_eff.loc[key]
            eff, lo, hi = r["contrast_effect"], r["ci_lo"], r["ci_hi"]
            col = "#1f77b4" if (lo > 0 or hi < 0) else "#888"
            ax.plot([lo, hi], [y, y], color=col, linewidth=2)
            ax.scatter([eff], [y], s=36, color=col, edgecolor="black",
                       linewidth=0.5, zorder=3)
        ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.set_title(DV_INFO[item]["label"], fontsize=10)
        ax.set_xlabel(f"{a} − {b}", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    yticklabels = []
    prev_dim = None
    for dim, lvl in blocks:
        head = f"{DIM_PRETTY[dim]}: " if dim != prev_dim else ""
        yticklabels.append(head + lvl)
        prev_dim = dim
    axes[0].set_yticks(yticks); axes[0].set_yticklabels(yticklabels, fontsize=8)
    plt.suptitle(f"Subgroup contrast: {a} − {b}   across three DVs",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def plot_interaction_politics(df: pd.DataFrame, item: str, out: Path) -> None:
    """Line plot: political orientation on x, one line per condition."""
    sub = df[df["item_id"] == item]
    pivot = sub.groupby(["politics", "condition"])["y"].mean().unstack("condition")
    order = DIM_ORDER["politics"]
    pivot = pivot.reindex([o for o in order if o in pivot.index])[[c for c in CONDITIONS if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"S": "#C6373A", "P": "#3A82B5", "R": "#7A4E9C", "SD": "#2A9D8F"}
    for cond in pivot.columns:
        ax.plot(pivot.index, pivot[cond], "-o", linewidth=2.2, markersize=8,
                color=colors[cond], label=cond)
    ax.set_ylabel(f"{item} mean", fontsize=10)
    ax.set_title(f"Political orientation × condition — {DV_INFO[item]['label']}",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=True, title="Condition")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def plot_regional_choropleth(df: pd.DataFrame, item: str, a: str, b: str,
                             out: Path) -> None:
    """NUTS-1 choropleth of a − b contrast per region."""
    sub = df[df["item_id"] == item].copy()
    pair = (sub.pivot_table(index=["respondent_id", "region", "nuts_id"],
                            columns="condition", values="y", aggfunc="first")
                .reset_index())
    if a not in pair or b not in pair:
        return
    pair = pair.dropna(subset=[a, b])
    pair["effect"] = pair[a] - pair[b]
    per_region = pair.groupby(["region", "nuts_id"]).agg(
        n=("effect", "count"), effect=("effect", "mean")
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
             legend_kwds={"label": f"{a} − {b}", "shrink": 0.6})
    for _, r in gdf.iterrows():
        if r["geometry"] is None or r["geometry"].is_empty:
            continue
        c = r["geometry"].representative_point()
        if pd.isna(r["effect"]):
            txt = f"{r.get('region', r['NUTS_ID'])}\n(no data)"
        else:
            txt = f"{r['region']}\nΔ={r['effect']:+.2f}\nn={int(r['n'])}"
        ax.annotate(txt, xy=(c.x, c.y), fontsize=6.5, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1))
    ax.set_title(f"Regional contrast {a} − {b} — {item}", fontsize=11, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading experiment data …")
    exp = load_experiment()
    print(f"  {len(exp):,} OK records")
    sub = load_subgroups()
    print(f"  {len(sub):,} TGSS subgroup rows")

    df = exp.merge(sub, on="respondent_id", how="left")
    n_missing = df["gender"].isna().sum()
    print(f"  merged: {len(df):,} rows; missing subgroup meta = {n_missing}")
    df = df.dropna(subset=["gender"])

    # CSV: subgroup means (per dim)
    print("\nCSV exports:")
    means_all = []
    for dim in DIM_PRETTY:
        sm = subgroup_means(df, dim)
        sm.insert(0, "dim", dim)
        sm = sm.rename(columns={dim: "level"})
        means_all.append(sm)
    pd.concat(means_all, ignore_index=True).to_csv(CSV_DIR / "subgroup_means.csv", index=False)
    print(f"  ✓ {(CSV_DIR / 'subgroup_means.csv').relative_to(ROOT)}")

    contrasts_all = []
    for dim in DIM_PRETTY:
        for item in DV_LIST:
            for a, b in CONTRASTS:
                c = paired_contrast(df, dim, item, a, b)
                if not c.empty:
                    contrasts_all.append(c)
    pd.concat(contrasts_all, ignore_index=True).to_csv(CSV_DIR / "contrasts.csv", index=False)
    print(f"  ✓ {(CSV_DIR / 'contrasts.csv').relative_to(ROOT)}")

    print("\nForest plots — per DV:")
    for item in DV_LIST:
        plot_forest_one_dv(df, item, "S", "P", PLOT_DIR / f"forest_S_vs_P__{item}")
        plot_forest_one_dv(df, item, "R", "SD", PLOT_DIR / f"forest_R_vs_SD__{item}")

    print("\nSmall-multiples forests across DVs:")
    plot_forest_all_dvs(df, "S", "P", PLOT_DIR / "forest_all_DVs__S_vs_P")
    plot_forest_all_dvs(df, "R", "SD", PLOT_DIR / "forest_all_DVs__R_vs_SD")

    print("\nInteraction plots (politics × condition):")
    for item in DV_LIST:
        plot_interaction_politics(df, item, PLOT_DIR / f"interaction_politics__{item}")

    print("\nRegional choropleths (S − P):")
    for item in DV_LIST:
        plot_regional_choropleth(df, item, "S", "P",
                                 PLOT_DIR / f"regional_S_vs_P__{item}")

    # Concise text summary of key contrasts
    print("\n" + "=" * 72)
    print("  Summary — S − P and R − SD contrasts by subgroup × DV (★ = CI excludes 0)")
    print("=" * 72)
    for item in DV_LIST:
        print(f"\n  {item}:")
        for a, b in CONTRASTS:
            print(f"    ── {a} − {b} ──")
            ce_all = pd.concat(
                [paired_contrast(df, d, item, a, b).assign(dim=d) for d in DIM_PRETTY],
                ignore_index=True,
            )
            ce_all = ce_all.sort_values("contrast_effect")
            for _, r in ce_all.iterrows():
                sig = "★" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else " "
                print(f"      {sig} {r['dim']:14s} {str(r['level']):26s} "
                      f"Δ={r['contrast_effect']:+.3f} "
                      f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  n={int(r['n_respondents'])}")

    print(f"\nAll outputs under: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
