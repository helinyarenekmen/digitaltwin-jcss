"""
context_subgroup.py — Subgroup analysis for the S-vs-P context experiment
family (exp08 / exp10 / exp11).

Replaces 14c_context_subgroup.py (exp08) and 17d_exp10_subgroup.py (exp10).
The Kurdish subgroup uses the broader definition `kurd == 1 OR lankurd == 1`
matching the R-code's `kurd_all` variable.

Design: 2 conditions, 3 DVs. Because there is only one contrast (S − P), the
sub-group story is simpler than in the 4-condition experiment.

For each subgroup dimension (gender, age band, education, urbanization,
ethnic origin, political orientation, NUTS-1 region) we compute:

  * the paired within-persona effect  Δ = mean(S) − mean(P)
  * the 95 % CI on that effect
  * the baseline mean (average across S and P) — useful for spotting
    ceiling / floor effects that shrink observable Δ

Outputs (under <exp-dir>/subgroup_analysis/):

  csv/
    subgroup_means.csv     (per dim × level × DV × condition + baseline)
    contrasts.csv          (paired S − P + 95 % CI per dim × level × DV)
  plots/
    forest_all_DVs.{pdf,png}       ⭐ paper-headline small multiples
    forest_S_vs_P__<DV>.{pdf,png}  (one per DV)
    interaction_politics__<DV>.{pdf,png}
    interaction_ethnic__<DV>.{pdf,png}
    regional_S_vs_P__<DV>.{pdf,png}  (NUTS-1 choropleth)

Usage
-----
    python scripts/context_subgroup.py --exp exp08_context_only
    python scripts/context_subgroup.py --exp exp10_context_peacev3
    python scripts/context_subgroup.py --exp exp11_context_peacev4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
TGSS_PATH = ROOT / "data" / "tgss2024_clean.csv"
GEO_PATH = ROOT / "data" / "geo" / "nuts1_eu_2021.geojson"

VALID_EXPS = {
    "exp08_context_only",
    "exp10_context_peacev3",
    "exp11_context_peacev4",
}

_p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
_p.add_argument("--exp", required=True, choices=sorted(VALID_EXPS),
                help="Experiment id (folder under outputs/experiments/).")
_args = _p.parse_args()

EXP_ID = _args.exp
EXP_DIR = ROOT / "outputs" / "experiments" / EXP_ID
JSONL_LIKERT = EXP_DIR / f"{EXP_ID}_C7_gemini25flashlite_T08_ben_vs_cot_s0.jsonl"
JSONL_BINARY = EXP_DIR / f"{EXP_ID}_C7_gpt4omini_T0_ben_direct_s0.jsonl"
OUT_DIR = EXP_DIR / "subgroup_analysis"
CSV_DIR = OUT_DIR / "csv"
PLOT_DIR = OUT_DIR / "plots"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


DV_INFO = {
    "dv_legitimacy":        dict(label="Movement legitimacy (1–5)"),
    "dv_policy_support":    dict(label="Policy support (1–5)"),
    "dv_behavioral_intent": dict(label="Behavioral intent (1 = attend, 2 = no)"),
}
DV_LIST = list(DV_INFO)


# ---------------------------------------------------------------------------
# Subgroup recoding (identical scheme to earlier subgroup scripts)
# ---------------------------------------------------------------------------
GENDER_LABEL = {1.0: "Male", 2.0: "Female"}
DEGREE_COLLAPSE = {
    1.0: "Less than high school", 2.0: "Less than high school",
    3.0: "Less than high school", 4.0: "High school",
    5.0: "University or higher", 6.0: "University or higher",
    7.0: "University or higher", 8.0: "University or higher",
}
DEGURBA_LABEL = {1.0: "Rural", 2.0: "Intermediate", 3.0: "Dense urban"}
ETHNIC_COLLAPSE = {1.0: "Turkish", 2.0: "Kurdish"}   # legacy, no longer used


def kurd_all(row: pd.Series) -> str:
    """Kurdish flag (matches R code `kurd_all = kurd | lankurd`).

    `kurd`     = ancestry: "Soyunuzu nasıl tanımlarsınız? ... Kürt" (multi-choice)
    `lankurd`  = language: "Evde en sık konuştuğunuz dil ... Kürtçe"
    In TGSS coding: 1 = flagged (Kurdish), 2 = not flagged.
    """
    is_kurd = (row.get("kurd") == 1.0) or (row.get("lankurd") == 1.0)
    return "Kurdish" if is_kurd else "Non-Kurdish"

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
    "ethnic_origin": ["Non-Kurdish", "Kurdish"],
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
                              "kurd", "lankurd", "pidleftright", "nuts1"])
    df["respondent_id"] = df["id"].astype(int).apply(lambda x: f"TGSS_{x:04d}")
    df["gender"]        = df["gender"].map(GENDER_LABEL).fillna("Not reported")
    df["age_band"]      = df["age"].apply(age_band)
    df["education"]     = df["degree"].map(DEGREE_COLLAPSE).fillna("Not reported")
    df["urbanization"]  = df["degurba"].map(DEGURBA_LABEL).fillna("Not reported")
    df["ethnic_origin"] = df.apply(kurd_all, axis=1)
    df["politics"]      = df["pidleftright"].apply(politics_band)
    df["region"]        = df["nuts1"].astype(int).map(NUTS1_LABEL)
    df["nuts_id"]       = df["nuts1"].astype(int).map(NUTS1_TO_EUROSTAT)
    return df[["respondent_id", "gender", "age_band", "education", "urbanization",
               "ethnic_origin", "politics", "region", "nuts_id"]]


# ---------------------------------------------------------------------------
def paired_contrast(df: pd.DataFrame, dim: str, item: str) -> pd.DataFrame:
    sub = df[df["item_id"] == item].copy()
    wide = (sub.pivot_table(index=["respondent_id", dim], columns="condition",
                            values="y", aggfunc="first")
                .reset_index())
    if "S" not in wide or "P" not in wide:
        return pd.DataFrame()
    wide = wide.dropna(subset=["S", "P"])
    rows = []
    for level, g in wide.groupby(dim):
        d = (g["S"] - g["P"]).values
        if len(d) == 0:
            continue
        m = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        ci = 1.96 * se
        baseline = float(((g["S"] + g["P"]) / 2.0).mean())
        rows.append({
            "dim": dim, "level": level, "item": item, "contrast": "S-P",
            "n_respondents": int(len(d)),
            "mean_S": float(g["S"].mean()),
            "mean_P": float(g["P"].mean()),
            "baseline_mean": baseline,
            "contrast_effect": m,
            "se": se, "ci_lo": m - ci, "ci_hi": m + ci,
        })
    return pd.DataFrame(rows)


def subgroup_means_table(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    rows = []
    for item in DV_LIST:
        sub = df[df["item_id"] == item]
        g = (sub.groupby([dim, "condition"])["y"]
                .agg(["mean", "std", "count"]).reset_index())
        g["item"] = item
        g = g.rename(columns={dim: "level"})
        g["dim"] = dim
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
def _order(dim, levels_in_data):
    order = DIM_ORDER.get(dim, sorted(levels_in_data))
    return order + [l for l in levels_in_data if l not in order]


def plot_forest_one_dv(df: pd.DataFrame, item: str, out: Path) -> None:
    dims = list(DIM_PRETTY)
    blocks = []
    for dim in dims:
        eff = paired_contrast(df, dim, item)
        if eff.empty:
            continue
        ord_levels = _order(dim, eff["level"].tolist())
        eff = (eff.set_index("level").reindex(ord_levels).reset_index()
                  .dropna(subset=["contrast_effect"]))
        blocks.append(eff)
    full = pd.concat(blocks, ignore_index=True)
    rows = []
    prev_dim = None
    for _, r in full.iterrows():
        if prev_dim is not None and r["dim"] != prev_dim:
            rows.append(None)
        rows.append(r.to_dict())
        prev_dim = r["dim"]

    fig, ax = plt.subplots(figsize=(9.5, 0.30 * len(rows) + 2))
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
        ax.scatter([eff], [y], s=42, color=col,
                   edgecolor="black", linewidth=0.6, zorder=3)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Context effect: mean(Security) − mean(Peace)   [scale units]",
                  fontsize=10)
    ax.set_title(f"Subgroup forest — {DV_INFO[item]['label']}", fontsize=11,
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.text(0.01, -0.02,
            "Negative ⇒ peace context raises the DV; positive ⇒ security context raises it.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#555")
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def plot_forest_all_dvs(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(DV_LIST), figsize=(17, 10), sharey=True)
    dims = list(DIM_PRETTY)

    blocks = []
    for dim in dims:
        eff = paired_contrast(df, dim, DV_LIST[0])
        for lvl in _order(dim, eff["level"].tolist()):
            blocks.append((dim, lvl))
    yticks = list(range(len(blocks)))[::-1]

    for ax_idx, item in enumerate(DV_LIST):
        ax = axes[ax_idx]
        all_eff = pd.concat(
            [paired_contrast(df, d, item).assign(dim=d) for d in dims],
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
        ax.set_title(DV_INFO[item]["label"], fontsize=10.5)
        ax.set_xlabel("S − P", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    yticklabels = []
    prev_dim = None
    for dim, lvl in blocks:
        head = f"{DIM_PRETTY[dim]}: " if dim != prev_dim else ""
        yticklabels.append(head + lvl)
        prev_dim = dim
    axes[0].set_yticks(yticks); axes[0].set_yticklabels(yticklabels, fontsize=8)
    plt.suptitle(f"{EXP_ID} — S − P context effect across subgroups × DVs",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def plot_interaction(df: pd.DataFrame, dim: str, item: str, out: Path) -> None:
    sub = df[df["item_id"] == item]
    pivot = sub.groupby([dim, "condition"])["y"].mean().unstack("condition")
    order = DIM_ORDER.get(dim, sorted(pivot.index))
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    if "S" not in pivot or "P" not in pivot:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for lvl, row in pivot.iterrows():
        ax.plot(["Security", "Peace"], [row["S"], row["P"]],
                "-o", linewidth=2.2, markersize=8, label=str(lvl))
    ax.set_ylabel(f"{item} mean", fontsize=10)
    ax.set_title(f"{DIM_PRETTY[dim]} × context — {DV_INFO[item]['label']}",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=True, title=DIM_PRETTY[dim])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


def plot_regional_choropleth(df: pd.DataFrame, item: str, out: Path) -> None:
    sub = df[df["item_id"] == item].copy()
    pair = (sub.pivot_table(index=["respondent_id", "region", "nuts_id"],
                            columns="condition", values="y", aggfunc="first")
                .reset_index())
    if "S" not in pair or "P" not in pair:
        return
    pair = pair.dropna(subset=["S", "P"])
    pair["effect"] = pair["S"] - pair["P"]
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
             legend_kwds={"label": "S − P", "shrink": 0.6})
    for _, r in gdf.iterrows():
        if r["geometry"] is None or r["geometry"].is_empty:
            continue
        c = r["geometry"].representative_point()
        if pd.isna(r["effect"]):
            txt = f"{r.get('region', r['NUTS_ID'])}\n(no data)"
        else:
            txt = f"{r['region']}\nΔ={r['effect']:+.2f}\nn={int(r['n'])}"
        ax.annotate(txt, xy=(c.x, c.y), fontsize=6.5, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.6, pad=1))
    ax.set_title(f"Regional S − P context effect — {DV_INFO[item]['label']}",
                 fontsize=11, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.with_suffix('.pdf').relative_to(ROOT)}")


# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading experiment + subgroup meta …")
    exp = load_experiment()
    sub = load_subgroups()
    df = exp.merge(sub, on="respondent_id", how="left").dropna(subset=["gender"])
    print(f"  {len(df):,} rows after merge")

    # CSV outputs
    print("\nCSV exports:")
    means_all = pd.concat([subgroup_means_table(df, d) for d in DIM_PRETTY],
                          ignore_index=True)
    means_all.to_csv(CSV_DIR / "subgroup_means.csv", index=False)
    print(f"  ✓ {(CSV_DIR / 'subgroup_means.csv').relative_to(ROOT)}")

    contrasts = pd.concat(
        [paired_contrast(df, d, it) for d in DIM_PRETTY for it in DV_LIST],
        ignore_index=True,
    )
    contrasts.to_csv(CSV_DIR / "contrasts.csv", index=False)
    print(f"  ✓ {(CSV_DIR / 'contrasts.csv').relative_to(ROOT)}")

    # Forests
    print("\nForest plots:")
    plot_forest_all_dvs(df, PLOT_DIR / "forest_all_DVs")
    for item in DV_LIST:
        plot_forest_one_dv(df, item, PLOT_DIR / f"forest_S_vs_P__{item}")

    # Interactions
    print("\nInteraction plots (politics × condition, ethnic × condition):")
    for item in DV_LIST:
        plot_interaction(df, "politics", item,
                         PLOT_DIR / f"interaction_politics__{item}")
        plot_interaction(df, "ethnic_origin", item,
                         PLOT_DIR / f"interaction_ethnic__{item}")

    # Choropleths
    print("\nRegional choropleths:")
    for item in DV_LIST:
        plot_regional_choropleth(df, item,
                                 PLOT_DIR / f"regional_S_vs_P__{item}")

    # Concise text summary
    print("\n" + "=" * 74)
    print("  Summary — S − P by subgroup × DV (★ = CI excludes 0)")
    print("=" * 74)
    for item in DV_LIST:
        print(f"\n  {item}:")
        ce_all = pd.concat(
            [paired_contrast(df, d, item).assign(dim=d) for d in DIM_PRETTY],
            ignore_index=True,
        ).sort_values("contrast_effect")
        for _, r in ce_all.iterrows():
            sig = "★" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else " "
            print(f"    {sig} {r['dim']:14s} {str(r['level']):26s} "
                  f"Δ={r['contrast_effect']:+.3f} "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  n={int(r['n_respondents'])}")

    print(f"\nAll outputs under: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
