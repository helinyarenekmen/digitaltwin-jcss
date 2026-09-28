"""
12c_b_consolidate_subgroup_csvs.py — Roll the 49 per-(dim, item) CSV files
produced by 12c_experiment_subgroup_analysis.py into a single, navigable
Excel workbook with one sheet per substantive view.

Reads
-----
  outputs/experiments/exp02_kurd_education_2x2/subgroup_analysis/csv/
    subgroup_means_<dim>__<item>.csv     (24 files)
    context_effect_<dim>__<item>.csv     (24 files)
    context_effect_all.csv               (1 file)

Writes
------
  outputs/experiments/exp02_kurd_education_2x2/subgroup_analysis/
    subgroup_analysis.xlsx
        00_README              : sheet directory + column legend
        01_overview_all        : master long table, every (dim × level × DV) effect
        02_baselines           : per-subgroup mean across all four cells (for each DV)
        10_effects_gender      : context effect for each DV by gender
        11_effects_age         : ...by age band
        12_effects_education   : ...by education
        13_effects_urbanization: ...by urbanization
        14_effects_ethnic      : ...by ethnic origin
        15_effects_politics    : ...by political orientation
        16_effects_region      : ...by NUTS-1 region
        20_means_gender        : per-cell means × DV by gender (long)
        21_means_age           : ...
        … (one means sheet per dimension)

  Then archives the original per-(dim, item) CSVs into
  outputs/experiments/exp02_kurd_education_2x2/subgroup_analysis/csv_archive/
  (kept on disk for reproducibility but moved out of the working folder).

Usage
-----
  python scripts/12c_b_consolidate_subgroup_csvs.py
  python scripts/12c_b_consolidate_subgroup_csvs.py --no-archive  # keep CSVs in place
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]   # scripts/_helpers/*.py → project root
EXP_DIR = ROOT / "outputs" / "experiments" / "exp02_kurd_education_2x2"
SG_DIR = EXP_DIR / "subgroup_analysis"
CSV_DIR = SG_DIR / "csv"
JSONL = EXP_DIR / "exp02_kurd_education_2x2_C11_gpt4omini_T08_ben_vs_cot_s0.jsonl"
XLSX = SG_DIR / "subgroup_analysis.xlsx"
ARCHIVE = SG_DIR / "csv_archive"

DIMS = ["gender", "age_band", "education", "urbanization",
        "ethnic_origin", "politics", "region"]
DIM_PRETTY = {
    "gender":        "Gender",
    "age_band":      "Age band",
    "education":     "Education",
    "urbanization":  "Urbanization",
    "ethnic_origin": "Ethnic origin",
    "politics":      "Political orientation",
    "region":        "NUTS-1 region",
}
DV_LIST = ["dv_legitimacy", "dv_threat", "dv_policy_support", "dv_behavioral_intent"]
DV_PRETTY = {
    "dv_legitimacy":        "7a Movement legitimacy (1-5)",
    "dv_threat":            "7b Threat perception (1-5)",
    "dv_policy_support":    "7c Policy support (1-5)",
    "dv_behavioral_intent": "7d Behavioral intent (1-6; ↓=attend)",
}

# Sheet name → (left-to-right column order, friendly column rename)
EFFECT_COLS = ["item", "dim", "level", "n_respondents",
               "mean_security", "mean_peace",
               "context_effect", "se", "ci_lo", "ci_hi"]
MEANS_COLS = ["item", "dim", "level", "condition", "mean", "std", "count", "se"]


# ---------------------------------------------------------------------------
def _read_concat(pattern: str, dim: str) -> pd.DataFrame:
    files = sorted(CSV_DIR.glob(pattern.format(dim=dim)))
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        df = pd.read_csv(f)
        df["dim"] = dim
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def collect_effects(dim: str) -> pd.DataFrame:
    df = _read_concat("context_effect_{dim}__*.csv", dim)
    if df.empty:
        return df
    df = df.rename(columns={"level": "level"})
    keep = [c for c in EFFECT_COLS if c in df.columns]
    df = df[keep].copy()
    df["item_pretty"] = df["item"].map(DV_PRETTY)
    return df


def collect_means(dim: str) -> pd.DataFrame:
    df = _read_concat("subgroup_means_{dim}__*.csv", dim)
    if df.empty:
        return df
    # rename level column if it differs across exports
    if dim in df.columns and "level" not in df.columns:
        df = df.rename(columns={dim: "level"})
    keep = [c for c in MEANS_COLS if c in df.columns]
    df = df[keep].copy()
    df["item_pretty"] = df["item"].map(DV_PRETTY)
    return df


def overall_long() -> pd.DataFrame:
    p = CSV_DIR / "context_effect_all.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    keep = [c for c in EFFECT_COLS if c in df.columns]
    df = df[keep].copy()
    df["item_pretty"] = df["item"].map(DV_PRETTY)
    return df


def compute_baselines() -> pd.DataFrame:
    """Per-subgroup mean across all 4 cells (the 'level' table).

    Reproduces the addendum table in exp_log.md without re-reading the
    JSONL through pandas every time. Lightweight subset.
    """
    # We need the TGSS subgroup info; reuse the merged frame the
    # subgroup-analysis script produced indirectly by reading the JSONL.
    rows = []
    for line in JSONL.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("parse_status") != "ok" or r.get("predicted_value") is None:
            continue
        rows.append({"rid": r["respondent_id"], "item": r["item_id"],
                     "y": int(r["predicted_value"])})
    exp = pd.DataFrame(rows)

    tgss = pd.read_csv(ROOT / "data" / "tgss2024_clean.csv",
                       encoding="utf-8",
                       usecols=["id", "age", "gender", "degree", "degurba",
                                "eidfinal", "pidleftright", "nuts1"])
    tgss["rid"] = tgss["id"].astype(int).apply(lambda x: f"TGSS_{x:04d}")

    def age_band(a):
        if pd.isna(a): return "Not reported"
        if a < 30: return "18-29"
        if a < 45: return "30-44"
        if a < 60: return "45-59"
        return "60+"

    def pol(v):
        if pd.isna(v): return "Not reported"
        if v <= 3: return "Left (0-3)"
        if v <= 6: return "Center (4-6)"
        return "Right (7-10)"

    EDU = {1.0:"Less than high school",2.0:"Less than high school",
           3.0:"Less than high school",4.0:"High school",
           5.0:"University or higher",6.0:"University or higher",
           7.0:"University or higher",8.0:"University or higher"}
    DEGU = {1.0:"Rural",2.0:"Intermediate",3.0:"Dense urban"}
    GEN  = {1.0:"Male",2.0:"Female"}
    ETH  = {1.0:"Turkish",2.0:"Kurdish"}
    NUTS = {1:"İstanbul",2:"Western Marmara",3:"Aegean",4:"Eastern Marmara",
            5:"Western Anatolia",6:"Mediterranean",7:"Central Anatolia",
            8:"Western Black Sea",9:"Eastern Black Sea",10:"Northeast Anatolia",
            11:"East-Central Anatolia",12:"Southeast Anatolia"}

    tgss["gender"]        = tgss["gender"].map(GEN).fillna("Not reported")
    tgss["age_band"]      = tgss["age"].apply(age_band)
    tgss["education"]     = tgss["degree"].map(EDU).fillna("Not reported")
    tgss["urbanization"]  = tgss["degurba"].map(DEGU).fillna("Not reported")
    tgss["ethnic_origin"] = tgss["eidfinal"].map(ETH).fillna("Other / Not reported")
    tgss["politics"]      = tgss["pidleftright"].apply(pol)
    tgss["region"]        = tgss["nuts1"].astype(int).map(NUTS)

    merged = exp.merge(
        tgss[["rid", "gender", "age_band", "education", "urbanization",
              "ethnic_origin", "politics", "region"]],
        on="rid",
    )

    out = []
    for dim in DIMS:
        for item in DV_LIST:
            g = (merged[merged["item"] == item]
                 .groupby(dim)["y"]
                 .agg(["count", "mean", "std", "median"])
                 .reset_index()
                 .rename(columns={dim: "level"}))
            g.insert(0, "dim", dim)
            g.insert(0, "item", item)
            g["item_pretty"] = item.map if False else DV_PRETTY[item]
            out.append(g)
    bl = pd.concat(out, ignore_index=True)
    bl["item_pretty"] = bl["item"].map(DV_PRETTY)
    # n is number of records per subgroup; for paired analyses we use
    # n_respondents from the effects sheet, but the level table is over
    # all 4 cells so count = 4 × n_respondents.
    bl = bl.rename(columns={"count": "n_records", "mean": "baseline_mean",
                            "std": "baseline_sd", "median": "baseline_median"})
    return bl[["item", "item_pretty", "dim", "level", "n_records",
               "baseline_mean", "baseline_sd", "baseline_median"]]


# ---------------------------------------------------------------------------
# Excel writing — header styling consistent across sheets
# ---------------------------------------------------------------------------
def _style(ws, df: pd.DataFrame, freeze: str = "A2") -> None:
    hf = Font(bold=True, color="FFFFFF")
    hb = PatternFill("solid", fgColor="4472C4")
    ce = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for ci, col in enumerate(df.columns, 1):
        c = ws.cell(row=1, column=ci)
        c.font, c.fill, c.alignment = hf, hb, ce
        col_letter = get_column_letter(ci)
        sample = df[col].astype(str).head(80).tolist()
        ml = max([len(str(col))] + [min(len(s), 60) for s in sample])
        ws.column_dimensions[col_letter].width = min(max(ml + 2, 9), 36)
    ws.freeze_panes = freeze


def round_numeric(df: pd.DataFrame, ndigits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(ndigits)
    return out


def build_xlsx() -> None:
    print(f"Reading {CSV_DIR}/ ...")
    overall = round_numeric(overall_long())
    baselines = round_numeric(compute_baselines())

    effects_by_dim = {dim: round_numeric(collect_effects(dim)) for dim in DIMS}
    means_by_dim   = {dim: round_numeric(collect_means(dim))   for dim in DIMS}

    print(f"\nWriting {XLSX.relative_to(ROOT)} ...")
    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:

        # 00_README ----------------------------------------------------
        readme = pd.DataFrame({
            "sheet": [
                "00_README",
                "01_overview_all",
                "02_baselines",
                *[f"1{i}_effects_{d}" for i, d in enumerate(DIMS)],
                *[f"2{i}_means_{d}"   for i, d in enumerate(DIMS)],
            ],
            "purpose": [
                "Directory of sheets + key column legend.",
                "Master long table: every (subgroup × DV) context effect + 95% CI.",
                "Per-subgroup baseline level (mean across all 4 cells) for each DV. Read together with the effects sheets to avoid conflating 'effect = 0' with 'low baseline'.",
                *[f"Context effect (Security − Peace) by {DIM_PRETTY[d]}: paired mean diff with 95% CI per DV." for d in DIMS],
                *[f"Per-cell means (raw scale) by {DIM_PRETTY[d]} × condition × DV." for d in DIMS],
            ],
        })
        legend = pd.DataFrame({
            "column": [
                "item", "item_pretty", "dim", "level",
                "n_respondents", "n_records",
                "mean_security", "mean_peace",
                "context_effect", "se", "ci_lo", "ci_hi",
                "baseline_mean", "baseline_sd", "baseline_median",
                "condition",
            ],
            "meaning": [
                "DV item ID (e.g. dv_legitimacy).",
                "Human-readable DV label (e.g. '7a Movement legitimacy (1-5)').",
                "Subgroup dimension (gender, age_band, education, ...).",
                "Subgroup level value (e.g. 'Female', 'Left (0-3)').",
                "Number of distinct personas contributing to the paired effect.",
                "Number of raw records (≈ 4 × n_respondents for level sheets).",
                "Mean of the DV in security-context cells (G).",
                "Mean of the DV in peace-context cells (B).",
                "Paired mean difference: mean(G) − mean(B).",
                "Standard error of the paired difference.",
                "Lower bound of the 95% CI for the context effect.",
                "Upper bound of the 95% CI for the context effect.",
                "Baseline mean across all four 2x2 cells (no manipulation contrast).",
                "Standard deviation of the baseline (across all 4 cells pooled).",
                "Median of the baseline (across all 4 cells pooled).",
                "2-letter cell code: GE, GK, BE, BK.",
            ],
        })
        readme.to_excel(w, sheet_name="00_README", index=False, startrow=0)
        # Add a small space then legend
        legend.to_excel(w, sheet_name="00_README", index=False, startrow=len(readme) + 3)
        ws = w.sheets["00_README"]
        # Style only the first header row; legend header is the row at startrow.
        _style(ws, readme, freeze="A2")

        # 01_overview_all ----------------------------------------------
        overall.to_excel(w, sheet_name="01_overview_all", index=False)
        _style(w.sheets["01_overview_all"], overall)

        # 02_baselines -------------------------------------------------
        baselines.to_excel(w, sheet_name="02_baselines", index=False)
        _style(w.sheets["02_baselines"], baselines)

        # 10–16 effects_<dim> ------------------------------------------
        for i, d in enumerate(DIMS):
            sheet = f"1{i}_effects_{d}"
            df = effects_by_dim[d]
            df.to_excel(w, sheet_name=sheet, index=False)
            _style(w.sheets[sheet], df)

        # 20–26 means_<dim> --------------------------------------------
        for i, d in enumerate(DIMS):
            sheet = f"2{i}_means_{d}"
            df = means_by_dim[d]
            df.to_excel(w, sheet_name=sheet, index=False)
            _style(w.sheets[sheet], df)

    print(f"  ✓ {XLSX.relative_to(ROOT)}")
    print(f"    Sheets: README + overview_all + baselines + {len(DIMS)} effects + {len(DIMS)} means = {3 + 2 * len(DIMS)} sheets")


def archive_csvs() -> int:
    ARCHIVE.mkdir(exist_ok=True)
    moved = 0
    for f in list(CSV_DIR.glob("subgroup_means_*.csv")) + list(CSV_DIR.glob("context_effect_*.csv")):
        if f.name == "context_effect_all.csv":
            continue  # keep this one in place for the main script's own users
        shutil.move(str(f), str(ARCHIVE / f.name))
        moved += 1
    return moved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-archive", action="store_true",
                    help="Do NOT move per-(dim,item) CSVs into csv_archive/")
    args = ap.parse_args()

    if not CSV_DIR.exists():
        raise SystemExit(f"Expected {CSV_DIR} from 12c_experiment_subgroup_analysis.py")

    build_xlsx()

    if not args.no_archive:
        moved = archive_csvs()
        print(f"\nArchived {moved} per-(dim, item) CSVs into "
              f"{ARCHIVE.relative_to(ROOT)}/  (master Excel is now the working artefact).")
        print("Kept in place: context_effect_all.csv  (master long table; useful for ad-hoc joins).")
    else:
        print("\n--no-archive: per-(dim, item) CSVs left in place.")


if __name__ == "__main__":
    main()
