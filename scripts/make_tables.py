"""
Regenerate paper Tables 1, 2, 5-17 as CSVs in outputs/tables/ from the
archived Excel workbooks and CSVs.

Tables 3-4 come from the variable dictionary and the config module directly.

Usage
-----
    python scripts/make_tables.py             # write all tables
    python scripts/make_tables.py --table 2   # write only Table N
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.ablations import CONFIGS

RESULTS = ROOT / "outputs" / "results"
OUT     = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load(dv: str) -> pd.DataFrame:
    return pd.read_excel(RESULTS / f"{dv}_all_phases.xlsx", sheet_name="All cells")


def _final_pacdemons() -> pd.DataFrame:
    x = _load("pacdemons")
    return x[(x["Config"] == "C6") & (x["Model"] == "gpt-4o-mini") &
             (x["Sampling"] == "direct") & (x["Temperature"] == 0.8) &
             x["Source_CSV"].fillna("").str.contains("politicalcontext")]


def _final_womenwork() -> pd.DataFrame:
    x = _load("womenwork")
    return x[(x["Config"] == "C6") &
             x["Model"].fillna("").str.contains("gpt-5.4-mini") &
             x["Sampling"].fillna("").str.contains("vs") &
             (x["Temperature"] == 0.8) &
             x["Source_CSV"].fillna("").str.contains("politicalcontext", case=False)]


# ---------------------------------------------------------------------------
# Table 1 — Cross-model comparison at seed zero
# ---------------------------------------------------------------------------
def table_1() -> pd.DataFrame:
    """Cross-model comparison at seed zero (paper Table 1).

    Panel A — pacdemons under C6 + direct + T=0.8, seed 0.
    Panel B — womenwork best-config-per-model under verbalized sampling + T=0.8, seed 0.
    """
    # --- Panel A ---
    p = _load("pacdemons")
    a = p[(p["Config"] == "C6") & (p["Sampling"] == "direct") &
          (p["Temperature"] == 0.8) & (p["Seed"] == 0)].copy()
    a["delta_p"] = ((a["pred_mean"] - a["gt_mean"]) * 100).round(1)  # percentage points
    panel_a = a[["Model", "Config", "jsd", "delta_p",
                 "recall_minority", "precision_minority"]].copy()
    panel_a.columns = ["Model", "Config", "JSD", "delta_p_pp", "Rec+", "Prec+"]
    panel_a.insert(0, "Panel", "A")

    # --- Panel B ---
    w = _load("womenwork")
    b = w[(w["Sampling"].fillna("").str.contains("vs")) &
          (w["Temperature"] == 0.8) & (w["Seed"] == 0)].copy()
    panel_b = b[["Model", "Config", "wasserstein", "weighted_kappa_quadratic",
                 "pred_mean"]].copy()
    panel_b.columns = ["Model", "Config", "Wass", "kappa_w", "sim_mean"]
    panel_b.insert(0, "Panel", "B")

    return pd.concat([panel_a, panel_b], ignore_index=True)


# ---------------------------------------------------------------------------
# Table 2 — Final calibrated protocols
# ---------------------------------------------------------------------------
def table_2() -> pd.DataFrame:
    p3 = _final_pacdemons()
    w3 = _final_womenwork()
    return pd.DataFrame({
        "Component":              ["Persona configuration", "Sampling strategy",
                                    "Temperature", "Model", "Address mode",
                                    "Prompt framing", "Positive-class recall",
                                    "JSD", "Wasserstein distance", "kappa_w", "n"],
        "pacdemons (binary)":     ["C6", "Direct answering", 0.8, "GPT-4o-mini",
                                    "First person", "+ ideological background",
                                    round(p3["recall_minority"].mean(), 3),
                                    round(p3["jsd"].mean(), 4), "—", "—",
                                    int(p3["n_matched"].iloc[0])],
        "womenwork (5-pt Likert)":["C6", "Verbalized sampling", 0.8, "GPT-5.4-mini",
                                    "First person", "+ ideological background",
                                    "—", "—",
                                    round(w3["wasserstein"].mean(), 3),
                                    round(w3["weighted_kappa_quadratic"].mean(), 3),
                                    int(w3["n_matched"].iloc[0])],
    })


# ---------------------------------------------------------------------------
# Tables 3, 4 — Persona variables + configurations
# ---------------------------------------------------------------------------
def table_3() -> pd.DataFrame:
    x = pd.read_excel(ROOT / "data" / "derived" / "TGSS2024_Persona_Variables.xlsx",
                      sheet_name="Variables")
    return x[["Variable Group", "Variable Code", "Question"]]


def table_4() -> pd.DataFrame:
    rows = []
    # Only paper configurations C0-C10 (drop C6-holdout-* which live in Appendix G)
    for c in [c for c in CONFIGS if not c.config_id.startswith("C6-holdout")]:
        rows.append({
            "paper_id":    c.config_id,
            "legacy_id":   c.legacy_id,
            "name":        c.name,
            "group_count": len(c.group_aliases),
            "groups":      ";".join(c.group_aliases),
            "description": c.description,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables 5-14 — Stage 1-4 + seed stability metric tables (from Excel workbooks)
# ---------------------------------------------------------------------------
def _phase_slice(dv: str, phase_prefix: str) -> pd.DataFrame:
    x = _load(dv)
    return x[x["Phase"].astype(str).str.startswith(phase_prefix)].reset_index(drop=True)


def table_5_6(): return {
    "Table5_stage1_pacdemons":   _phase_slice("pacdemons", "Phase 1"),
    "Table6_stage1_womenwork":   _phase_slice("womenwork", "Phase 1"),
}


def table_7_9(): return {
    "Table7-9_stage2_pacdemons": _phase_slice("pacdemons", "Phase 2"),
    "Table7-9_stage2_womenwork": _phase_slice("womenwork", "Phase 2"),
}


def _stage_slice(dv: str, stage: str) -> pd.DataFrame:
    x = _load(dv)
    return x[x["Stage"].astype(str).str.contains(stage, na=False)].reset_index(drop=True)


def table_10_11(): return {
    "Table10-11_stage3_pacdemons": _stage_slice("pacdemons", "Stage 3"),
    "Table10-11_stage3_womenwork": _stage_slice("womenwork", "Stage 3"),
}
def table_12_13(): return {
    "Table12-13_stage4_pacdemons": _stage_slice("pacdemons", "Stage 4"),
    "Table12-13_stage4_womenwork": _stage_slice("womenwork", "Stage 4"),
}
def table_14(): return {
    "Table14_stage5_pacdemons": _stage_slice("pacdemons", "Stage 5"),
    "Table14_stage5_womenwork": _stage_slice("womenwork", "Stage 5"),
}


# ---------------------------------------------------------------------------
# Table 15 — Seed-pair agreement (exact, ±1, unweighted kappa, weighted kappa)
# ---------------------------------------------------------------------------
def table_15() -> pd.DataFrame:
    """Recompute respondent-level seed-pair agreement from the archived JSONL.

    Reads the three seed files (s0, s1, s2) for the paper's final protocol on
    each outcome, then for each pair reports:
      * exact       — fraction of respondents with identical predictions
      * within_1    — fraction with predictions differing by at most 1
      * cohen_kappa
      * weighted_kappa_quadratic (ordinal only)
    """
    import json, os
    from itertools import combinations
    from sklearn.metrics import cohen_kappa_score
    from src.paths import CACHE_DIR

    rows = []
    for outcome, cache_leaf, model_alias, sampling in [
        ("pacdemons", "screening_T08",       "gpt4omini",  "direct"),
        ("womenwork", "screening_vs_cot_T08", "gpt54mini", "vs_cot"),
    ]:
        preds = {}
        for seed in (0, 1, 2):
            # Filenames use the legacy config id "C7" (paper's C6). Search common
            # locations: the repo cache, the historical ~/Library location, and
            # any DT_CACHE_DIR override.
            candidates = [
                CACHE_DIR / cache_leaf / outcome / f"C7_{model_alias}_T08_ben_{sampling}_nocot_pc_s{seed}.jsonl",
                Path(os.path.expanduser("~/Library/Caches/digitaltwin_calibration")) / cache_leaf / outcome / f"C7_{model_alias}_T08_ben_{sampling}_nocot_pc_s{seed}.jsonl",
            ]
            fp = next((c for c in candidates if c.exists()), None)
            if fp is None:
                continue
            preds[seed] = {r["respondent_id"]: r["predicted_value"]
                            for r in map(json.loads, fp.open())}
        if len(preds) < 2:
            print(f"  [Table 15] skipping {outcome} — need ≥2 seed files, found {len(preds)}")
            continue
        for s_a, s_b in combinations(sorted(preds), 2):
            common = set(preds[s_a]) & set(preds[s_b])
            y1 = np.array([preds[s_a][r] for r in common])
            y2 = np.array([preds[s_b][r] for r in common])
            exact = float((y1 == y2).mean())
            within1 = float((np.abs(y1 - y2) <= 1).mean())
            k = float(cohen_kappa_score(y1, y2))
            kq = float(cohen_kappa_score(y1, y2, weights="quadratic")) if outcome == "womenwork" else np.nan
            rows.append({
                "outcome": outcome,
                "seed_pair": f"s{s_a}-s{s_b}",
                "n": len(common),
                "exact_agreement": round(exact, 4),
                "within_1_agreement": round(within1, 4),
                "cohen_kappa": round(k, 4),
                "weighted_kappa_quadratic": round(kq, 4) if not np.isnan(kq) else None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables 16, 17 — Transfer items (Appendix G)
# ---------------------------------------------------------------------------
def table_16() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "table_16_transfer_binary.csv")


def table_17() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "table_17_transfer_ordinal.csv")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
TABLES = {
    1:  lambda: {"Table1_cross_model": table_1()},
    2:  lambda: {"Table2_final_protocols": table_2()},
    3:  lambda: {"Table3_persona_variables": table_3()},
    4:  lambda: {"Table4_persona_configurations": table_4()},
    56: table_5_6,
    79: table_7_9,
    1011: table_10_11,
    1213: table_12_13,
    14: table_14,
    15: lambda: {"Table15_seed_pair_agreement": table_15()},
    16: lambda: {"Table16_transfer_binary": table_16()},
    17: lambda: {"Table17_transfer_ordinal": table_17()},
}

_ALIAS = {1: [1], 2: [2], 3: [3], 4: [4], 5: [56], 6: [56], 7: [79], 8: [79], 9: [79],
          10: [1011], 11: [1011], 12: [1213], 13: [1213], 14: [14], 15: [15], 16: [16], 17: [17]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=int, help="Only write this paper table (1-17)")
    args = ap.parse_args()

    keys = TABLES.keys() if args.table is None else _ALIAS[args.table]
    for k in dict.fromkeys(keys):
        result = TABLES[k]()
        for name, df in result.items():
            out = OUT / f"{name}.csv"
            df.to_csv(out, index=False)
            print(f"  ✓ {out.relative_to(ROOT)}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
