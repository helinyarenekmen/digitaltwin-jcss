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
def _delta_p_pacdemons(pred_mean: float, gt_mean: float) -> float:
    """Paper Δp for pacdemons: simulated minus observed positive-rate in
    percentage points. Because pacdemons is coded 1=Yes, 2=No, the positive
    share is (2 − mean_value); therefore Δp = (gt_mean − pred_mean) × 100."""
    return round((gt_mean - pred_mean) * 100, 1)


def table_1() -> pd.DataFrame:
    """Cross-model comparison at seed zero (paper Table 1 = paper Tables 10-11).

    Panel A — pacdemons, paper C6, direct answering, T=0.8, first-person,
    seed 0, NO ideology background. Four models per paper Table 10.
    Panel B — womenwork, best surviving config per model under verbalized
    sampling, T=0.8, seed 0. Four models per paper Table 11.
    """
    PAPER_MODELS = ["gpt-4o-mini", "gpt-5.4-mini",
                     "meta-llama/llama-3.3-70b-instruct", "gemini-2.5-flash-lite"]

    # --- Panel A: Stage 3 cross-model + GPT-4o-mini repeated from Stage 2 ---
    p = _load("pacdemons")
    stage3 = p[p["Stage"].astype(str).str.contains("Stage 3", na=False) &
               (p["Config"] == "C6") & (p["Sampling"] == "direct") &
               (p["Temperature"] == 0.8) & (p["Seed"] == 0) &
               ~p["Source_CSV"].fillna("").str.contains("politicalcontext")].copy()
    # GPT-4o-mini isn't in Stage 3 (it was the reference); pull it from Stage 2
    gpt4o = p[(p["Config"] == "C6") & (p["Model"] == "gpt-4o-mini") &
              (p["Sampling"] == "direct") & (p["Temperature"] == 0.8) &
              (p["Seed"] == 0) &
              p["Source_CSV"].fillna("").str.contains("shortlist_direct_T08")].copy()
    panel_a_src = pd.concat([gpt4o, stage3]).drop_duplicates(subset=["Model"], keep="first")
    panel_a_src = panel_a_src[panel_a_src["Model"].isin(PAPER_MODELS)]
    panel_a_src["delta_p"] = panel_a_src.apply(
        lambda r: _delta_p_pacdemons(r["pred_mean"], r["gt_mean"]), axis=1)
    panel_a = panel_a_src[["Model", "Config", "jsd", "delta_p",
                            "recall_minority", "precision_minority"]].copy()
    panel_a.columns = ["Model", "Config", "JSD", "delta_p_pp", "Rec+", "Prec+"]
    panel_a["Model"] = pd.Categorical(panel_a["Model"], categories=PAPER_MODELS, ordered=True)
    panel_a = panel_a.sort_values("Model").reset_index(drop=True)
    panel_a["Model"] = panel_a["Model"].astype(str)
    panel_a.insert(0, "Panel", "A")

    # --- Panel B ---
    w = _load("womenwork")
    stage3_w = w[w["Stage"].astype(str).str.contains("Stage 3", na=False) &
                 (w["Config"] == "C6") &
                 w["Sampling"].fillna("").str.contains("vs") &
                 (w["Temperature"] == 0.8) & (w["Seed"] == 0)].copy()
    gpt4o_w = w[(w["Config"] == "C6") & (w["Model"] == "gpt-4o-mini") &
                w["Sampling"].fillna("").str.contains("vs") &
                (w["Temperature"] == 0.8) & (w["Seed"] == 0) &
                ~w["Source_CSV"].fillna("").str.contains("politicalcontext")].copy()
    panel_b_src = pd.concat([gpt4o_w, stage3_w]).drop_duplicates(subset=["Model"], keep="first")
    panel_b_src = panel_b_src[panel_b_src["Model"].isin(PAPER_MODELS)]
    panel_b = panel_b_src[["Model", "Config", "wasserstein", "weighted_kappa_quadratic",
                            "pred_mean"]].copy()
    panel_b.columns = ["Model", "Config", "Wass", "kappa_w", "sim_mean"]
    panel_b["Model"] = pd.Categorical(panel_b["Model"], categories=PAPER_MODELS, ordered=True)
    panel_b = panel_b.sort_values("Model").reset_index(drop=True)
    panel_b["Model"] = panel_b["Model"].astype(str)
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
# ---------------------------------------------------------------------------
# Helper: format columns and drop internal ones for a paper-facing table
# ---------------------------------------------------------------------------
_INTERNAL_COLS = ["Source_CSV", "Notes", "composite_score", "ac2_quadratic",
                   "legacy_id", "Ablation", "Address mode"]


def _clean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 5 — Stage 1 pacdemons
# ---------------------------------------------------------------------------
def table_5() -> pd.DataFrame:
    p = _load("pacdemons")
    r = p[(p["Phase"] == "Phase 1") & (p["Sampling"] == "direct") &
          (p["Temperature"] == 0.0) & (p["Model"] == "gpt-4o-mini")].copy()
    r["Configuration"] = r["Config"]
    r["n"] = r["n_matched"]
    r["Rec+"] = r["recall_minority"].round(3)
    r["Prec+"] = r["precision_minority"].round(3)
    r["JSD"] = r["jsd"].round(5)
    r["MCC"] = r["mcc"].round(3)
    r["k"] = r["cohen_kappa"].round(3)
    order = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]
    r["Configuration"] = pd.Categorical(r["Configuration"], categories=order, ordered=True)
    r = r.sort_values("Configuration").astype({"Configuration": str})
    return _clean(r, ["Configuration", "n", "JSD", "MCC", "k", "Rec+", "Prec+"])


def table_6() -> pd.DataFrame:
    w = _load("womenwork")
    r = w[(w["Phase"] == "Phase 1") & (w["Sampling"] == "direct") &
          (w["Temperature"] == 0.0) & (w["Model"] == "gpt-4o-mini")].copy()
    r["Configuration"] = r["Config"]
    r["n"] = r["n_matched"]
    r["Wass"] = r["wasserstein"].round(3)
    r["JSD"] = r["jsd"].round(3)
    r["k"] = r["cohen_kappa"].round(3)
    r["k_w"] = r["weighted_kappa_quadratic"].round(3)
    r["Sim. mean"] = r["pred_mean"].round(2)
    order = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]
    r["Configuration"] = pd.Categorical(r["Configuration"], categories=order, ordered=True)
    r = r.sort_values("Configuration").astype({"Configuration": str})
    return _clean(r, ["Configuration", "n", "Wass", "JSD", "k", "k_w", "Sim. mean"])


# ---------------------------------------------------------------------------
# Tables 7-8 — Stage 2 pacdemons and womenwork sampling × temperature grid
# ---------------------------------------------------------------------------
def _stage2_grid(dv: str) -> pd.DataFrame:
    x = _load(dv)
    shortlist = ["C1", "C2", "C3", "C5", "C6", "C10"]
    r = x[(x["Config"].isin(shortlist)) & (x["Model"] == "gpt-4o-mini") &
          (x["Seed"] == 0)].copy()
    # keep only direct + vs_cot at T=0/0.4/0.8
    r = r[r["Sampling"].isin(["direct", "vs_cot", "vs"])]
    r = r[r["Temperature"].isin([0.0, 0.4, 0.8])]
    # de-duplicate keeping earliest
    r = r.drop_duplicates(subset=["Config", "Sampling", "Temperature"], keep="first")
    return r


def table_7() -> pd.DataFrame:
    """pacdemons Stage 2 grid: 6 configs × 2 sampling × 3 T."""
    r = _stage2_grid("pacdemons")
    r["JSD"] = r["jsd"].round(5)
    r["MCC"] = r["mcc"].round(3)
    r["Rec+"] = r["recall_minority"].round(3)
    r["Sampling"] = r["Sampling"].str.replace("vs_cot", "verbalized").str.replace("vs", "verbalized").str.replace("direct", "direct")
    r["T"] = r["Temperature"]
    r["Config"] = pd.Categorical(r["Config"], categories=["C1","C2","C3","C5","C6","C10"], ordered=True)
    r["Sampling"] = pd.Categorical(r["Sampling"], categories=["direct","verbalized"], ordered=True)
    r = r.sort_values(["Sampling", "Config", "T"]).astype({"Config": str, "Sampling": str})
    return _clean(r, ["Sampling", "Config", "T", "JSD", "MCC", "Rec+"])


def table_8() -> pd.DataFrame:
    """womenwork Stage 2 grid: 6 configs × 2 sampling × 3 T."""
    r = _stage2_grid("womenwork")
    r["Wass"] = r["wasserstein"].round(3)
    r["JSD"] = r["jsd"].round(3)
    r["k_w"] = r["weighted_kappa_quadratic"].round(3)
    r["Sampling"] = r["Sampling"].replace({"vs_cot": "verbalized", "vs": "verbalized"})
    r["T"] = r["Temperature"]
    r["Config"] = pd.Categorical(r["Config"], categories=["C1","C2","C3","C5","C6","C10"], ordered=True)
    r["Sampling"] = pd.Categorical(r["Sampling"], categories=["direct","verbalized"], ordered=True)
    r = r.sort_values(["Sampling", "Config", "T"]).astype({"Config": str, "Sampling": str})
    return _clean(r, ["Sampling", "Config", "T", "Wass", "JSD", "k_w"])


# ---------------------------------------------------------------------------
# Table 9 — Fit-agreement trade-off for womenwork VS T=0.8 seed 0
# ---------------------------------------------------------------------------
def table_9() -> pd.DataFrame:
    r = _stage2_grid("womenwork")
    r = r[(r["Sampling"].isin(["vs_cot", "vs"])) & (r["Temperature"] == 0.8)].copy()
    r["Configuration"] = r["Config"]
    r["Wass"] = r["wasserstein"].round(3)
    r["k_w"] = r["weighted_kappa_quadratic"].round(3)
    r["Sim. mean"] = r["pred_mean"].round(2)
    order = ["C10", "C5", "C3", "C2", "C1", "C6"]
    r["Configuration"] = pd.Categorical(r["Configuration"], categories=order, ordered=True)
    r = r.sort_values("Configuration").astype({"Configuration": str})
    return _clean(r, ["Configuration", "Wass", "k_w", "Sim. mean"])


# ---------------------------------------------------------------------------
# Tables 10-11 — Stage 3 cross-model
# ---------------------------------------------------------------------------
def table_10() -> pd.DataFrame:
    """pacdemons C6 direct T=0.8, 4 models."""
    p = _load("pacdemons")
    stage3 = p[p["Stage"].astype(str).str.contains("Stage 3", na=False) &
                (p["Config"] == "C6") & (p["Sampling"] == "direct") &
                (p["Temperature"] == 0.8) & (p["Seed"] == 0) &
                ~p["Source_CSV"].fillna("").str.contains("politicalcontext")].copy()
    gpt4o = p[(p["Config"] == "C6") & (p["Model"] == "gpt-4o-mini") &
              (p["Sampling"] == "direct") & (p["Temperature"] == 0.8) &
              (p["Seed"] == 0) &
              p["Source_CSV"].fillna("").str.contains("shortlist_direct_T08")].copy()
    r = pd.concat([gpt4o, stage3]).drop_duplicates("Model", keep="first")
    order = ["gpt-4o-mini", "gpt-5.4-mini", "meta-llama/llama-3.3-70b-instruct", "gemini-2.5-flash-lite"]
    r = r[r["Model"].isin(order)]
    r["Delta_p_pp"] = r.apply(lambda x: _delta_p_pacdemons(x["pred_mean"], x["gt_mean"]), axis=1)
    r["n"] = r["n_matched"]
    r["JSD"] = r["jsd"].round(5)
    r["MCC"] = r["mcc"].round(3)
    r["Rec+"] = r["recall_minority"].round(3)
    r["Prec+"] = r["precision_minority"].round(3)
    r["Model"] = pd.Categorical(r["Model"], categories=order, ordered=True)
    r = r.sort_values("Model").astype({"Model": str})
    return _clean(r, ["Model", "n", "JSD", "Delta_p_pp", "MCC", "Rec+", "Prec+"])


def table_11() -> pd.DataFrame:
    """womenwork verbalized T=0.8 across 4 configs × 4-5 models."""
    w = _load("womenwork")
    stage3 = w[w["Stage"].astype(str).str.contains("Stage 3", na=False) &
                w["Sampling"].fillna("").str.contains("vs") &
                (w["Temperature"] == 0.8) & (w["Seed"] == 0)].copy()
    gpt4o = w[(w["Model"] == "gpt-4o-mini") & (w["Config"].isin(["C2","C3","C6","C10"])) &
              w["Sampling"].fillna("").str.contains("vs") &
              (w["Temperature"] == 0.8) & (w["Seed"] == 0) &
              ~w["Source_CSV"].fillna("").str.contains("politicalcontext")].copy()
    r = pd.concat([gpt4o, stage3]).drop_duplicates(subset=["Model", "Config"], keep="first")
    order_m = ["gpt-4o-mini", "gpt-5.4-mini", "meta-llama/llama-3.3-70b-instruct",
                "gemini-2.5-flash-lite", "gpt-4.1-mini"]
    order_c = ["C2", "C3", "C6", "C10"]
    r = r[r["Model"].isin(order_m)]
    r = r[r["Config"].isin(order_c)]
    r["n"] = r["n_matched"]
    r["Wass"] = r["wasserstein"].round(3)
    r["JSD"] = r["jsd"].round(3)
    r["k_w"] = r["weighted_kappa_quadratic"].round(3)
    r["Sim. mean"] = r["pred_mean"].round(2)
    r["Model"] = pd.Categorical(r["Model"], categories=order_m, ordered=True)
    r["Config"] = pd.Categorical(r["Config"], categories=order_c, ordered=True)
    r = r.sort_values(["Model", "Config"]).astype({"Model": str, "Config": str})
    return _clean(r, ["Model", "Config", "n", "Wass", "JSD", "k_w", "Sim. mean"])


# ---------------------------------------------------------------------------
# Tables 12-13 — Stage 4 prompt-presentation stress tests
# ---------------------------------------------------------------------------
_VARIANT_ORDER = ["Baseline", "Second person", "Reasoning instruction",
                   "Ideology background", "Natural rewrite"]


def _stage4_variant_label(row) -> str:
    src = str(row.get("Source_CSV", "") or "").lower()
    if "sendili" in src: return "Second person"
    if "explicitcot" in src: return "Reasoning instruction"
    if "politicalcontext" in src: return "Ideology background"
    if "natural_ablation" in src: return "Natural rewrite"
    # Baseline comes from Stage 2 shortlist row
    return "Baseline"


def table_12() -> pd.DataFrame:
    """pacdemons Stage 4 stress tests + Stage 2 shortlist baseline row."""
    p = _load("pacdemons")
    stage4 = p[p["Stage"].astype(str).str.contains("Stage 4", na=False) &
                (p["Config"] == "C6") & (p["Model"] == "gpt-4o-mini") &
                (p["Sampling"] == "direct") & (p["Temperature"] == 0.8) &
                (p["Seed"] == 0)].copy()
    baseline = p[(p["Config"] == "C6") & (p["Model"] == "gpt-4o-mini") &
                  (p["Sampling"] == "direct") & (p["Temperature"] == 0.8) &
                  (p["Seed"] == 0) &
                  p["Source_CSV"].fillna("").str.contains("shortlist_direct_T08")].copy()
    r = pd.concat([baseline, stage4])
    r["Variant"] = r.apply(_stage4_variant_label, axis=1)
    r = r.drop_duplicates(subset=["Variant"], keep="first")
    r = r[r["Variant"].isin(_VARIANT_ORDER)]
    r["n"] = r["n_matched"]
    r["JSD"] = r["jsd"].round(5)
    r["MCC"] = r["mcc"].round(3)
    r["k"] = r["cohen_kappa"].round(3)
    r["Rec+"] = r["recall_minority"].round(3)
    r["Variant"] = pd.Categorical(r["Variant"], categories=_VARIANT_ORDER, ordered=True)
    r = r.sort_values("Variant").astype({"Variant": str})
    return _clean(r, ["Variant", "n", "JSD", "MCC", "k", "Rec+"])


def table_13() -> pd.DataFrame:
    """womenwork Stage 4 stress tests + Stage 2 shortlist baseline row."""
    w = _load("womenwork")
    stage4 = w[w["Stage"].astype(str).str.contains("Stage 4", na=False) &
                (w["Config"] == "C6") & (w["Model"] == "gpt-5.4-mini") &
                w["Sampling"].fillna("").str.contains("vs") &
                (w["Temperature"] == 0.8) & (w["Seed"] == 0)].copy()
    baseline = w[(w["Config"] == "C6") & (w["Model"] == "gpt-5.4-mini") &
                  w["Sampling"].fillna("").str.contains("vs") &
                  (w["Temperature"] == 0.8) & (w["Seed"] == 0) &
                  ~w["Source_CSV"].fillna("").str.contains("politicalcontext")].copy()
    r = pd.concat([baseline, stage4])
    r["Variant"] = r.apply(_stage4_variant_label, axis=1)
    r = r.drop_duplicates(subset=["Variant"], keep="first")
    r = r[r["Variant"].isin(_VARIANT_ORDER)]
    r["n"] = r["n_matched"]
    r["Wass"] = r["wasserstein"].round(3)
    r["JSD"] = r["jsd"].round(3)
    r["k_w"] = r["weighted_kappa_quadratic"].round(3)
    r["Variant"] = pd.Categorical(r["Variant"], categories=_VARIANT_ORDER, ordered=True)
    r = r.sort_values("Variant").astype({"Variant": str})
    return _clean(r, ["Variant", "n", "Wass", "JSD", "k_w"])


# ---------------------------------------------------------------------------
# Table 14 — Stage 5 aggregate seed stability with and without background
# ---------------------------------------------------------------------------
def table_14() -> dict[str, pd.DataFrame]:
    p = _load("pacdemons")
    # Baseline three seeds: Phase 12 rows without PC
    bp = p[(p["Config"] == "C6") & (p["Model"] == "gpt-4o-mini") &
            (p["Sampling"] == "direct") & (p["Temperature"] == 0.8) &
            (p["Seed"].isin([0, 1, 2])) &
            ~p["Source_CSV"].fillna("").str.contains("politicalcontext") &
            p["Source_CSV"].fillna("").str.contains("shortlist_direct_T08|seed[12]_shortlist")].copy()
    bp["Framing"] = "Baseline"
    # +Background three seeds: Phase 15 rows
    bg = p[(p["Config"] == "C6") & (p["Model"] == "gpt-4o-mini") &
            (p["Sampling"] == "direct") & (p["Temperature"] == 0.8) &
            (p["Seed"].isin([0, 1, 2])) &
            p["Source_CSV"].fillna("").str.contains("politicalcontext")].copy()
    bg["Framing"] = "+ Background"
    pa = pd.concat([bp, bg])
    pa["Delta_p_pp"] = pa.apply(lambda x: _delta_p_pacdemons(x["pred_mean"], x["gt_mean"]), axis=1)
    pa["JSD"] = pa["jsd"].round(5)
    pa["MCC"] = pa["mcc"].round(3)
    pa["k"] = pa["cohen_kappa"].round(3)
    pa["Rec+"] = pa["recall_minority"].round(3)
    pa["Framing"] = pd.Categorical(pa["Framing"], categories=["Baseline", "+ Background"], ordered=True)
    pa = pa[["Framing", "Seed", "JSD", "Delta_p_pp", "MCC", "k", "Rec+"]].sort_values(["Framing", "Seed"]).astype({"Framing": str})

    w = _load("womenwork")
    # Plain baseline only: drop rows that carry a stress-test variant
    # (Second person / Reasoning instruction / Ideology background / Natural rewrite).
    _STRESS = "sendili|explicitcot|politicalcontext|natural_ablation"
    bw = w[(w["Config"] == "C6") & (w["Model"] == "gpt-5.4-mini") &
            w["Sampling"].fillna("").str.contains("vs") & (w["Temperature"] == 0.8) &
            (w["Seed"].isin([0, 1, 2])) &
            ~w["Source_CSV"].fillna("").str.contains(_STRESS)].copy()
    bw["Framing"] = "Baseline"
    bgw = w[(w["Config"] == "C6") & (w["Model"] == "gpt-5.4-mini") &
             w["Sampling"].fillna("").str.contains("vs") & (w["Temperature"] == 0.8) &
             (w["Seed"].isin([0, 1, 2])) &
             w["Source_CSV"].fillna("").str.contains("politicalcontext", case=False)].copy()
    bgw["Framing"] = "+ Background"
    pb = pd.concat([bw, bgw])
    pb["Wass"] = pb["wasserstein"].round(3)
    pb["JSD"] = pb["jsd"].round(3)
    pb["k_w"] = pb["weighted_kappa_quadratic"].round(3)
    pb["Sim. mean"] = pb["pred_mean"].round(3)
    pb["Framing"] = pd.Categorical(pb["Framing"], categories=["Baseline", "+ Background"], ordered=True)
    pb = pb[["Framing", "Seed", "Wass", "JSD", "k_w", "Sim. mean"]].sort_values(["Framing", "Seed"]).astype({"Framing": str})

    return {"Table14A_pacdemons": pa.reset_index(drop=True),
            "Table14B_womenwork": pb.reset_index(drop=True)}


# Backward-compatible aliases retained so dispatcher still works
def table_5_6(): return {"Table5_stage1_pacdemons": table_5(),
                          "Table6_stage1_womenwork": table_6()}
def table_7_9(): return {"Table7_stage2_pacdemons": table_7(),
                          "Table8_stage2_womenwork": table_8(),
                          "Table9_stage2_fit_agreement_womenwork": table_9()}
def table_10_11(): return {"Table10_stage3_pacdemons": table_10(),
                            "Table11_stage3_womenwork": table_11()}
def table_12_13(): return {"Table12_stage4_pacdemons": table_12(),
                            "Table13_stage4_womenwork": table_13()}


# ---------------------------------------------------------------------------
# Table 15 — Seed-pair agreement (exact, ±1, unweighted kappa, weighted kappa)
# ---------------------------------------------------------------------------
def table_15() -> pd.DataFrame:
    """Reproduce paper Table 15 (respondent-level agreement between base seeds).

    The paper protocol samples from each respondent's verbalized distribution
    using a SINGLE numpy Generator initialised once at the base seed and
    advanced across respondents in file-appearance order. For pacdemons (direct
    answering) there is no client-side sampling, so the archived
    predicted_value is used directly.

    Fresh-clone fallback: when the raw completions cache is not present, the
    committed copy at outputs/results/table_15_seed_pair_agreement.csv is
    returned so that Path B (reproduce from archived outputs) does not require
    the raw JSONL bundle.
    """
    import json, os, re
    from itertools import combinations
    import numpy as np
    from sklearn.metrics import cohen_kappa_score
    from src.paths import CACHE_DIR

    def _find(cache_leaf, outcome, model_alias, sampling, seed):
        candidates = [
            CACHE_DIR / cache_leaf / outcome / f"C7_{model_alias}_T08_ben_{sampling}_nocot_pc_s{seed}.jsonl",
            Path(os.path.expanduser("~/Library/Caches/digitaltwin_calibration")) / cache_leaf / outcome / f"C7_{model_alias}_T08_ben_{sampling}_nocot_pc_s{seed}.jsonl",
        ]
        return next((c for c in candidates if c.exists()), None)

    # Fresh-clone shortcut: no cache anywhere → return the committed CSV.
    any_cache = any(
        _find(leaf, outcome, ma, samp, s) is not None
        for outcome, leaf, ma, samp in (
            ("pacdemons", "screening_T08",        "gpt4omini", "direct"),
            ("womenwork", "screening_vs_cot_T08", "gpt54mini", "vs_cot"),
        )
        for s in (0, 1, 2)
    )
    if not any_cache:
        committed = ROOT / "outputs" / "results" / "table_15_seed_pair_agreement.csv"
        if committed.exists():
            print(f"  [Table 15] cache absent → using committed {committed.relative_to(ROOT)}")
            return pd.read_csv(committed)

    _DIST_RE = re.compile(r"DAĞILIM:\s*(\{[^}]+\})", re.DOTALL)

    def _sample_one_rng(records, base_seed, valid_range):
        """Apply paper's one-RNG-per-run protocol to a list of records
        (already sorted by respondent_id). Returns {rid: predicted_value}."""
        lo, hi = valid_range
        rng = np.random.default_rng(int(base_seed))
        out = {}
        for r in records:
            m = _DIST_RE.search(r.get("raw_response") or "")
            if not m:
                # Cannot resample; keep whatever was archived
                out[r["respondent_id"]] = r.get("predicted_value")
                continue
            try:
                raw = json.loads(m.group(1))
            except Exception:
                out[r["respondent_id"]] = r.get("predicted_value")
                continue
            probs = {int(k): float(v) for k, v in raw.items() if lo <= int(k) <= hi}
            total = sum(probs.values())
            if total <= 0:
                out[r["respondent_id"]] = r.get("predicted_value")
                continue
            keys = list(probs); ps = [probs[k] / total for k in keys]
            out[r["respondent_id"]] = int(rng.choice(keys, p=ps))
        return out

    rows = []
    for outcome, cache_leaf, model_alias, sampling, valid, resample in [
        ("pacdemons", "screening_T08",        "gpt4omini",  "direct",  (1, 2), False),
        ("womenwork", "screening_vs_cot_T08", "gpt54mini",  "vs_cot",  (1, 5), True),
    ]:
        preds = {}
        for seed in (0, 1, 2):
            fp = _find(cache_leaf, outcome, model_alias, sampling, seed)
            if fp is None:
                continue
            records = [json.loads(l) for l in fp.open() if l.strip()]
            records = [r for r in records if r.get("parse_status") == "ok"]
            # Paper protocol: single RNG per base seed, advanced across
            # respondents in FILE-APPEARANCE ORDER (async completion order at
            # inference time). Sorting here would change the specific samples.
            if resample:
                preds[seed] = _sample_one_rng(records, seed, valid)
            else:
                preds[seed] = {r["respondent_id"]: r["predicted_value"] for r in records}
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
