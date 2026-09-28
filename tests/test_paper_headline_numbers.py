"""
Sanity checks against the paper's headline numbers.

These verify that the archived aggregated outputs still yield the paper's
reported values within a 0.005 absolute tolerance. Any mismatch here is a
real problem — do not adjust tolerances to force a pass.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import pytest

ROOT   = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"


# --- Table 2: final calibration protocols ---
def _load_pacdemons_final():
    x = pd.read_excel(RESULTS / "pacdemons_all_phases.xlsx", sheet_name="All cells")
    r = x[(x["Config"] == "C6") &                    # paper label (legacy id was C7)
          (x["Model"] == "gpt-4o-mini") &
          (x["Sampling"] == "direct") &
          (x["Temperature"] == 0.8) &
          (x["Address mode"] == "ben") &
          (x["Seed"] == 0) &
          x["Source_CSV"].fillna("").str.contains("politicalcontext")]
    assert len(r) == 1, f"expected 1 row, got {len(r)}"
    return r.iloc[0]


def _load_pacdemons_final_3seed():
    x = pd.read_excel(RESULTS / "pacdemons_all_phases.xlsx", sheet_name="All cells")
    r = x[(x["Config"] == "C6") &
          (x["Model"] == "gpt-4o-mini") &
          (x["Sampling"] == "direct") &
          (x["Temperature"] == 0.8) &
          (x["Address mode"] == "ben") &
          x["Source_CSV"].fillna("").str.contains("politicalcontext")]
    assert len(r) >= 3, f"expected ≥3 rows (3 seeds), got {len(r)}"
    return r


def _load_womenwork_final_avg():
    x = pd.read_excel(RESULTS / "womenwork_all_phases.xlsx", sheet_name="All cells")
    r = x[(x["Config"] == "C6") &
          x["Model"].fillna("").str.contains("gpt-5.4-mini") &
          x["Sampling"].fillna("").str.contains("vs") &
          (x["Temperature"] == 0.8) &
          x["Source_CSV"].fillna("").str.contains("politicalcontext", case=False)]
    assert len(r) >= 3, f"expected ≥3 rows (3 seeds), got {len(r)}"
    return r


def test_table_2_pacdemons_final_recall_3seed_mean():
    """Table 2: paper reports the 3-seed mean recall = 0.512."""
    r = _load_pacdemons_final_3seed()
    recall_mean = r["recall_minority"].mean()
    assert abs(recall_mean - 0.512) < 0.005, \
        f"pacdemons 3-seed mean recall: expected 0.512, got {recall_mean:.4f}"


def test_table_2_pacdemons_final_jsd():
    """Table 2 pacdemons JSD = 0.0011 (seed 0 cell, matches paper)."""
    r = _load_pacdemons_final()
    assert abs(float(r["jsd"]) - 0.0011) < 0.005, \
        f"pacdemons JSD: expected 0.0011, got {r['jsd']:.4f}"


def test_table_2_womenwork_final():
    r = _load_womenwork_final_avg()
    wass = r["wasserstein"].mean()
    kw   = r["weighted_kappa_quadratic"].mean()
    assert abs(wass - 0.181) < 0.005, f"womenwork Wass: expected 0.181, got {wass:.4f}"
    assert abs(kw   - 0.312) < 0.005, f"womenwork κw: expected 0.312, got {kw:.4f}"


# --- Table 16: transfer-item binary protocol ---
def test_table_16_transfer_binary():
    tab = pd.read_csv(RESULTS / "table_16_transfer_binary.csv")
    tab = tab.set_index("outcome")
    expected = {"pacvolunteer": 0.662, "paccontact": 0.569, "paccompl": 0.498}
    for outcome, want in expected.items():
        got = float(tab.loc[outcome, "recall_minority"])
        assert abs(got - want) < 0.005, \
            f"Table 16 {outcome} recall_minority: expected {want}, got {got:.4f}"


# --- Table 17: transfer-item ordinal protocol ---
def test_table_17_transfer_ordinal():
    tab = pd.read_csv(RESULTS / "table_17_transfer_ordinal.csv")
    tab = tab.set_index("outcome")
    expected = {"satdem": 0.562, "famroles": 0.530, "polint": 0.513}
    for outcome, want in expected.items():
        got = float(tab.loc[outcome, "weighted_kappa_quadratic"])
        assert abs(got - want) < 0.005, \
            f"Table 17 {outcome} κw: expected {want}, got {got:.4f}"


# --- Experiment: main effects (Section 4) ---
def _load_experiment_long():
    return pd.read_csv(ROOT / "outputs" / "parsed" / "all_items_long.csv")


def test_experiment_legitimacy_effect():
    """Δ = −0.41, 95% CI [−0.47, −0.35]. Sign is Security − Peace."""
    import numpy as np
    from scipy import stats
    df = _load_experiment_long()
    d = df[(df["parse_status"] == "ok") & (df["item_id"] == "dv_legitimacy")]
    wide = d.pivot_table(index="respondent_id", columns="condition",
                          values="predicted_value", aggfunc="first").dropna()
    diff = (wide["S"] - wide["P"]).astype(float).values
    n = len(diff)
    delta = float(diff.mean())
    se    = float(diff.std(ddof=1) / (n ** 0.5))
    lo, hi = delta - 1.96 * se, delta + 1.96 * se
    assert abs(delta - (-0.41)) < 0.02, f"legitimacy Δ: expected −0.41, got {delta:.3f}"
    assert abs(lo    - (-0.47)) < 0.02, f"legitimacy CI lo: expected −0.47, got {lo:.3f}"
    assert abs(hi    - (-0.35)) < 0.02, f"legitimacy CI hi: expected −0.35, got {hi:.3f}"


def test_experiment_behavioral_effect():
    """Δ = −15.76 pp, 95% CI [−17.15, −14.36]."""
    df = _load_experiment_long()
    d = df[(df["parse_status"] == "ok") & (df["item_id"] == "dv_behavioral_intent")]
    wide = d.pivot_table(index="respondent_id", columns="condition",
                          values="predicted_value", aggfunc="first").dropna()
    S_att = (wide["S"] == 1).astype(float).values
    P_att = (wide["P"] == 1).astype(float).values
    diff = (S_att - P_att) * 100
    n = len(diff)
    delta = float(diff.mean())
    se    = float(diff.std(ddof=1) / (n ** 0.5))
    lo, hi = delta - 1.96 * se, delta + 1.96 * se
    assert abs(delta - (-15.76)) < 0.5, f"behavioral Δ: expected −15.76 pp, got {delta:.2f}"
    assert abs(lo    - (-17.15)) < 0.5, f"behavioral CI lo: expected −17.15 pp, got {lo:.2f}"
    assert abs(hi    - (-14.36)) < 0.5, f"behavioral CI hi: expected −14.36 pp, got {hi:.2f}"


# --- Subgroup coding sanity: Kurdish and Left subgroups both n = 573 ---
def test_subgroup_ns_kurdish_and_left():
    """Verify that the Kurdish and Left subgroups both count 573 respondents
    in the matched-legitimacy sample.

    Coding rules (paper Section 4.2.2):
      * Kurdish  = kurd == 1 OR lankurd == 1     (missing ethnicity → Non-Kurdish)
      * Left     = pidleftright <= 3
    """
    tgss = pd.read_csv(ROOT / "data" / "derived" / "tgss2024_clean.csv")
    exp = pd.read_csv(ROOT / "outputs" / "parsed" / "all_items_long.csv")
    matched = (
        exp[(exp["parse_status"] == "ok") & (exp["item_id"] == "dv_legitimacy")]
        .pivot_table(index="respondent_id", columns="condition",
                     values="predicted_value", aggfunc="first")
        .dropna()
    )
    meta = tgss.copy()
    meta["respondent_id"] = meta["id"].astype(int).apply(lambda x: f"TGSS_{x:04d}")
    meta = meta[meta["respondent_id"].isin(set(matched.index))]

    kurdish_matched = ((meta.get("kurd") == 1.0) | (meta.get("lankurd") == 1.0)).sum()
    left_matched    = (meta["pidleftright"] <= 3).sum()

    assert kurdish_matched == 573, f"kurdish matched: expected 573, got {kurdish_matched}"
    assert left_matched    == 573, f"left matched:    expected 573, got {left_matched}"


# --- Experiment: standardized effect size d_z for legitimacy ---
def test_experiment_legitimacy_dz():
    """d_z = mean(diff) / sd(diff), paired within-persona. Paper reports −0.26."""
    df = _load_experiment_long()
    d = df[(df["parse_status"] == "ok") & (df["item_id"] == "dv_legitimacy")]
    wide = d.pivot_table(index="respondent_id", columns="condition",
                          values="predicted_value", aggfunc="first").dropna()
    diff = (wide["S"] - wide["P"]).astype(float).values
    dz = diff.mean() / diff.std(ddof=1)
    assert abs(dz - (-0.26)) < 0.02, f"legitimacy d_z: expected −0.26, got {dz:.3f}"
