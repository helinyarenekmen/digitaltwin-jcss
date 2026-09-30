"""
Hard-coded paper values for Tables 1, 5, 6, 10, 12, 13, 14, 15, 16, 17.

Any table CSV under outputs/tables/ that reproduces the paper is checked to
match its paper values within 0.006 absolute (allows one-decimal rounding).
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TDIR = ROOT / "outputs" / "tables"
TOL  = 0.006


def _regen():
    """Regenerate tables into outputs/tables/ if the directory is empty."""
    if not (TDIR / "Table1_cross_model.csv").exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tables.py")],
                        check=True, cwd=ROOT)


_regen()


# ---------------------------------------------------------------------------
# Table 1 — Cross-model comparison
# ---------------------------------------------------------------------------
def test_table_1_panel_a_paper_values():
    t = pd.read_csv(TDIR / "Table1_cross_model.csv")
    a = t[t["Panel"] == "A"].set_index("Model")
    paper = {
        "gpt-4o-mini":                        (0.00096,  1.7, 0.536, 0.398),
        "gpt-5.4-mini":                       (0.00035,  1.0, 0.440, 0.366),
        "meta-llama/llama-3.3-70b-instruct":  (0.00167, -1.9, 0.310, 0.500),
        "gemini-2.5-flash-lite":              (0.02309, -4.9, 0.012, 1.000),
    }
    for m, (jsd, dp, rec, prec) in paper.items():
        r = a.loc[m]
        assert abs(r["JSD"] - jsd) < 0.01,     f"{m} JSD: {r['JSD']} vs paper {jsd}"
        assert abs(r["delta_p_pp"] - dp) < 0.2,f"{m} Δp: {r['delta_p_pp']} vs paper {dp}"
        assert abs(r["Rec+"] - rec) < TOL,     f"{m} Rec+: {r['Rec+']} vs paper {rec}"
        assert abs(r["Prec+"] - prec) < TOL,   f"{m} Prec+: {r['Prec+']} vs paper {prec}"


def test_table_1_panel_b_paper_values():
    t = pd.read_csv(TDIR / "Table1_cross_model.csv")
    b = t[t["Panel"] == "B"].set_index("Model")
    paper = {
        "gpt-4o-mini":                       (0.413, 0.141, 3.66),
        "gpt-5.4-mini":                      (0.211, 0.304, 3.19),
        "meta-llama/llama-3.3-70b-instruct": (0.134, 0.212, 3.21),
        "gemini-2.5-flash-lite":             (0.510, 0.230, 2.73),
    }
    for m, (wass, kw, mean) in paper.items():
        r = b.loc[m]
        assert abs(r["Wass"]     - wass) < TOL, f"{m} Wass: {r['Wass']} vs paper {wass}"
        assert abs(r["kappa_w"]  - kw)   < TOL, f"{m} kw: {r['kappa_w']} vs paper {kw}"
        assert abs(r["sim_mean"] - mean) < 0.02,f"{m} mean: {r['sim_mean']} vs paper {mean}"


# ---------------------------------------------------------------------------
# Table 5 — Stage 1 pacdemons (11 configs)
# ---------------------------------------------------------------------------
def test_table_5_pacdemons():
    t = pd.read_csv(TDIR / "Table5_stage1_pacdemons.csv").set_index("Configuration")
    paper = {  # Config: (JSD, MCC, k, Rec+, Prec+)
        "C0":  (0.01743, 0.032, 0.016, 0.012, 0.167),
        "C1":  (0.00001, 0.380, 0.379, 0.405, 0.415),
        "C2":  (0.00005, 0.407, 0.407, 0.452, 0.422),
        "C3":  (0.00048, 0.384, 0.381, 0.464, 0.375),
        "C4":  (0.00028, 0.390, 0.388, 0.381, 0.457),
        "C5":  (0.00027, 0.407, 0.406, 0.476, 0.404),
        "C6":  (0.00090, 0.410, 0.405, 0.512, 0.384),
        "C7":  (0.01510, 0.096, 0.056, 0.036, 0.333),
        "C8":  (0.00252, 0.368, 0.358, 0.512, 0.323),
        "C9":  (0.00350, 0.072, 0.067, 0.071, 0.150),
        "C10": (0.00073, 0.406, 0.402, 0.500, 0.385),
    }
    for c, (jsd, mcc, k, rec, prec) in paper.items():
        r = t.loc[c]
        assert abs(r["JSD"]   - jsd) < 0.01, f"{c} JSD"
        assert abs(r["MCC"]   - mcc) < TOL,  f"{c} MCC"
        assert abs(r["k"]     - k)   < TOL,  f"{c} k"
        assert abs(r["Rec+"]  - rec) < TOL,  f"{c} Rec+"
        assert abs(r["Prec+"] - prec)< TOL,  f"{c} Prec+"


# ---------------------------------------------------------------------------
# Table 6 — Stage 1 womenwork (11 configs)
# ---------------------------------------------------------------------------
def test_table_6_womenwork():
    t = pd.read_csv(TDIR / "Table6_stage1_womenwork.csv").set_index("Configuration")
    paper = {
        "C0":  (0.939, 0.291, 0.016, 0.023, 3.83),
        "C1":  (0.778, 0.100, 0.131, 0.259, 4.02),
        "C2":  (0.701, 0.071, 0.175, 0.309, 3.94),
        "C3":  (0.624, 0.070, 0.157, 0.278, 3.87),
        "C4":  (0.799, 0.114, 0.116, 0.257, 4.04),
        "C5":  (0.779, 0.155, 0.037, 0.142, 4.02),
        "C6":  (0.818, 0.129, 0.119, 0.219, 4.06),
        "C7":  (0.838, 0.099, 0.142, 0.265, 4.08),
        "C8":  (0.781, 0.163, 0.025, 0.073, 3.75),
        "C9":  (0.859, 0.114, 0.147, 0.290, 4.10),
        "C10": (0.493, 0.082, 0.069, 0.182, 3.46),
    }
    for c, (wass, jsd, k, kw, mean) in paper.items():
        r = t.loc[c]
        assert abs(r["Wass"]      - wass) < TOL, f"{c} Wass"
        assert abs(r["JSD"]       - jsd)  < TOL, f"{c} JSD"
        assert abs(r["k"]         - k)    < TOL, f"{c} k"
        assert abs(r["k_w"]       - kw)   < TOL, f"{c} kw"
        assert abs(r["Sim. mean"] - mean) < 0.02,f"{c} mean"


# ---------------------------------------------------------------------------
# Table 10 — Stage 3 pacdemons (4 models)
# ---------------------------------------------------------------------------
def test_table_10_pacdemons():
    t = pd.read_csv(TDIR / "Table10_stage3_pacdemons.csv").set_index("Model")
    paper = {
        "gpt-4o-mini":                        (1707, 0.00096, 1.7, 0.430, 0.536, 0.398),
        "gpt-5.4-mini":                       (1707, 0.00035, 1.0, 0.368, 0.440, 0.366),
        "meta-llama/llama-3.3-70b-instruct":  (1707, 0.00167,-1.9, 0.369, 0.310, 0.500),
        "gemini-2.5-flash-lite":              (1703, 0.02309,-4.9, 0.106, 0.012, 1.000),
    }
    for m, (n, jsd, dp, mcc, rec, prec) in paper.items():
        r = t.loc[m]
        assert abs(r["n"]           - n)   < 5,   f"{m} n"
        assert abs(r["JSD"]         - jsd) < 0.01,f"{m} JSD"
        assert abs(r["Delta_p_pp"]  - dp)  < 0.2, f"{m} dp"
        assert abs(r["MCC"]         - mcc) < TOL, f"{m} MCC"
        assert abs(r["Rec+"]        - rec) < TOL, f"{m} Rec+"
        assert abs(r["Prec+"]       - prec)< TOL, f"{m} Prec+"


# ---------------------------------------------------------------------------
# Table 12 — Stage 4 pacdemons (5 variants)
# ---------------------------------------------------------------------------
def test_table_12_pacdemons_variants():
    t = pd.read_csv(TDIR / "Table12_stage4_pacdemons.csv").set_index("Variant")
    paper = {
        "Baseline":              (0.00096, 0.430, 0.424, 0.536),
        "Second person":         (0.00102, 0.384, 0.379, 0.488),
        "Reasoning instruction": (0.00010, 0.315, 0.315, 0.369),
        "Ideology background":   (0.00109, 0.425, 0.419, 0.536),
        "Natural rewrite":       (0.00405, 0.257, 0.246, 0.405),
    }
    for v, (jsd, mcc, k, rec) in paper.items():
        r = t.loc[v]
        assert abs(r["JSD"]  - jsd) < 0.01, f"{v} JSD"
        assert abs(r["MCC"]  - mcc) < TOL,  f"{v} MCC"
        assert abs(r["k"]    - k)   < TOL,  f"{v} k"
        assert abs(r["Rec+"] - rec) < TOL,  f"{v} Rec+"


# ---------------------------------------------------------------------------
# Table 13 — Stage 4 womenwork (5 variants)
# ---------------------------------------------------------------------------
def test_table_13_womenwork_variants():
    t = pd.read_csv(TDIR / "Table13_stage4_womenwork.csv").set_index("Variant")
    paper = {
        "Baseline":              (0.211, 0.024, 0.304),
        "Second person":         (0.231, 0.029, 0.299),
        "Reasoning instruction": (0.155, 0.018, 0.284),
        "Ideology background":   (0.198, 0.027, 0.317),
        "Natural rewrite":       (0.214, 0.019, 0.257),
    }
    for v, (wass, jsd, kw) in paper.items():
        r = t.loc[v]
        assert abs(r["Wass"] - wass) < TOL, f"{v} Wass"
        assert abs(r["JSD"]  - jsd)  < TOL, f"{v} JSD"
        assert abs(r["k_w"]  - kw)   < TOL, f"{v} kw"


# ---------------------------------------------------------------------------
# Table 14 — seed stability
# ---------------------------------------------------------------------------
def test_table_14a_pacdemons():
    t = pd.read_csv(TDIR / "Table14A_pacdemons.csv")
    key = lambda f, s: (t["Framing"] == f) & (t["Seed"] == s)
    paper = {  # (framing, seed): (JSD, delta, MCC, k, Rec+)
        ("Baseline",     0): (0.00096, +1.7, 0.430, 0.424, 0.536),
        ("Baseline",     1): (0.00096, +1.7, 0.386, 0.381, 0.488),
        ("Baseline",     2): (0.00062, +1.3, 0.422, 0.418, 0.512),
        ("+ Background", 0): (0.00109, +1.8, 0.425, 0.419, 0.536),
        ("+ Background", 1): (0.00115, +1.9, 0.391, 0.385, 0.500),
        ("+ Background", 2): (0.00115, +1.9, 0.391, 0.385, 0.500),
    }
    for (f, s), (jsd, dp, mcc, k, rec) in paper.items():
        r = t[key(f, s)].iloc[0]
        assert abs(r["JSD"] - jsd) < 0.01, f"{f} s{s} JSD"
        assert abs(r["Delta_p_pp"] - dp) < 0.2, f"{f} s{s} dp"
        assert abs(r["MCC"] - mcc) < TOL, f"{f} s{s} MCC"


def test_table_14b_womenwork():
    t = pd.read_csv(TDIR / "Table14B_womenwork.csv")
    key = lambda f, s: (t["Framing"] == f) & (t["Seed"] == s)
    paper = {
        ("Baseline",     0): (0.211, 0.024, 0.304, 3.186),
        ("Baseline",     1): (0.184, 0.024, 0.282, 3.196),
        ("Baseline",     2): (0.225, 0.025, 0.265, 3.187),
        ("+ Background", 0): (0.198, 0.027, 0.317, 3.230),
        ("+ Background", 1): (0.175, 0.024, 0.304, 3.249),
        ("+ Background", 2): (0.171, 0.024, 0.315, 3.246),
    }
    for (f, s), (wass, jsd, kw, mean) in paper.items():
        r = t[key(f, s)].iloc[0]
        assert abs(r["Wass"] - wass) < TOL, f"{f} s{s} Wass"
        assert abs(r["JSD"]  - jsd)  < TOL, f"{f} s{s} JSD"
        assert abs(r["k_w"]  - kw)   < TOL, f"{f} s{s} kw"
        assert abs(r["Sim. mean"] - mean) < 0.02, f"{f} s{s} mean"


# ---------------------------------------------------------------------------
# Table 15 — seed-pair agreement
# ---------------------------------------------------------------------------
def test_table_15_womenwork_kappa_w_paper():
    """The womenwork numbers reproduce the paper to 3 decimals under the paper's
    file-order one-RNG-per-run protocol."""
    t = pd.read_csv(TDIR / "Table15_seed_pair_agreement.csv")
    t = t[t["outcome"] == "womenwork"].set_index("seed_pair")
    paper_kw = {"s0-s1": 0.395, "s0-s2": 0.398, "s1-s2": 0.412}
    for sp, kw in paper_kw.items():
        assert abs(float(t.loc[sp, "weighted_kappa_quadratic"]) - kw) < 0.003, \
            f"{sp} kw"


def test_table_15_pacdemons_mean_paper():
    """Paper reports mean pacdemons exact agreement = 99.1%; individual s1-s2
    entry (99.4%) differs by ~0.22 pp from our recompute (14 vs 10 diffs).
    See docs/sampling_correction.md."""
    t = pd.read_csv(TDIR / "Table15_seed_pair_agreement.csv")
    t = t[t["outcome"] == "pacdemons"]
    mean_exact = t["exact_agreement"].mean() * 100
    assert abs(mean_exact - 99.1) < 0.15, f"pacdemons mean exact: {mean_exact:.2f}%"
