"""
05_compute_metrics.py — Step 2B: compute screening metrics and select shortlist.

Reads all Step 2A JSONL files, computes per-cell metrics, writes:
  outputs/calibration/master_results.csv    — one row per (outcome × config)
  outputs/calibration/screening_ranking.csv — config ranks, shortlist flags
  outputs/calibration/shortlist.json        — configs for Step 2C

Usage:
  python scripts/05_compute_metrics.py
  python scripts/05_compute_metrics.py --check_only   # QA gates only, no write

See CALIBRATION_SPEC.md §7 (metrics) and §5 (shortlist selection).
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import CACHE_DIR as _CACHE, PERSONA_DIR as _PERSONA

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV_PATH      = ROOT / "data" / "tgss2024_clean.csv"
CALIB_DIR     = Path(str(_CACHE))

# Valid response ranges per outcome (matching 04_run_calibration.py)
OUTCOME_RANGES: dict[str, tuple[int, int]] = {
    "pacdemons": (1, 2),
    "womenwork": (1, 5),
    "neilang":   (1, 3),
}
OUTCOME_MINORITY = {"pacdemons": 1}  # minority class code for MCC etc.

# Ranking primary metric per outcome (lower = better)
RANK_METRIC = {
    "pacdemons": "composite_score",
    "womenwork": "wasserstein",
    "neilang":   "wasserstein",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def zscore_series(s: pd.Series) -> pd.Series:
    std = s.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def dist_array(vals: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Proportion array over [lo..hi] from integer sample array."""
    categories = np.arange(lo, hi + 1)
    counts = np.array([(vals == c).sum() for c in categories], dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def compute_cell_metrics(
    pred_vals: np.ndarray,
    gt_vals: np.ndarray,
    outcome: str,
    n_parse_failures: int,
    n_api_errors: int,
) -> dict:
    lo, hi = OUTCOME_RANGES[outcome]
    categories = list(range(lo, hi + 1))
    n = len(pred_vals)

    pred_dist = dist_array(pred_vals, lo, hi)
    gt_dist   = dist_array(gt_vals,   lo, hi)

    # --- Distributional metrics ---
    # scipy.jensenshannon returns sqrt(JSD_base2); we store the squared value.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jsd_sqrt = float(jensenshannon(pred_dist, gt_dist, base=2))
    jsd = jsd_sqrt ** 2  # range [0, 1] in bits

    # 1-Wasserstein on integer categories with frequency weights
    wass = float(wasserstein_distance(
        categories, categories, u_weights=pred_dist, v_weights=gt_dist
    ))

    # Total Variation Distance = 0.5 * L1
    tvd = float(np.sum(np.abs(pred_dist - gt_dist)) / 2)

    # Prevalence / marginal mean
    pred_mean = float(np.mean(pred_vals))
    gt_mean   = float(np.mean(gt_vals))
    prev_diff = pred_mean - gt_mean  # signed (sycophancy diagnostic)

    row: dict = dict(
        n_matched=n,
        n_parse_failures=n_parse_failures,
        n_api_errors=n_api_errors,
        jsd=jsd,
        wasserstein=wass,
        tvd=tvd,
        pred_mean=pred_mean,
        gt_mean=gt_mean,
        prevalence_diff=prev_diff,
        prevalence_error=abs(prev_diff),
        # placeholders for pacdemons-specific metrics (filled below or left NaN)
        precision_minority=np.nan,
        recall_minority=np.nan,
        f1_minority=np.nan,
        macro_f1=np.nan,
        mcc=np.nan,
        cohen_kappa=np.nan,
        weighted_kappa_quadratic=np.nan,
        weighted_kappa_linear=np.nan,
        composite_score=np.nan,  # filled in second pass
    )

    # --- Cohen's κ variants (works for every outcome; quadratic weighting is
    #     the standard for ordinal Likert / ordinal categorical outcomes such
    #     as womenwork and neilang. For binary pacdemons the weighted variants
    #     collapse to the unweighted κ, which is harmless but not additionally
    #     informative). Requires ≥ 2 distinct categories in both gt and pred;
    #     otherwise sklearn raises, so we guard and leave NaN. ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(np.unique(gt_vals)) >= 2 and len(np.unique(pred_vals)) >= 2:
            row["cohen_kappa"] = float(cohen_kappa_score(gt_vals, pred_vals))
            row["weighted_kappa_quadratic"] = float(
                cohen_kappa_score(gt_vals, pred_vals, weights="quadratic")
            )
            row["weighted_kappa_linear"] = float(
                cohen_kappa_score(gt_vals, pred_vals, weights="linear")
            )

    # --- pacdemons individual-level metrics ---
    if outcome == "pacdemons":
        minority = OUTCOME_MINORITY["pacdemons"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mcc  = float(matthews_corrcoef(gt_vals, pred_vals))
            prec = float(precision_score(gt_vals, pred_vals, pos_label=minority, zero_division=0))
            rec  = float(recall_score(gt_vals, pred_vals, pos_label=minority, zero_division=0))
            f1m  = float(f1_score(gt_vals, pred_vals, pos_label=minority, zero_division=0))
            maf1 = float(f1_score(gt_vals, pred_vals, average="macro", zero_division=0))
        row.update(
            mcc=mcc,
            precision_minority=prec,
            recall_minority=rec,
            f1_minority=f1m,
            macro_f1=maf1,
        )

    return row


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

_TEMP_LABELS = {0.0: "T0", 0.3: "T03", 0.4: "T04", 0.7: "T07", 0.8: "T08", 1.0: "T1"}


_MODEL_ALIASES = {
    "gpt-4o-mini": "gpt4omini",
    "gpt-5-mini":  "gpt5mini",
    "gpt-5.4-mini": "gpt54mini",
    "claude-haiku": "claudehaiku",
    "claude-haiku-4-5-20251001": "claudehaiku",
    "gemini-2.5-flash-lite": "geminiflashlite",
    "gemini-2.5-flash":      "geminiflash",
    "gemini-2.5-pro":        "geminipro",
    "meta-llama/llama-3.3-70b-instruct": "llama3370b",
    "meta-llama/llama-3.1-70b-instruct": "llama3170b",
    "meta-llama/llama-3.1-8b-instruct":  "llama318b",
}


def run(variant: str = "direct", temperature: float = 0.0,
        seed: int = 0, address_mode: str = "ben",
        model: str = "gpt-4o-mini",
        cot: bool = False,
        political_context: bool = False,
        check_only: bool = False) -> None:
    """variant: 'direct' (Step 2A) or 'vs_cot' (verbalized sampling screening)."""
    temp_str = _TEMP_LABELS.get(temperature, f"T{temperature}".replace(".", ""))
    seed_suffix = "" if seed == 0 else f"_s{seed}"
    addr_suffix = "" if address_mode == "ben" else f"_{address_mode}"
    model_alias = _MODEL_ALIASES.get(model, model.replace("-", "").replace(".", ""))
    model_suffix = "" if model == "gpt-4o-mini" else f"_{model_alias}"
    cot_token = "cot" if cot else "nocot"
    cot_suffix = "_cot" if cot else ""
    pc_filename_token = "_pc" if political_context else ""
    pc_suffix = "_pc" if political_context else ""
    if variant == "direct" and temperature == 0.0:
        screening_dir = CALIB_DIR / "screening"
        suffix = addr_suffix + model_suffix + cot_suffix + pc_suffix + seed_suffix
    elif variant == "direct":
        screening_dir = CALIB_DIR / f"screening_{temp_str}"
        suffix = f"_{temp_str}" + addr_suffix + model_suffix + cot_suffix + pc_suffix + seed_suffix
    elif temperature == 0.0:
        screening_dir = CALIB_DIR / "screening_vs_cot"
        suffix = "_vs_cot" + addr_suffix + model_suffix + cot_suffix + pc_suffix + seed_suffix
    else:
        screening_dir = CALIB_DIR / f"screening_vs_cot_{temp_str}"
        suffix = f"_vs_cot_{temp_str}" + addr_suffix + model_suffix + cot_suffix + pc_suffix + seed_suffix

    if not screening_dir.exists():
        print(f"ERROR: {screening_dir} not found.")
        sys.exit(1)

    print(f"=== variant={variant} | reading from {screening_dir} ===")
    print("Loading ground truth...")
    df_gt = pd.read_csv(CSV_PATH, encoding="utf-8")
    df_gt["respondent_id"] = df_gt["id"].apply(lambda x: f"TGSS_{int(x):04d}")

    from config.ablations import CONFIGS
    config_ids = [c.config_id for c in CONFIGS]
    outcomes   = list(OUTCOME_RANGES.keys())

    rows: list[dict] = []
    missing_cells: list[str] = []

    for outcome in outcomes:
        lo, hi = OUTCOME_RANGES[outcome]
        gt_valid = df_gt[df_gt[outcome].between(lo, hi)][["respondent_id", outcome]].copy()
        gt_valid = gt_valid.rename(columns={outcome: "gt_value"})
        print(f"\n[{outcome}] Ground truth: {len(gt_valid)} valid respondents")

        out_dir = screening_dir / outcome
        for config_id in config_ids:
            sampling_token = "direct" if variant == "direct" else "vs_cot"
            jsonl_path = out_dir / f"{config_id}_{model_alias}_{temp_str}_{address_mode}_{sampling_token}_{cot_token}{pc_filename_token}_s{seed}.jsonl"
            pred_df = load_jsonl(jsonl_path)

            if pred_df.empty:
                print(f"  [{config_id}] MISSING — no JSONL found at {jsonl_path.name}")
                missing_cells.append(f"{outcome}/{config_id}")
                continue

            n_total        = len(pred_df)
            n_api_errors   = int((pred_df.get("parse_status", pd.Series()) == "api_error").sum())
            n_parse_fail   = int((pred_df.get("parse_status", pd.Series()) == "parse_failure").sum())
            pred_ok        = pred_df[pred_df["parse_status"] == "ok"][["respondent_id", "predicted_value"]]

            # Inner join: only respondents with valid GT AND valid prediction
            merged = gt_valid.merge(pred_ok, on="respondent_id", how="inner")
            n_matched = len(merged)

            if n_matched == 0:
                print(f"  [{config_id}] EMPTY after merge — no matched predictions")
                missing_cells.append(f"{outcome}/{config_id} (empty merge)")
                continue

            pred_vals = merged["predicted_value"].astype(int).values
            gt_vals   = merged["gt_value"].astype(int).values

            metrics = compute_cell_metrics(pred_vals, gt_vals, outcome, n_parse_fail, n_api_errors)

            pct_fail = (n_parse_fail + n_api_errors) / n_total * 100 if n_total else 0
            flag = " ⚠ HIGH FAIL RATE" if pct_fail > 2 else ""
            print(
                f"  [{config_id}] n={n_matched:,} | "
                f"wass={metrics['wasserstein']:.4f} | "
                f"jsd={metrics['jsd']:.4f} | "
                f"mcc={metrics['mcc']:.3f} | "
                f"fail={pct_fail:.1f}%{flag}"
            )

            # JSONL'deki ilk satırdan gerçek inference metadata'sını oku
            # (hardcode etmek vs_cot/T=0.8 cell'lerinde yalan üretiyordu).
            sample_row = pred_df.iloc[0]
            rows.append(dict(
                outcome=outcome,
                config_id=config_id,
                step="2A_screening",
                model=sample_row.get("model", "gpt-4o-mini"),
                temperature=float(sample_row.get("temperature", 0.0)),
                address_mode=sample_row.get("address_mode", "ben"),
                sampling=sample_row.get("sampling", "direct"),
                cot=bool(sample_row.get("cot", False)),
                seed=int(sample_row.get("seed", 0)),
                n_total=n_total,
                **metrics,
                timestamp=datetime.now().isoformat(),
            ))

    if not rows:
        print("\nERROR: No cells computed — nothing to write.")
        sys.exit(1)

    master_df = pd.DataFrame(rows)

    # --- Second pass: composite z-score for pacdemons ---
    pac_mask = master_df["outcome"] == "pacdemons"
    if pac_mask.sum() > 1:
        pac = master_df.loc[pac_mask].copy()
        pac["z_jsd"]     = zscore_series(pac["jsd"])
        pac["z_prev_err"] = zscore_series(pac["prevalence_error"])
        pac["z_neg_mcc"] = zscore_series(-pac["mcc"])
        pac["composite_score"] = pac["z_jsd"] + pac["z_prev_err"] + pac["z_neg_mcc"]
        master_df.loc[pac_mask, "composite_score"] = pac["composite_score"].values

    # --- Ranking ---
    rank_rows: list[dict] = []
    for config_id in config_ids:
        rank_row: dict = {"config_id": config_id}
        for outcome in outcomes:
            subset = master_df[
                (master_df["outcome"] == outcome) & (master_df["config_id"] == config_id)
            ]
            if subset.empty:
                rank_row[f"{outcome}_metric"] = np.nan
                rank_row[f"{outcome}_rank"]   = np.nan
            else:
                rank_row[f"{outcome}_metric"] = float(subset[RANK_METRIC[outcome]].iloc[0])
        rank_rows.append(rank_row)

    rank_df = pd.DataFrame(rank_rows)
    for outcome in outcomes:
        metric_col = f"{outcome}_metric"
        rank_col   = f"{outcome}_rank"
        valid = rank_df[metric_col].notna()
        rank_df.loc[valid, rank_col] = (
            rank_df.loc[valid, metric_col]
            .rank(method="min", ascending=True)
            .astype(float)
        )

    rank_df["mean_rank"] = rank_df[[f"{o}_rank" for o in outcomes]].mean(axis=1)

    # Tie-break on pacdemons rank (most theoretically loaded)
    rank_df = rank_df.sort_values(
        ["mean_rank", "pacdemons_rank"], ascending=[True, True]
    ).reset_index(drop=True)

    # Shortlist: top 3 + C0 + C1 mandatory
    top3 = list(rank_df.head(3)["config_id"])
    mandatory = [c for c in ["C1", "C0"] if c not in top3]
    shortlist = top3 + mandatory
    shortlist = list(dict.fromkeys(shortlist))  # deduplicate preserving order

    rank_df["in_shortlist"] = rank_df["config_id"].isin(shortlist)

    print("\n" + "=" * 60)
    print("SCREENING RANKING")
    print("=" * 60)
    display_cols = ["config_id", "pacdemons_rank", "womenwork_rank", "neilang_rank",
                    "mean_rank", "in_shortlist"]
    print(rank_df[display_cols].to_string(index=False, float_format="%.2f"))

    # --- Quality gates ---
    print("\n--- Quality Gates ---")
    gates_pass = True

    # Gate 1: all 33 cells present
    expected = len(config_ids) * len(outcomes)
    actual = len(master_df)
    g1 = actual == expected and not missing_cells
    print(f"[{'PASS' if g1 else 'FAIL'}] Gate 1: all {expected} cells present ({actual} found)")
    if missing_cells:
        print(f"  Missing: {missing_cells}")
    if not g1:
        gates_pass = False

    # Gate 2: parse failure ≤2% per cell
    master_df["fail_rate"] = (master_df["n_parse_failures"] + master_df["n_api_errors"]) / master_df["n_total"]
    bad_cells = master_df[master_df["fail_rate"] > 0.02][["outcome", "config_id", "fail_rate"]]
    g2 = bad_cells.empty
    print(f"[{'PASS' if g2 else 'WARN'}] Gate 2: parse failures ≤2% per cell")
    if not bad_cells.empty:
        print(bad_cells.to_string(index=False))
    if not g2:
        gates_pass = False

    # Gate 3: shortlist size 3–5, C0+C1 present
    g3a = 3 <= len(shortlist) <= 5
    g3b = "C0" in shortlist and "C1" in shortlist
    print(f"[{'PASS' if g3a and g3b else 'FAIL'}] Gate 3: shortlist={shortlist}")
    if not (g3a and g3b):
        gates_pass = False

    # Gate 4: mode collapse alarm for pacdemons
    pac_cells = master_df[master_df["outcome"] == "pacdemons"]
    for _, r in pac_cells.iterrows():
        minority_n = int(r["n_matched"] * r["pred_mean"] / 2)  # rough: pred Evet ~pred_mean/2
    collapse_cells = pac_cells[pac_cells["precision_minority"] == 0]
    if not collapse_cells.empty:
        print(f"[WARN ] Gate 4: mode collapse in pacdemons: {list(collapse_cells['config_id'])}")
    else:
        print("[PASS ] Gate 4: no complete mode collapse in pacdemons")

    print(f"\nOverall: {'ALL GATES PASS' if gates_pass else 'SOME GATES FAILED — review before Step 2C'}")

    if check_only:
        print("\n(--check_only: no files written)")
        return

    # --- Write outputs (vs_cot results get _vs_cot suffix to keep direct untouched) ---
    CALIB_DIR.mkdir(parents=True, exist_ok=True)

    master_path = CALIB_DIR / f"master_results{suffix}.csv"
    master_df.drop(columns=["fail_rate"], errors="ignore").to_csv(master_path, index=False)
    print(f"\nWrote: {master_path}")

    rank_path = CALIB_DIR / f"screening_ranking{suffix}.csv"
    rank_df.to_csv(rank_path, index=False, float_format="%.4f")
    print(f"Wrote: {rank_path}")

    shortlist_payload = {
        "variant": variant,
        "shortlist": shortlist,
        "mandatory": ["C1", "C0"],
        "top_3_empirical": top3,
        "mean_ranks": {r["config_id"]: round(float(r["mean_rank"]), 4)
                       for _, r in rank_df.iterrows()},
        "generated_from": str(screening_dir),
        "missing_cells": missing_cells,
        "timestamp": datetime.now().isoformat(),
    }
    shortlist_path = CALIB_DIR / f"shortlist{suffix}.json"
    with open(shortlist_path, "w", encoding="utf-8") as f:
        json.dump(shortlist_payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {shortlist_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["direct", "vs_cot"], default="direct",
                        help="'direct' reads outputs/calibration/screening/; "
                             "'vs_cot' reads outputs/calibration/screening_vs_cot/")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature value to look up in filenames (default 0.0).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed value to look up in filenames (default 0).")
    parser.add_argument("--address_mode", choices=["ben", "sen"], default="ben",
                        help="Address mode in filenames (default ben).")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model in filenames (default gpt-4o-mini).")
    parser.add_argument("--cot", action="store_true",
                        help="Read explicit-CoT JSONL files (with _cot_ suffix).")
    parser.add_argument("--political_context", action="store_true",
                        help="Read political-context JSONL files (with _pc_ suffix).")
    parser.add_argument("--check_only", action="store_true",
                        help="Compute and print QA gates only; do not write output files.")
    args = parser.parse_args()
    run(variant=args.variant, temperature=args.temperature,
        seed=args.seed, address_mode=args.address_mode,
        model=args.model, cot=args.cot,
        political_context=args.political_context,
        check_only=args.check_only)


if __name__ == "__main__":
    main()
