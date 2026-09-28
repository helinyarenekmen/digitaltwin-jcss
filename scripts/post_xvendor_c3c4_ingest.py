"""
Post-processing script — ingest C3 & C4 cross-vendor womenwork cells into
calibration_final/ (raw + metrics + Excel).

Run this AFTER the 12 cross-vendor cells finish in
~/Library/Caches/digitaltwin_calibration/screening_{T08,vs_cot_T08}/womenwork/.

Vendors: gpt-5.4-mini, Gemini FL, Llama 3.3 70B  (Claude excluded).
Configs:  C3, C4
Samplings: direct, vs (vs_cot cache alias)

Actions:
  1. Copy each new JSONL to stages/stage_3_cross_vendor/raw_jsonl/womenwork/
  2. Compute metrics per cell
  3. Append rows to the correct stage_3 metrics CSV
     (e.g. gpt54mini_direct_T08_metrics.csv)
  4. Append rows to per_outcome/womenwork/womenwork_all_phases.xlsx
     with Phase (7/9/10) + Stage 3 labels

Idempotent: if a JSONL is already ingested (config+sampling+vendor row already
in the metrics CSV), it is skipped.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import cohen_kappa_score

ROOT = Path("/Users/helinekmen/Desktop/digitaltwin")
CACHE = Path.home() / "Library/Caches/digitaltwin_calibration"
FINAL = ROOT / "outputs/calibration_final"
GT_CSV = ROOT / "agent-survey/agent-calibration/womenwork/womenwork-data/womenwork_data_output.csv"

VENDORS = [
    # (alias_in_filename, human_model_name, source_csv_stem, phase_label)
    ("gpt54mini",       "gpt-5.4-mini",                       "gpt54mini",       "Phase 7"),
    ("geminiflashlite", "gemini-2.5-flash-lite",              "geminiflashlite", "Phase 9"),
    ("llama3370b",      "meta-llama/llama-3.3-70b-instruct",  "llama3370b",      "Phase 10"),
]
CONFIGS = ["C3", "C4"]
SAMPLINGS = [
    # (excel_sampling, cache_sampling, filename_sampling)
    # Stage 2 decision: VS-only for womenwork Likert — direct cells intentionally excluded
    # even if they exist in the cache (accidental gpt-5.4-mini direct run 2026-08-02).
    ("vs",     "vs_cot", "vs_cot"),
]
TEMPERATURE = 0.8
STAGE = "Stage 3 — Cross-vendor"


# ---- helpers ----
def load_gt():
    df = pd.read_csv(GT_CSV)
    df["rid"] = df["persona_id"].astype(str).str.zfill(4).apply(lambda x: f"TGSS_{x}")
    return dict(zip(df["rid"], df["gt_womenwork"].astype(int)))


def load_preds(jsonl: Path) -> dict[str, int]:
    out = {}
    for line in jsonl.open(encoding="utf-8"):
        try: r = json.loads(line)
        except: continue
        if r.get("parse_status") == "ok" and r.get("predicted_value") is not None:
            out[r["respondent_id"]] = int(r["predicted_value"])
    return out


def dist(vals, lo, hi):
    counts = np.zeros(hi - lo + 1)
    for v in vals:
        if lo <= v <= hi: counts[v - lo] += 1
    return counts / max(counts.sum(), 1)


def compute_metrics(jsonl: Path, gts: dict) -> dict:
    preds = load_preds(jsonl)
    pairs = [(gts[r], preds[r]) for r in preds if r in gts]
    if not pairs:
        return None
    gt = np.array([p[0] for p in pairs])
    pr = np.array([p[1] for p in pairs])
    labels = [1, 2, 3, 4, 5]
    gt_p = dist(gt, 1, 5); pr_p = dist(pr, 1, 5)
    return dict(
        n_matched=len(pairs),
        wasserstein=float(wasserstein_distance([1,2,3,4,5], [1,2,3,4,5], gt_p, pr_p)),
        jsd=float(jensenshannon(gt_p, pr_p, base=2) ** 2),
        tvd=float(np.abs(gt_p - pr_p).sum() / 2),
        pred_mean=float(pr.mean()), gt_mean=float(gt.mean()),
        prevalence_error=float(abs(gt.mean() - pr.mean())),
        cohen_kappa=float(cohen_kappa_score(gt, pr, labels=labels)),
        weighted_kappa_quadratic=float(cohen_kappa_score(gt, pr, labels=labels, weights="quadratic")),
        weighted_kappa_linear=float(cohen_kappa_score(gt, pr, labels=labels, weights="linear")),
    )


def ingest():
    gts = load_gt()
    stage3_raw = FINAL / "stages/stage_3_cross_vendor/raw_jsonl/womenwork"
    stage3_metrics = FINAL / "stages/stage_3_cross_vendor/metrics"
    stage3_raw.mkdir(parents=True, exist_ok=True)

    excel_fp = FINAL / "per_outcome/womenwork/womenwork_all_phases.xlsx"
    wb = openpyxl.load_workbook(excel_fp)
    ws = wb["All cells"]
    header = [c.value for c in ws[1]]

    new_excel_rows = []
    copied = 0
    metrics_appended = 0

    for alias, model_name, csv_stem, phase_label in VENDORS:
        for cfg in CONFIGS:
            for excel_samp, cache_samp, fname_samp in SAMPLINGS:
                # Source JSONL
                subdir = "screening_T08" if fname_samp == "direct" else "screening_vs_cot_T08"
                cell_fname = f"{cfg}_{alias}_T08_ben_{fname_samp}_nocot_s0.jsonl"
                src_jsonl = CACHE / subdir / "womenwork" / cell_fname
                if not src_jsonl.exists():
                    print(f"  ⚠ MISSING: {cell_fname}")
                    continue

                # (1) copy raw
                dst_jsonl = stage3_raw / cell_fname
                if not dst_jsonl.exists() or dst_jsonl.stat().st_size != src_jsonl.stat().st_size:
                    shutil.copy2(src_jsonl, dst_jsonl); copied += 1

                # (2) compute metrics
                m = compute_metrics(src_jsonl, gts)
                if m is None:
                    print(f"  ⚠ no matched pairs: {cell_fname}"); continue

                # (3) append to stage_3 metrics CSV
                csv_name = f"{csv_stem}_{excel_samp}_T08_metrics.csv"
                csv_fp = stage3_metrics / csv_name
                new_row = dict(
                    outcome="womenwork", config_id=cfg, step="2A_screening",
                    model=model_name, temperature=TEMPERATURE, address_mode="ben",
                    sampling=cache_samp, cot=False, seed=0,
                    n_total=m["n_matched"], n_matched=m["n_matched"],
                    n_parse_failures=0, n_api_errors=0,
                    jsd=m["jsd"], wasserstein=m["wasserstein"], tvd=m["tvd"],
                    pred_mean=m["pred_mean"], gt_mean=m["gt_mean"],
                    prevalence_diff=m["pred_mean"] - m["gt_mean"],
                    prevalence_error=m["prevalence_error"],
                    precision_minority=None, recall_minority=None, f1_minority=None,
                    macro_f1=None, mcc=None, cohen_kappa=m["cohen_kappa"],
                    composite_score=None,
                    timestamp=pd.Timestamp.utcnow().isoformat(),
                )
                if csv_fp.exists():
                    df = pd.read_csv(csv_fp)
                    mask = ((df["outcome"] == "womenwork") & (df["config_id"] == cfg)
                            & (df["sampling"] == cache_samp))
                    if mask.any():
                        print(f"    {csv_name}: {cfg} already in file, skipping")
                    else:
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        df.to_csv(csv_fp, index=False)
                        metrics_appended += 1
                        print(f"    ✓ {csv_name}: appended {cfg}")
                else:
                    pd.DataFrame([new_row]).to_csv(csv_fp, index=False)
                    metrics_appended += 1
                    print(f"    ✓ {csv_name}: created with {cfg}")

                # (4) append to Excel
                excel_row = {
                    "Phase": phase_label, "Stage": STAGE,
                    "Section": {"Phase 7": "S07", "Phase 9": "S09",
                                "Phase 10": "S10"}[phase_label],
                    "Experiment": f"{model_name} {excel_samp.upper()} T=0.8 ({cfg}) xvendor",
                    "Config": cfg, "Model": model_name,
                    "Sampling": excel_samp, "Temperature": TEMPERATURE,
                    "Address mode": "ben", "CoT": False, "Seed": 0,
                    "n_matched": m["n_matched"],
                    "wasserstein": m["wasserstein"], "jsd": m["jsd"],
                    "tvd": m["tvd"], "pred_mean": m["pred_mean"],
                    "gt_mean": m["gt_mean"], "prevalence_error": m["prevalence_error"],
                    "Notes": (f"Cross-vendor extension: {cfg} added to Stage 3 to test "
                              "whether Stage 2 winners (C3, C4) match C7's cross-vendor "
                              "performance."),
                    "Source_CSV": f"stages/stage_3_cross_vendor/metrics/{csv_name}",
                    "cohen_kappa": m["cohen_kappa"],
                    "weighted_kappa_quadratic": m["weighted_kappa_quadratic"],
                    "weighted_kappa_linear": m["weighted_kappa_linear"],
                    "ac2_quadratic": None,
                }
                new_excel_rows.append(excel_row)

    # Check for excel duplicates before appending
    existing = set()
    for row in ws.iter_rows(min_row=2, values_only=True):
        rec = dict(zip(header, row))
        existing.add((rec.get("Config"), rec.get("Model"), rec.get("Sampling"),
                      rec.get("Temperature"), rec.get("Seed"), rec.get("Phase")))

    excel_added = 0
    for row in new_excel_rows:
        key = (row["Config"], row["Model"], row["Sampling"],
               row["Temperature"], row["Seed"], row["Phase"])
        if key in existing:
            print(f"    Excel: {key} already present, skipping")
            continue
        ws.append([row.get(h) for h in header])
        excel_added += 1

    wb.save(excel_fp)

    print()
    print(f"✓ SUMMARY")
    print(f"  Raw JSONL copied to stage_3: {copied}")
    print(f"  Metrics CSV rows appended:   {metrics_appended}")
    print(f"  Excel rows added:            {excel_added}  →  {excel_fp.relative_to(ROOT)}")


if __name__ == "__main__":
    ingest()
