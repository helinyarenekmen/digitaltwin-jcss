"""
11_multi_seed_analysis.py — Multi-seed stability analysis for finalist combinations.

5 finalist × 3 seed (s0 baseline + s1 + s2) için mean ± std hesaplar.
Çıktı: outputs/multi_seed_synthesis.{md,csv}

Usage:
  python3 scripts/11_multi_seed_analysis.py
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import CACHE_DIR as _CACHE, PERSONA_DIR as _PERSONA
import pandas as pd
import numpy as np

CALIB = Path(str(_CACHE))
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# 5 finalist'in tanımı: (label, outcome, config, model, sampling, T, primary_metric, secondary)
FINALISTS = [
    ("F1: pacdemons C7 4o-mini Direct T=0.8",
     "pacdemons", "C7", "gpt-4o-mini", "direct", 0.8, "mcc", "jsd"),
    ("F2: womenwork C4 4o-mini VS-CoT T=0.4",
     "womenwork", "C4", "gpt-4o-mini", "vs_cot", 0.4, "wasserstein", "jsd"),
    ("F3: womenwork C7 Gemini VS-CoT T=0.8",
     "womenwork", "C7", "gemini-2.5-flash-lite", "vs_cot", 0.8, "wasserstein", "jsd"),
    ("F4: neilang C7 4o-mini Direct T=0 (determinism check)",
     "neilang", "C7", "gpt-4o-mini", "direct", 0.0, "wasserstein", "jsd"),
    ("F5: neilang C7 Llama VS-CoT T=0.8",
     "neilang", "C7", "meta-llama/llama-3.3-70b-instruct", "vs_cot", 0.8, "wasserstein", "jsd"),
]

_MODEL_ALIASES = {
    "gpt-4o-mini": "gpt4omini",
    "gemini-2.5-flash-lite": "geminiflashlite",
    "meta-llama/llama-3.3-70b-instruct": "llama3370b",
}
_TEMP_LABELS = {0.0: "T0", 0.4: "T04", 0.8: "T08"}


def csv_path_for(model: str, sampling: str, temperature: float, seed: int) -> Path:
    """Replicate the 05_compute_metrics.py suffix logic to find CSV files."""
    temp_str = _TEMP_LABELS[temperature]
    seed_suffix = "" if seed == 0 else f"_s{seed}"
    model_alias = _MODEL_ALIASES[model]
    model_suffix = "" if model == "gpt-4o-mini" else f"_{model_alias}"

    if sampling == "direct" and temperature == 0.0:
        suffix = model_suffix + seed_suffix
    elif sampling == "direct":
        suffix = f"_{temp_str}" + model_suffix + seed_suffix
    elif temperature == 0.0:
        suffix = "_vs_cot" + model_suffix + seed_suffix
    else:
        suffix = f"_vs_cot_{temp_str}" + model_suffix + seed_suffix
    return CALIB / f"master_results{suffix}.csv"


def load_finalist_row(outcome, config, model, sampling, temperature, seed):
    p = csv_path_for(model, sampling, temperature, seed)
    if not p.exists():
        return None, str(p)
    df = pd.read_csv(p)
    sub = df[(df.outcome == outcome) & (df.config_id == config)]
    if len(sub) == 0:
        return None, str(p)
    return sub.iloc[0], str(p)


def main():
    rows = []
    print("Multi-seed analysis — 5 finalists × 3 seeds")
    print("=" * 80)

    for entry in FINALISTS:
        label, outcome, config, model, sampling, T, primary, secondary = entry
        print(f"\n{label}")
        primary_vals = []
        secondary_vals = []
        for seed in [0, 1, 2]:
            r, p = load_finalist_row(outcome, config, model, sampling, T, seed)
            if r is None:
                print(f"  seed={seed}: MISSING ({p})")
                continue
            pv = r[primary]
            sv = r[secondary]
            primary_vals.append(pv)
            secondary_vals.append(sv)
            print(f"  seed={seed}: {primary}={pv:.4f}  {secondary}={sv:.4f}")
        if len(primary_vals) < 2:
            continue
        arr = np.array(primary_vals)
        sec_arr = np.array(secondary_vals)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        cv = std / abs(mean) * 100 if mean != 0 else 0.0
        if cv < 1:
            verdict = "✅ Highly stable"
        elif cv < 5:
            verdict = "✓ Stable"
        elif cv < 10:
            verdict = "⚠ Moderate variance"
        else:
            verdict = "❌ High variance"
        rows.append({
            "finalist": label,
            "outcome": outcome,
            "config": config,
            "model": model,
            "sampling": sampling,
            "temperature": T,
            "primary_metric": primary,
            "n_seeds": len(primary_vals),
            "seeds_tested": "0,1,2"[:2*len(primary_vals)-1],
            "primary_seed_0": primary_vals[0] if len(primary_vals) > 0 else None,
            "primary_seed_1": primary_vals[1] if len(primary_vals) > 1 else None,
            "primary_seed_2": primary_vals[2] if len(primary_vals) > 2 else None,
            "primary_mean": mean,
            "primary_std": std,
            "primary_cv_pct": cv,
            "secondary_metric": secondary,
            "secondary_mean": float(sec_arr.mean()),
            "secondary_std": float(sec_arr.std(ddof=1)) if len(sec_arr) > 1 else 0.0,
            "stability_verdict": verdict,
        })
        print(f"  → mean={mean:.4f}, std={std:.4f}, CV={cv:.2f}% — {verdict}")

    # Write CSV
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "multi_seed_synthesis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV: {csv_path}")

    # Write Markdown
    md = []
    md.append("# Multi-Seed Stability Analysis (5 finalists × 3 seeds)\n")
    md.append("Each finalist combination was run with seeds {0, 1, 2}. Reported metric is the primary metric for the outcome (MCC for pacdemons binary, Wasserstein for ordinal). Stability assessed via Coefficient of Variation (CV = std/mean).\n")
    md.append("## Summary table\n")
    md.append("| Finalist | Primary metric | s0 | s1 | s2 | Mean | Std | CV % | Verdict |")
    md.append("|----------|----------------|------|------|------|------|-----|------|---------|")
    for r in rows:
        s0 = f"{r['primary_seed_0']:.4f}" if r['primary_seed_0'] is not None else "—"
        s1 = f"{r['primary_seed_1']:.4f}" if r['primary_seed_1'] is not None else "—"
        s2 = f"{r['primary_seed_2']:.4f}" if r['primary_seed_2'] is not None else "—"
        md.append(f"| {r['finalist']} | {r['primary_metric']} | {s0} | {s1} | {s2} | "
                  f"{r['primary_mean']:.4f} | {r['primary_std']:.4f} | "
                  f"{r['primary_cv_pct']:.2f}% | {r['stability_verdict']} |")
    md.append("")
    md.append("## Stability verdict thresholds\n")
    md.append("- **CV < 1%** → highly stable (deterministic-like)")
    md.append("- **CV 1–5%** → stable (single-seed estimates reliable)")
    md.append("- **CV 5–10%** → moderate variance (report mean ± std in paper)")
    md.append("- **CV > 10%** → high variance (more seeds needed)")
    md.append("")
    md.append("## Interpretation\n")
    # Generate dynamic interpretation
    for r in rows:
        md.append(f"### {r['finalist']}")
        md.append(f"- **Primary ({r['primary_metric']}):** {r['primary_mean']:.4f} ± {r['primary_std']:.4f} (CV={r['primary_cv_pct']:.2f}%)")
        md.append(f"- **Secondary ({r['secondary_metric']}):** {r['secondary_mean']:.4f} ± {r['secondary_std']:.4f}")
        md.append(f"- **Stability:** {r['stability_verdict']}")
        md.append("")

    md_path = OUT_DIR / "multi_seed_synthesis.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
