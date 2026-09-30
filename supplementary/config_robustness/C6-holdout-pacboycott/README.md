# C7starpacboycott — leakage-guarded pacboycott probe

## Purpose
Predict `pacboycott` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `pacboycott` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `pacboycott` — Past-year participation: boycotting products
- **Valid range:** 1–2
- **Minority code:** 1

## Protocol
- Config: **C7starpacboycott** (C7 groups minus `pacboycott`)
- Sampling: direct
- Temperature: 0.8
- Model: gpt-4o-mini
- Address mode: ben
- Political-context prepend: yes
- Seed: 0
- CoT: no
- Matched respondents: 1,692

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 1692 |
| jsd | 0.0161 |
| wasserstein | 0.1454 |
| tvd | 0.1454 |
| pred_mean | 1.5366 |
| gt_mean | 1.6820 |
| prevalence_error | 0.1454 |
| cohen_kappa | -0.0396 |
| weighted_kappa_quadratic | -0.0396 |
| mcc | -0.0414 |
| recall_minority | 0.4331 |
| precision_minority | 0.2972 |
| f1_minority | 0.3525 |
| macro_f1 | 0.4687 |
