# C7starpaccult — leakage-guarded paccult probe

## Purpose
Predict `paccult` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `paccult` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `paccult` — Past-year: volunteering in social/cultural activities
- **Valid range:** 1–2
- **Minority code:** 1

## Protocol
- Config: **C7starpaccult** (C7 groups minus `paccult`)
- Direct · gpt-4o-mini · +PC · s0
- Matched respondents: 1,713

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 1713 |
| jsd | 0.0002 |
| wasserstein | 0.0123 |
| tvd | 0.0123 |
| pred_mean | 1.8471 |
| gt_mean | 1.8593 |
| prevalence_error | 0.0123 |
| cohen_kappa | 0.4572 |
| weighted_kappa_quadratic | 0.4572 |
| mcc | 0.4578 |
| recall_minority | 0.5602 |
| precision_minority | 0.5153 |
| f1_minority | 0.5368 |
| macro_f1 | 0.7285 |
