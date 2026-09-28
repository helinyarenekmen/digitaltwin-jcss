# C7starthimmig — leakage-guarded thimmig probe

## Purpose
Predict `thimmig` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `thimmig` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `thimmig` — Immigrants as perceived threat to Turkey (0–10 ordinal)
- **Valid range:** 0–10
- **Minority code:** n/a (ordinal)

## Protocol
- Config: **C7starthimmig** (C7 groups minus `thimmig`)
- VS-CoT · gpt-5.4-mini · +PC · s0
- Matched respondents: 2,578

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 2578 |
| jsd | 0.4870 |
| wasserstein | 1.2859 |
| tvd | 0.7199 |
| pred_mean | 8.3991 |
| gt_mean | 8.5617 |
| prevalence_error | 0.1625 |
| cohen_kappa | 0.0008 |
| weighted_kappa_quadratic | 0.0610 |
