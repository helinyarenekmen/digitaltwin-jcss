# C7starfamroles — leakage-guarded famroles probe

## Purpose
Predict `famroles` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `famroles` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `famroles` — Men earn, women home-and-family (Likert 1-5)
- **Valid range:** 1–5
- **Minority code:** n/a (ordinal)

## Protocol
- Config: **C7starfamroles** (C7 groups minus `famroles`)
- VS-CoT · gpt-5.4-mini · +PC · s0
- Matched respondents: 2,609

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 2609 |
| jsd | 0.0385 |
| wasserstein | 0.3990 |
| tvd | 0.1890 |
| pred_mean | 3.2031 |
| gt_mean | 2.8777 |
| prevalence_error | 0.3254 |
| cohen_kappa | 0.2047 |
| weighted_kappa_quadratic | 0.5302 |
