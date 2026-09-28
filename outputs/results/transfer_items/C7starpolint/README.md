# C7starpolint — leakage-guarded polint probe

## Purpose
Predict `polint` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `polint` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `polint` — Interest in politics (1–4 ordinal)
- **Valid range:** 1–4
- **Minority code:** n/a (ordinal)

## Protocol
- Config: **C7starpolint** (C7 groups minus `polint`)
- VS-CoT · gpt-5.4-mini · +PC · s0
- Matched respondents: 1,683

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 1683 |
| jsd | 0.0462 |
| wasserstein | 0.3203 |
| tvd | 0.1931 |
| pred_mean | 2.1943 |
| gt_mean | 1.8824 |
| prevalence_error | 0.3119 |
| cohen_kappa | 0.2236 |
| weighted_kappa_quadratic | 0.5127 |
