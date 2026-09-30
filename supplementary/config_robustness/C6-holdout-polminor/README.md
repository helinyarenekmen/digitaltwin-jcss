# C7starpolminor — leakage-guarded polminor probe

## Purpose
Predict `polminor` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `polminor` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `polminor` — Minorities' rights are protected in Turkey (Likert 1-5)
- **Valid range:** 1–5
- **Minority code:** n/a (ordinal)

## Protocol
- Config: **C7starpolminor** (C7 groups minus `polminor`)
- VS-CoT · gpt-5.4-mini · +PC · s0
- Matched respondents: 2,136

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 2136 |
| jsd | 0.2562 |
| wasserstein | 1.1039 |
| tvd | 0.5229 |
| pred_mean | 1.9494 |
| gt_mean | 3.0534 |
| prevalence_error | 1.1039 |
| cohen_kappa | 0.0154 |
| weighted_kappa_quadratic | 0.0066 |
