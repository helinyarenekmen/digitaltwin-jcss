# C7starsatdem — leakage-guarded satdem probe

## Purpose
Predict `satdem` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `satdem` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `satdem` — Satisfaction with democracy in Turkey (Likert 1-5)
- **Valid range:** 1–5
- **Minority code:** n/a (ordinal)

## Protocol
- Config: **C7starsatdem** (C7 groups minus `satdem`)
- VS-CoT · gpt-5.4-mini · +PC · s0
- Matched respondents: 1,585

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 1585 |
| jsd | 0.0301 |
| wasserstein | 0.2808 |
| tvd | 0.1628 |
| pred_mean | 2.1205 |
| gt_mean | 2.4013 |
| prevalence_error | 0.2808 |
| cohen_kappa | 0.2762 |
| weighted_kappa_quadratic | 0.5620 |
