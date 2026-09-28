# C7starpacvolunteer — leakage-guarded pacvolunteer probe

## Purpose
Predict `pacvolunteer` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `pacvolunteer` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `pacvolunteer` — Past-year: volunteering for a nonprofit/charity
- **Valid range:** 1–2
- **Minority code:** 1

## Protocol
- Config: **C7starpacvolunteer** (C7 groups minus `pacvolunteer`)
- Direct · gpt-4o-mini · +PC · s0
- Matched respondents: 1,711

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 1711 |
| jsd | 0.0059 |
| wasserstein | 0.0707 |
| tvd | 0.0707 |
| pred_mean | 1.7738 |
| gt_mean | 1.8445 |
| prevalence_error | 0.0707 |
| cohen_kappa | 0.4349 |
| weighted_kappa_quadratic | 0.4349 |
| mcc | 0.4466 |
| recall_minority | 0.6617 |
| precision_minority | 0.4548 |
| f1_minority | 0.5391 |
| macro_f1 | 0.7152 |
