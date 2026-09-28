# C7starpaccompl — leakage-guarded paccompl probe

## Purpose
Predict `paccompl` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `paccompl` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `paccompl` — Past-year: filing a request or complaint
- **Valid range:** 1–2
- **Minority code:** 1

## Protocol
- Config: **C7starpaccompl** (C7 groups minus `paccompl`)
- Direct · gpt-4o-mini · +PC · s0
- Matched respondents: 1,690

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 1690 |
| jsd | 0.0001 |
| wasserstein | 0.0095 |
| tvd | 0.0095 |
| pred_mean | 1.8172 |
| gt_mean | 1.8266 |
| prevalence_error | 0.0095 |
| cohen_kappa | 0.3736 |
| weighted_kappa_quadratic | 0.3736 |
| mcc | 0.3738 |
| recall_minority | 0.4983 |
| precision_minority | 0.4725 |
| f1_minority | 0.4850 |
| macro_f1 | 0.6867 |
