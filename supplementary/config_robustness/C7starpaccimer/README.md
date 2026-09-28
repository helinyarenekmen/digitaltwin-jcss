# C7starpaccimer — leakage-guarded paccimer probe

## Purpose
Predict `paccimer` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `paccimer` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `paccimer` — Past-year: filing a complaint to CİMER
- **Valid range:** 1–2
- **Minority code:** 1

## Protocol
- Config: **C7starpaccimer** (C7 groups minus `paccimer`)
- Direct · gpt-4o-mini · +PC · s0
- Matched respondents: 1,698

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 1698 |
| jsd | 0.0304 |
| wasserstein | 0.1266 |
| tvd | 0.1266 |
| pred_mean | 1.9505 |
| gt_mean | 1.8239 |
| prevalence_error | 0.1266 |
| cohen_kappa | 0.1200 |
| weighted_kappa_quadratic | 0.1200 |
| mcc | 0.1512 |
| recall_minority | 0.1204 |
| precision_minority | 0.4286 |
| f1_minority | 0.1880 |
| macro_f1 | 0.5424 |
