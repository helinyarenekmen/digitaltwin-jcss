# C7starpaccharity — leakage-guarded paccharity probe

## Purpose
Predict `paccharity` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `paccharity` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `paccharity` — Past-year: donating to or joining aid activities
- **Valid range:** 1–2
- **Minority code:** 1

## Protocol
- Config: **C7starpaccharity** (C7 groups minus `paccharity`)
- Direct · gpt-4o-mini · +PC · s0
- Matched respondents: 1,704

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 1704 |
| jsd | 0.0112 |
| wasserstein | 0.1004 |
| tvd | 0.1004 |
| pred_mean | 1.8451 |
| gt_mean | 1.7447 |
| prevalence_error | 0.1004 |
| cohen_kappa | 0.4594 |
| weighted_kappa_quadratic | 0.4594 |
| mcc | 0.4821 |
| recall_minority | 0.4529 |
| precision_minority | 0.7462 |
| f1_minority | 0.5637 |
| macro_f1 | 0.7255 |
