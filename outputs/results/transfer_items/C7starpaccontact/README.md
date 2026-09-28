# C7starpaccontact — leakage-guarded paccontact probe

## Purpose
Predict `paccontact` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `paccontact` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `paccontact` — Past-year: contacting a politician or official
- **Valid range:** 1–2
- **Minority code:** 1

## Protocol
- Config: **C7starpaccontact** (C7 groups minus `paccontact`)
- Direct · gpt-4o-mini · +PC · s0
- Matched respondents: 1,657

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 1657 |
| jsd | 0.0018 |
| wasserstein | 0.0344 |
| tvd | 0.0344 |
| pred_mean | 1.8437 |
| gt_mean | 1.8781 |
| prevalence_error | 0.0344 |
| cohen_kappa | 0.4194 |
| weighted_kappa_quadratic | 0.4194 |
| mcc | 0.4238 |
| recall_minority | 0.5693 |
| precision_minority | 0.4440 |
| f1_minority | 0.4989 |
| macro_f1 | 0.7090 |
