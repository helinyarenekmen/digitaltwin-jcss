# C7starpaconline — leakage-guarded paconline probe

## Purpose
Predict `paconline` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `paconline` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `paconline` — Past-year participation: sharing political content online
- **Valid range:** 1–2
- **Minority code:** 1

## Protocol
- Config: **C7starpaconline** (C7 groups minus `paconline`)
- Sampling: direct
- Temperature: 0.8
- Model: gpt-4o-mini
- Address mode: ben
- Political-context prepend: yes
- Seed: 0
- CoT: no
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
| jsd | 0.0100 |
| wasserstein | 0.0817 |
| tvd | 0.0817 |
| pred_mean | 1.8994 |
| gt_mean | 1.8178 |
| prevalence_error | 0.0817 |
| cohen_kappa | 0.2837 |
| weighted_kappa_quadratic | 0.2837 |
| mcc | 0.3007 |
| recall_minority | 0.2922 |
| precision_minority | 0.5294 |
| f1_minority | 0.3766 |
| macro_f1 | 0.6369 |
