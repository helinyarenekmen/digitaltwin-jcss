# C7starpacparty — leakage-guarded pacparty probe

## Purpose
Predict `pacparty` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `pacparty` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `pacparty` — Past-year participation: party political activities
- **Valid range:** 1–2
- **Minority code:** 1

## Protocol
- Config: **C7starpacparty** (C7 groups minus `pacparty`)
- Sampling: direct
- Temperature: 0.8
- Model: gpt-4o-mini
- Address mode: ben
- Political-context prepend: yes
- Seed: 0
- CoT: no
- Matched respondents: 1,694

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 1694 |
| jsd | 0.0020 |
| wasserstein | 0.0307 |
| tvd | 0.0307 |
| pred_mean | 1.9191 |
| gt_mean | 1.8884 |
| prevalence_error | 0.0307 |
| cohen_kappa | 0.3366 |
| weighted_kappa_quadratic | 0.3366 |
| mcc | 0.3419 |
| recall_minority | 0.3439 |
| precision_minority | 0.4745 |
| f1_minority | 0.3988 |
| macro_f1 | 0.6674 |
