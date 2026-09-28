# C7starpolfrlim — leakage-guarded polfrlim probe

## Purpose
Predict `polfrlim` with the C7 persona spec after removing that variable itself
from the persona.  Isolates whether the LLM can predict `polfrlim` from the
remaining features when the trivial-copy channel is blocked.

## Outcome
- **Variable:** `polfrlim` — Personal freedoms may be limited to fight terrorism (Likert 1-5)
- **Valid range:** 1–5
- **Minority code:** n/a (ordinal)

## Protocol
- Config: **C7starpolfrlim** (C7 groups minus `polfrlim`)
- VS-CoT · gpt-5.4-mini · +PC · s0
- Matched respondents: 2,195

## Files
- `predictions.jsonl` — raw API results
- `metrics.csv`       — single-row metrics
- `personas/`         — 2,615 rendered persona markdown files
- `README.md`         — this file

## Metrics

| Metric | Value |
|--------|------:|
| n_matched | 2195 |
| jsd | 0.2638 |
| wasserstein | 1.3490 |
| tvd | 0.5153 |
| pred_mean | 4.3681 |
| gt_mean | 3.0191 |
| prevalence_error | 1.3490 |
| cohen_kappa | 0.0277 |
| weighted_kappa_quadratic | 0.0850 |
