# C7* robustness / leakage test — pacdemons

## Purpose
Probe whether the final pacdemons calibration (C7 · Direct · T=0.8 · gpt-4o-mini · ben · +PC · s0)
draws part of its signal from *past-year political-participation* items in the persona.
If C7* (same protocol, participation items removed) matches C7's metrics, no leakage.
If C7* degrades substantially, the C7 gain was partly driven by participation-item overlap.

## What was removed
9 variables from the "6. Political" group, each measuring past-year participation:

| Code | Label |
|------|-------|
| paccontact   | Contacting a politician or official |
| paccompl     | Filing a request or complaint |
| paccimer     | Filing a complaint to CİMER |
| pacparty     | Party political activities |
| pacvolunteer | Volunteering for a nonprofit or charity |
| pacboycott   | Boycotting products |
| paconline    | Sharing political content online |
| paccult      | Volunteering in social/cultural activities |
| paccharity   | Donating to or joining aid activities |

The outcome `pacdemons` itself is already excluded from all personas via the
leakage guard in `scripts/01_persona_engine.py` (`OUTCOME_VARS`).

## Protocol (identical to final C7 pick apart from `drop_vars`)
- Config: **C7\*** (same 5 groups as C7 minus 9 vars above)
- Sampling: direct
- Temperature: 0.8
- Model: gpt-4o-mini
- Address mode: ben
- Political-context prepend: **yes**
- Seed: 0
- CoT: no
- Matched respondents: 1,707

## Files
- `predictions.jsonl`  — raw API results (one line per respondent)
- `metrics.csv`        — single-row metrics for C7*
- `comparison_C7_vs_C7star.csv` — side-by-side vs C7 baseline (same protocol)
- `personas/`          — 2,615 rendered C7* persona markdown files
- `README.md`          — this file

## Result at a glance

| Metric | C7 baseline | C7* | Δ (C7* − C7) |
|--------|------------:|----:|-------------:|
| mcc | 0.4251 | 0.1777 | -0.2474 |
| recall_minority | 0.5357 | 0.2976 | -0.2381 |
| precision_minority | 0.3913 | 0.1773 | -0.2140 |
| f1_minority | 0.4523 | 0.2222 | -0.2300 |
| cohen_kappa | 0.4192 | 0.1711 | -0.2481 |
| jsd | 0.0011 | 0.0033 | +0.0022 |
| wasserstein | 0.0182 | 0.0334 | +0.0152 |
| prevalence_error | 0.0182 | 0.0334 | +0.0152 |

(A near-zero Δ across MCC/recall_minority/κ indicates no leakage; a large
drop indicates C7's signal partly relied on past-year participation items.)
