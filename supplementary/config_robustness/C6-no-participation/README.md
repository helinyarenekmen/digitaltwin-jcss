# C6-no-participation

**Not reported in the paper.** Robustness probe that predicts `pacdemons` under
the paper's calibrated binary protocol with **all nine past-year political-
participation variables** removed from the persona (`paccontact`, `paccompl`,
`paccimer`, `pacparty`, `pacvolunteer`, `pacboycott`, `paconline`, `paccult`,
`paccharity`). Every other persona variable is kept in place. `pacdemons`
itself is already excluded from every persona by the leakage guard.

## Purpose

Complements the single-variable Appendix G holdouts by testing whether the
paper's calibrated protocol still recovers pacdemons when the entire
past-year participation family is withheld from the persona. Included for
transparency; not a paper claim.

## Protocol

Identical to the paper's final pacdemons cell in Table 2:

* Config: **C6** minus the nine variables listed above
* Sampling: direct answering
* Temperature: 0.8
* Model: GPT-4o-mini
* Address mode: first person
* Prompt framing: + ideological background
* Seed: 0

## Files

* `predictions.jsonl` — raw API results
* `metrics.csv` — single-row metrics
* `gt_vs_pred.{pdf,png}` — ground-truth vs predicted distribution plot
* `comparison_C6_vs_no_participation.csv` — side-by-side against the paper's
  Table 2 pacdemons cell

## Provenance

The archived JSONL filenames in the raw completions cache use the legacy
config id `C7star` (the pre-publication numbering). This has been captured
in `config/config_id_mapping.csv` as `paper_id = C6-no-participation`,
`legacy_id = C7star`.
