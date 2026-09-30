# C6-holdout-paccompl — Appendix G target-item holdout probe

Predicts `paccompl` under the paper's calibrated protocol with `paccompl` removed
from the persona to prevent trivial-copy leakage. All other persona variables
are kept in place.

* Protocol (see paper Table 2):
  - Binary items (pacvolunteer, paccontact, paccompl):
    C6 · direct answering · T=0.8 · GPT-4o-mini · first-person · +ideological
    background · seed 0.
  - Ordinal items (famroles, satdem, polint):
    C6 · verbalized sampling · T=0.8 · GPT-5.4-mini · first-person · +ideological
    background · seed 0.
* Files:
  - `predictions.jsonl` — raw API results
  - `metrics.csv`       — single-row metrics (`legacy_id` retained for traceability)
  - `gt_vs_pred.{pdf,png}` — ground-truth vs predicted distribution plot

Paper labels: the config id printed in `metrics.csv` is the paper label
(`C6-holdout-paccompl`); the archived cache filename used the legacy id
(`C7starpaccompl`) which is preserved as the `legacy_id` column.
