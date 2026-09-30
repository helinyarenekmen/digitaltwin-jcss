# Additional target-item holdout probes (not in the paper)

Ten additional holdout probes we ran during development. **None of these is
reported in the paper.** They are included for transparency and to document
the fuller set we tested.

Paper's Appendix G reports six holdout probes, all located under
`outputs/results/transfer_items/`:

| Protocol                                                       | Items reported in the paper |
|----------------------------------------------------------------|-----------------------------|
| C6 · direct · T=0.8 · GPT-4o-mini · +ideological background    | `pacvolunteer`, `paccontact`, `paccompl` |
| C6 · verbalized · T=0.8 · GPT-5.4-mini · +ideological background | `satdem`, `famroles`, `polint` |

Everything under this folder is additional. The items were either outside the
paper's scope (e.g., normative political attitudes such as `polminor`,
`polfrlim`, `thimmig`), sat downstream of the main calibration outcome
(`pacdemons`), or were superseded by cleaner probes.

| Folder                             | Removed outcome | Protocol |
|------------------------------------|-----------------|----------|
| `C6-holdout-pacdemons`             | pacdemons (paper's main outcome — see Section 4) | binary  |
| `C6-holdout-pacparty`              | pacparty (party political activities)             | binary  |
| `C6-holdout-pacboycott`            | pacboycott (product boycott)                      | binary  |
| `C6-holdout-paconline`             | paconline (online political sharing)              | binary  |
| `C6-holdout-paccimer`              | paccimer (CİMER complaint)                        | binary  |
| `C6-holdout-paccult`               | paccult (social/cultural volunteering)            | binary  |
| `C6-holdout-paccharity`            | paccharity (charity participation)                | binary  |
| `C6-holdout-polminor`              | polminor (minorities' rights protected)           | ordinal |
| `C6-holdout-polfrlim`              | polfrlim (freedoms may be limited for anti-terror)| ordinal |
| `C6-holdout-thimmig`               | thimmig (immigrants as threat, 0–10)              | ordinal |

Each folder contains `metrics.csv`, `predictions.jsonl`, and
`gt_vs_pred.{pdf,png}`. `all_16_summary.csv` aggregates all sixteen holdout
runs (the paper's six plus these ten).
