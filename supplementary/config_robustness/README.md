# Supplementary — additional target-item holdout probes

These runs are **not reported in the paper**. They are included for
transparency and to document the fuller set of leakage-guarded probes we
ran during development.

The paper's Appendix G (Tables 16–17) reports **six** target-item holdout
runs, all located in `outputs/results/transfer_items/`:

| Protocol | Outcomes reported in the paper |
|----------|-------------------------------|
| Binary (Direct · GPT-4o-mini · T=0.8 · C6 · ideology paragraph) | `pacvolunteer`, `paccontact`, `paccompl` |
| Ordinal (Verbalized sampling · GPT-5.4-mini · T=0.8 · C6 · ideology paragraph) | `satdem`, `famroles`, `polint` |

Everything in *this* folder is additional — ten target-item variants that we
tested but did not include in the paper because they either sit outside the
paper's scope (e.g., normative political attitudes such as `polminor`,
`polfrlim`, `thimmig`), sit downstream of the main calibration outcome
(`pacdemons`), or were superseded by cleaner probes:

| Config                | Removed outcome | Protocol |
|-----------------------|-----------------|----------|
| C6\*pacdemons         | pacdemons (paper's main outcome — see paper Section 4)    | binary  |
| C6\*pacparty          | pacparty  (party political activities)                     | binary  |
| C6\*pacboycott        | pacboycott (product boycott)                               | binary  |
| C6\*paconline         | paconline (online political sharing)                       | binary  |
| C6\*paccimer          | paccimer (CİMER complaint)                                 | binary  |
| C6\*paccult           | paccult (social/cultural volunteering)                     | binary  |
| C6\*paccharity        | paccharity (charity participation)                         | binary  |
| C6\*polminor          | polminor (minorities' rights protected)                    | ordinal |
| C6\*polfrlim          | polfrlim (freedoms may be limited for anti-terror)         | ordinal |
| C6\*thimmig           | thimmig (immigrants as threat, 0-10)                       | ordinal |

Each subfolder contains: `metrics.csv`, `predictions.jsonl`,
`gt_vs_pred.{pdf,png}`, and a `README.md` describing the individual run.
The aggregated view of all sixteen holdout runs (paper's six + these ten)
is provided in `all_16_summary.csv`.

**Naming note.** Folders are named `C7star<outcome>` because that is the
internal config-id used in `config/ablations.py`. In the paper, this same
configuration is labelled **C6** (the paper renumbers the eleven ablations
contiguously as C0–C10 for publication, whereas the internal registry
skips C2). The paper's C6 == internal C7 == "all persona groups except
Social-Psychological".
