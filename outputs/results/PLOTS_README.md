# Plots in calibration_final_v2

## Status per directory

| Location | Sampling protocol reflected | Notes |
|---|---|---|
| `per_outcome/womenwork/validation_plots/` | **FIXED** | Regenerated 2026-08-05. Includes `fig01`, `fig02`, and a new `fig03_sampling_protocol_correction` showing the Gemini 13-seed buggy-vs-fixed comparison. These are the authoritative womenwork validation figures for v2. |
| `stages/stage_1_configuration_screening/plots/` | Same as v1 | Stage 1 uses Direct sampling; unaffected by the VS-CoT sampling bug. Plots are correct. |
| `stages/stage_2_sampling_and_temperature/plots/S02/S03/S04/*` (VS variants) | Original (buggy) | These PNGs reflect the original per-respondent-RNG-reinit metrics. Use the metric CSVs in `stages/stage_2_.../metrics/*_vs_*_metrics.csv` for the authoritative fixed values. |
| `stages/stage_2_sampling_and_temperature/plots/S05_direct_temperature_ladder/` | Same as v1 | Direct sampling — unaffected. |
| `stages/stage_3_cross_vendor/plots/S07/S08/S09/S10/*` (VS panels) | Original (buggy) | VS panels reflect buggy metrics; Direct panels are correct. See CSVs. |
| `stages/stage_4_prompt_design/plots/S06/S12/S13/S15/*` (VS panels) | Original (buggy) | See revised `womenwork_stage4_decision.json` for corrected ablation deltas. |
| `stages/stage_4_prompt_design/natural_ablation_analysis/fig_structured_vs_natural.png` | Original (buggy) | See `womenwork_stage4_decision.json` `cross_model_natural_persona_finding` block for corrected values. |
| `stages/stage_5_multiseed_stability/plots/` | Original (buggy) | See revised `womenwork_stage5_decision.json` `sampling_artefact_correction` block for the corrected 13-seed spread. |

## Why some plots were not regenerated

The Stage-level PNGs (S02, S03, S04, S07, S08, S09, S10, S06, S12, S13, S15)
were originally produced by internal reporting scripts that no longer live in
the calibration_final tree. Regenerating them would require re-tracing those
scripts. The authoritative numeric values for all VS-CoT cells are in the
metric CSVs (`stages/*/metrics/*_vs_*_metrics.csv`) and per-outcome Excel
workbooks; the decision JSONs cite the corrected values directly. Anyone
using v2 as the paper-side data source should read numbers from the CSVs or
JSONs, not from the Stage-level PNGs.

## What to cite in the paper

For womenwork:
- `per_outcome/womenwork/validation_plots/fig01_final_pick_response_share_3seed.{png,pdf}` — observed vs simulated response distribution
- `per_outcome/womenwork/validation_plots/fig02_final_pick_per_seed_panels.{png,pdf}` — per-seed decomposition
- `per_outcome/womenwork/validation_plots/fig03_sampling_protocol_correction.{png,pdf}` — sampling-protocol correction figure (methodological)

For pacdemons validation figures: the pacdemons final pick uses Direct
sampling and its validation plots (if any are added later) do not need the
correction.
