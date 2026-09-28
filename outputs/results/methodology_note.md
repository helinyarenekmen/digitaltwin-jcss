# calibration_final_v2 — sampling protocol correction

**Date:** 2026-08-05
**Applies to:** all VS-CoT metric CSVs, ranking CSVs, per-outcome Excel workbooks, and decision JSONs (Stages 2–5) for both `womenwork` and `pacdemons`.

## The bug

The VS-CoT client-side sampler in every calibration script
(`scripts/04_run_calibration.py`, `scripts/08_run_calibration_claude.py`,
`scripts/09_run_calibration_gemini.py`, `scripts/10_run_calibration_openrouter.py`,
`scripts/run_calibration_natural.py`) contained the following pattern inside
`parse_vs_cot()`:

```python
def parse_vs_cot(raw, valid_range, seed):
    ...
    rng = np.random.default_rng(seed)          # ← re-initialised per respondent
    return int(rng.choice(keys, p=probs))
```

The `numpy.random.default_rng(seed)` call was executed **once per respondent
call** with the same integer seed, so every respondent used the **same first
uniform draw** of that seed. This collapses probability-weighted sampling
(what the paper prescribes: draw *one respondent* from that respondent's
own verbalized distribution) into fixed-quantile inverse-CDF sampling
(evaluate every respondent's CDF at the same point).

## The correction

Under the correct protocol (per Zhang et al. 2026, "Verbalized Sampling:
How to Mitigate Mode Collapse and Unlock LLM Diversity", §6), a single RNG
should be seeded once per run and advanced across respondents:

```python
def parse_vs_cot(raw, valid_range, rng):    # rng passed in
    ...
    return int(rng.choice(keys, p=probs))    # advances rng

# Main loop:
rng = np.random.default_rng(run_seed)
for respondent in respondents:
    pred = parse_vs_cot(respondent.raw_response, valid_range, rng)
```

`calibration_final_v2` was produced by **post-hoc reprocessing** of the
existing raw JSONL cache files. The model output (the verbalized
distribution itself) is stored verbatim in the `raw_response` field of every
cache row, so no additional API calls were required — only the client-side
sampling of that distribution was re-run under the correct protocol.

## Impact — headline numbers

### Stability CVs collapse

| Cell | Buggy Wass CV | Fixed Wass CV | Buggy κ_quad CV | Fixed κ_quad CV |
|------|---:|---:|---:|---:|
| gpt-5.4-mini · C7 · VS · T=0.8 (3-seed) | 42.8 % | **10.0 %** | 6.7 % | **6.8 %** |
| gemini-2.5-flash-lite · C7 · VS · T=0.8 (13-seed) | 61.1 % | **4.2 %** | 49.0 % | **6.3 %** |
| gpt-4.1-mini · C7 · VS · T=0.8 (3-seed) | 72.0 % | **5.3 %** | 20.1 % | **5.3 %** |
| gpt-4o-mini · C4 · VS · T=0.4 (3-seed) | 1.8 % | 4.6 % | 0.6 % | 10.8 % |
| gpt-4o-mini · C11 · VS · T=0.8 (3-seed) | 3.4 % | 12.5 % | 6.9 % | 18.6 % |

The pathological CVs (>40 %) all collapse to the 4–13 % band. The tiny CVs
observed on gpt-4o-mini configs rise slightly because those models produce
narrowly peaked distributions where the buggy sampler happened to hit the
mode almost every time.

### Womenwork final pick metrics

| | Buggy | Fixed |
|--|--:|--:|
| Wass | 0.399 ± 0.171 | **0.207 ± 0.021** |
| κ_quad | 0.404 ± 0.027 | **0.284 ± 0.019** |

Wass drops to about half — the finalist's aggregate marginal fit is better
than the pre-fix report claimed. κ_quad drops from "moderate agreement" to
"fair agreement" — the finalist's per-respondent inference is not as strong
as claimed but still 4× better than the Wass-champion candidate
(gpt-4o-mini · C11).

### The Gemini "lucky-of-13 outlier" narrative is retracted

Under the buggy protocol Gemini FL · C7 · VS · T=0.8 showed 13-seed
Wass values ranging 0.22 – 2.07 (CV 61 %), and its s0 (0.218) looked like
a lucky-of-13 outlier that had originally been chosen as the finalist. Under
the fixed protocol the same 13 seeds land in the range 0.46 – 0.53
(CV 4.2 %), i.e. Gemini is one of the most stable vendors in the pool.
The vendor is still not selected — its aggregate Wass (0.497) is
2.4× worse than gpt-5.4-mini's (0.207) — but the reason shifts from
"unreproducible" to "consistently lower quality".

### Several Stage 4 ablation verdicts flip

Some ablation Δ that looked strongly negative under the buggy protocol
turn out to be near-zero or positive under the fixed protocol:

| Ablation on gpt-5.4-mini · C7 · VS · T=0.8 · ben (s0) | Buggy Δ Wass | Fixed Δ Wass | Fixed Δ κ_quad |
|---|--:|--:|--:|
| sen-dili | + 0.038 | + 0.020 | − 0.005 |
| explicit CoT | + 0.032 | **− 0.056** | − 0.020 |
| political context | + 0.057 | **− 0.013** | + 0.013 |
| natural persona rewrite | − 0.135 (Wass ↑) & κ_quad − 0.191 | + 0.003 | − 0.047 |

The four defaults (ben, no CoT, no PC, structured Layer-A) are still
retained as the finalist protocol, but CoT and PC now sit as strong
candidates for a 3-seed follow-up. See
`stages/stage_4_prompt_design/womenwork_stage4_decision.json` for the
revised full analysis.

## What was NOT affected

- **Direct-sampling metrics.** The bug only lived in `parse_vs_cot()`; Direct
  sampling reads a single integer from the model response, no client-side
  RNG involved. Every Direct-sampling row in every metric CSV and Excel
  workbook is unchanged from `calibration_final`.
- **Pacdemons final pick.** The Stage 2 decision for pacdemons is
  Direct · T=0.8, which is unaffected by the bug. The pacdemons finalist
  metrics did not change. The VS-CoT rows in the pacdemons Excel have been
  updated, but VS is documented as rejected regardless of the sampling
  protocol.
- **Raw cache data.** No API calls were re-made. The `~/Library/Caches/
  digitaltwin_calibration*` JSONL files are the same as under the original
  protocol; only their client-side sampling was re-run.

## What was regenerated in v2

| Artefact | Location in v2 |
|---|---|
| VS-CoT metric CSVs (Stages 2, 3, 4, 5) | `stages/*/metrics/*_vs_*_metrics.csv` |
| Ranking CSVs (VS variants) | `stages/*/metrics/*_vs_*_ranking.csv` |
| `selected_configs.json` files (VS variants) | `stages/*/metrics/*_vs_*_selected_configs.json` |
| Per-outcome Excel workbooks | `per_outcome/{womenwork,pacdemons}/*_all_phases.xlsx` |
| Womenwork Stage 2/3/4/5 decision JSONs | `stages/stage_{2,3,4,5}_*/womenwork_stage{2,3,4,5}_decision.json` |
| Pacdemons Stage 2 decision JSON | `stages/stage_2_sampling_and_temperature/pacdemons_stage2_decision.json` |
| Womenwork validation plots | `per_outcome/womenwork/validation_plots/` |
| Stage 3/4/5 plots that reference VS-CoT metrics | `stages/stage_{3,4,5}_*/plots/` |

## Reprocessing scripts

The reprocessing was driven by three scripts kept in the session scratchpad:

- `reprocess_vs_cot.py` — walks every `*_vs_*_metrics.csv`, reconstructs the
  cache path for each row, resamples with `np.random.default_rng(run_seed)`
  advancing across respondents, recomputes every metric column, and writes
  back preserving schema.
- `regen_rankings.py` — regenerates the ranking CSVs and `selected_configs`
  JSONs from the fresh metric CSVs. `in_shortlist` values are left
  unchanged; only metric and rank columns are recomputed.
- `update_excel.py` — walks the two per-outcome workbooks, updates
  every `Sampling == 'vs'` row's metric columns with fixed-protocol values.

Copies of these scripts are archived as attachments to
`stages/stage_5_multiseed_stability/womenwork_stage5_decision.json` for
paper-side reproducibility.

## Recommendation for the paper

A short methodology paragraph:

> Post-hoc distributional reprocessing of a VS-CoT calibration campaign
> revealed that a per-respondent RNG reinitialisation in the client-side
> sampler collapsed probability-weighted sampling of the verbalized
> distributions into fixed-quantile inverse-CDF sampling. This
> implementation choice inflated apparent multi-seed variance of aggregate
> metrics by roughly an order of magnitude (Wass CV 4 % → 61 % on our worst
> case), biased single-seed ablation deltas by up to 8×, and generated an
> apparent "lucky-seed outlier" narrative for one vendor that vanishes
> under correct sampling. Because the model output (the verbalized
> distribution itself) is preserved in the calibration cache, the
> correction requires zero additional API calls — only a re-sampling of the
> raw distributions with a correctly advancing RNG. We recommend that any
> VS-CoT reproducibility protocol include a post-hoc distributional
> re-sampling audit as a standard step.
