# Raw model completions — archived on Zenodo

The raw LLM completions used to produce the tables and figures are **not
committed** to this git repository. They are approximately **~415 MB** total
across all configurations, seeds and outcomes, and are archived as compressed
Parquet files (`*.parquet`) on Zenodo:

**Zenodo DOI**: `TODO — add on release`

## What's in the bundle

| File pattern | Contents |
|--------------|----------|
| `screening_T0/<outcome>/<config>_gpt4omini_T0_ben_direct_nocot_s0.parquet` | Stage 1 configuration screening (T=0, direct) |
| `screening_T04/<outcome>/*.parquet` and `screening_T08/<outcome>/*.parquet` | Stage 2 temperature sweep (direct) |
| `screening_vs_cot/<outcome>/*.parquet`, `screening_vs_cot_T04/*`, `screening_vs_cot_T08/*` | Stage 2 verbalized sampling variants |
| `screening_T08/<outcome>/*_pc_s{0,1,2}.parquet` | Stage 4 political-context (ideology-paragraph) variant, three seeds |
| `screening_T08/<outcome>/C7star*.parquet` | Appendix G target-item holdout runs |
| `experiments/exp13/*.parquet` | Stage 2 within-persona experiment (Threat vs Opportunity) |

Each Parquet file contains one row per respondent × condition, with columns:
`respondent_id, config_id, outcome, model, temperature, address_mode,
sampling, cot, seed, timestamp, predicted_value, raw_response,
prompt_tokens, completion_tokens, parse_status`.

## How to use

Download the bundle from Zenodo, decompress into this folder, and the pipeline
scripts (`04_compute_metrics.py`, `06_experiment_analysis.py`, etc.) will pick
them up automatically. No API calls are required for the "reproduce from
archived outputs" path.
