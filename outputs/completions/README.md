# Raw model completions — archived on Zenodo

The raw LLM completions used to produce the paper's tables and figures are
not committed to git. They are approximately **~415 MB** total across all
paper configurations, seeds, and outcomes, and are archived as
zstd-compressed Parquet files (`*.parquet`) on Zenodo.

**Zenodo bundle DOI**: TODO — added on release

## What's in the bundle

Filenames use PAPER config labels (C0-C10, C6-holdout-<item>). Each file's
internal legacy id (the pre-publication config numbering used inside the
JSONL) is retained as a `legacy_config_id` column.

| File pattern | Contents |
|--------------|----------|
| `screening_T0/<outcome>/C6_gpt4omini_T0_ben_direct_nocot_s0.parquet` | Stage 1 configuration screening (T=0, direct) |
| `screening_T04/<outcome>/*.parquet`, `screening_T08/<outcome>/*.parquet` | Stage 2 temperature sweep (direct) |
| `screening_vs_cot/<outcome>/*.parquet`, `screening_vs_cot_T04/*`, `screening_vs_cot_T08/*` | Stage 2 verbalized sampling variants |
| `screening_T08/<outcome>/*_pc_s{0,1,2}.parquet` | Stage 4 ideological background variant, three seeds |
| `screening_T08/<outcome>/C6-holdout-<item>_*.parquet` | Appendix G target-item holdout runs |
| `experiments/exp_final/*.parquet` | Section 4 within-persona experiment (Threat vs Opportunity) |

Each Parquet file contains one row per respondent × condition, with columns:
`respondent_id, config_id, legacy_config_id, outcome, model, temperature,
address_mode, sampling, cot, seed, timestamp, predicted_value, raw_response,
prompt_tokens, completion_tokens, parse_status`.

## Rebuilding the bundle locally

```bash
python scripts/make_zenodo_bundle.py
```

The script reads from `DT_CACHE_DIR` (defaults to `./cache/calibration/`)
and from `~/Library/Caches/digitaltwin_calibration` if present, converts
every JSONL cell to Parquet+zstd, and writes an SHA-256 `MANIFEST.json`.

## Using the bundle

Download from the Zenodo record above, decompress into `./cache/calibration/`
(or wherever `DT_CACHE_DIR` points), and the pipeline scripts
(`04_run_calibration.py --resume`, `make_tables.py`, `make_figures.py`)
pick the completions up automatically. No API calls are required for the
"reproduce from archived outputs" path.
