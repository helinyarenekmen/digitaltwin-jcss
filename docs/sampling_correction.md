# Sampling protocol correction (short note)

## The bug

Every runner (`04_run_calibration.py`, `09_run_calibration_gemini.py`,
`10_run_calibration_openrouter.py`, `run_calibration_natural.py`, and the
supplementary Claude runner) originally called
`np.random.default_rng(seed)` **inside** `parse_vs_cot()`. Because the
function was re-entered once per respondent with the same integer seed,
every respondent's verbalized distribution was scored at the same uniform
quantile. This collapses probability-weighted sampling into fixed-quantile
inverse-CDF sampling.

## The fix

Every runner now creates the RNG **once per run** at the base seed and
threads it into `predict_one → parse_vs_cot` as a shared
`numpy.random.Generator`. The RNG advances naturally across respondents.

Files patched:

* `scripts/04_run_calibration.py`
* `scripts/09_run_calibration_gemini.py`
* `scripts/10_run_calibration_openrouter.py`
* `scripts/run_calibration_natural.py`
* `supplementary/scripts/08_run_calibration_claude.py`

## Why `07_resample_vs_cot.py` is kept

Re-running Stages 2 through 5 for both outcomes with the corrected RNG
would repeat several thousand paid API calls. The archived JSONL retains
`raw_response` — including the full verbalized distribution — so a
post-hoc resample is analytically identical to a correctly-seeded rerun.
The helper is preserved as the reproducibility bridge for the archived
cache. It should not be used on any new run performed under the corrected
runner.

## Reproducing paper Table 15

Paper Table 15 numbers are computed with the paper protocol — a single
`numpy.random.Generator` initialised at the base seed and advanced across
respondents in **file-appearance order** (the async completion order at
inference time). `scripts/make_tables.py --table 15` implements exactly
this protocol.

`make_tables.py` writes both `outputs/tables/Table15_seed_pair_agreement.csv`
and its committed copy at `outputs/results/table_15_seed_pair_agreement.csv`.
