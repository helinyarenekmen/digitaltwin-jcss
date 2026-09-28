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

In `scripts/04_run_calibration.py` the RNG is now created **once per run**
at the base seed and threaded into `predict_one → parse_vs_cot` as a shared
`numpy.random.Generator`. The RNG advances naturally across respondents.

The other runners (`09`, `10`, `run_calibration_natural`, and the
supplementary Claude runner) still contain the original per-respondent
reinit call. Applying the same fix to them is mechanical (mirror the change
in `04_run_calibration.py`); we did not touch them because the paper's
metrics for those cells were regenerated from the archived JSONL via
`07_resample_vs_cot.py`, which reprocesses the recorded verbalized
distributions with a correctly-advanced numpy RNG. Anyone rerunning
Stage 3 or the natural-rewrite cells from scratch should first port the
RNG fix into the corresponding runner, or apply `07_resample_vs_cot.py`
after the run.

## Why `07_resample_vs_cot.py` is kept

Re-running Stage 2 through Stage 5 for both outcomes with a corrected RNG
would repeat several thousand paid API calls. Instead the archived JSONL
retains `raw_response` — including the full verbalized distribution — so a
post-hoc resample is analytically identical to a correctly-seeded rerun.
This script is therefore preserved as the reproducibility bridge for the
archived cache. It should **not** be used on any new run performed under
the corrected runner in `04_run_calibration.py`.
