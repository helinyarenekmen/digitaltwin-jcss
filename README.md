# Calibrating Synthetic Respondents

**Replication materials for**: *Calibrating Synthetic Respondents: A
Within-Persona Experiment on Public Reactions to Protest in Turkey.*
M. Fuat Kına and Helin Yaren Ekmen · Institute of Population and Social
Research, Marmara University · Submitted to *Journal of Computational
Social Science* (Springer).

> Placeholder citation:
> Kına, M. F. & Ekmen, H. Y. (2026). *Calibrating Synthetic Respondents: A
> Within-Persona Experiment on Public Reactions to Protest in Turkey.*
> Journal of Computational Social Science, forthcoming.
> Replication archive: Zenodo, DOI TODO.

## Summary

We build persona-conditioned synthetic respondents from the 2024 Turkish
General Social Survey (TGSS 2024, n = 2,615), passing 117 respondent
variables organised into 6 thematic groups to a large language model that
answers survey items from the respondent's point of view. We calibrate the
inference pipeline through a four-stage funnel — configuration screening,
sampling method × temperature, cross-model comparison, and prompt-framing
stress tests — using two held-out outcomes (`pacdemons`, binary; `womenwork`,
5-point Likert). We then use the calibrated pipeline for a within-persona
experiment on Turkish public reactions to street protest under threat- vs
opportunity-framed vignettes about the Kurdish peace process, and we
externally validate the final protocol on six additional transfer items in
Appendix G.

## Repository layout

```
.
├── README.md                    # this file
├── LICENSE / LICENSE-DATA       # MIT (code) · CC BY 4.0 (derived data)
├── CITATION.cff / .zenodo.json  # citation metadata
├── requirements.txt             # pinned Python dependencies
├── .env.example                 # required API keys (names only)
├── config/
│   └── ablations.py             # 11 persona configurations (C0–C10 in the paper)
├── prompts/                     # verbatim Turkish system prompts, VS suffix,
│                                # ideology-background paragraph, vignettes, outcome questions
├── src/                         # library-style helpers (personas / inference / evaluation / experiment)
├── scripts/                     # numbered entry points (see "Reproduction" below)
├── data/
│   ├── raw/                     # EMPTY — put downloaded TGSS 2024 .sav here
│   └── derived/                 # cleaned TGSS CSV, variable dictionary, geo shapefile
├── outputs/
│   ├── completions/             # raw LLM completions (Zenodo bundle, see README)
│   ├── parsed/                  # long-format predictions for the experiment
│   └── results/                 # metric tables and Appendix G transfer items
├── figures/                     # every figure that appears in the paper (PDF + PNG)
├── supplementary/               # not reported in the paper — additional analyses
├── sections/                    # LaTeX manuscript fragments (for cross-referencing)
└── tests/                       # sanity checks against paper's headline numbers
```

## Data access

The **TGSS 2024 microdata** must be downloaded separately — it is not
distributed here:

> **Zenodo DOI**: [10.5281/zenodo.18721350](https://doi.org/10.5281/zenodo.18721350)

Save the SPSS file as `data/raw/TGSS2024.sav`. Detailed instructions live in
[`data/raw/README.md`](data/raw/README.md).

Derived data (cleaned CSV, variable dictionary, NUTS-1 GeoJSON) is included
in `data/derived/` under CC BY 4.0. Persona corpora (~500 MB) are
regenerated deterministically by `scripts/01_build_personas.py`.

The **raw model completions** (~415 MB) are archived on Zenodo. See
[`outputs/completions/README.md`](outputs/completions/README.md).

## Environment setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# ...then edit .env with your API keys
```

Four API providers are used:

| Provider | Env variable | Used for |
|----------|-------------|----------|
| OpenAI | `OPENAI_API_KEY` | GPT-4o-mini (all stages), GPT-5.4-mini (Stages 3, 4) |
| Anthropic | `ANTHROPIC_API_KEY` | Claude Haiku 4.5 (Stage 3) |
| Google AI Studio | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | Gemini 2.5 Flash Lite (Stage 3) |
| OpenRouter | `OPENROUTER_API_KEY` | Llama 3.3 70B (Stage 3) |

If any variable is missing, the corresponding script falls back to a
`getpass` prompt.

## Reproduction

### Path (a) — Re-run inference from scratch

**Expensive** (~US$50 in API costs, ~4 h on 8-way concurrency) and will
produce numerically slightly different results because commercial models
drift.

```bash
python scripts/00_prepare_data.py           # TGSS .sav → cleaned CSV
python scripts/01_build_personas.py         # persona corpora (all 27 configs)
python scripts/02_run_inference.py --stage screening   # ~$5, 30 min
python scripts/02_run_inference.py --stage sampling_temperature
python scripts/02_run_inference.py --stage models       # cross-vendor
python scripts/02_run_inference.py --stage prompt_variants
python scripts/02_run_inference.py --stage seed_stability
python scripts/02_run_inference.py --stage transfer_items    # Appendix G
python scripts/05_run_experiment.py         # within-persona experiment
python scripts/03_parse_completions.py
python scripts/04_compute_metrics.py
python scripts/06_experiment_analysis.py
python scripts/07_make_tables.py
python scripts/08_make_figures.py
```

### Path (b) — Reproduce every table and figure from archived completions

**No API calls**, ~5 minutes on a laptop.

```bash
# 1. Download the Zenodo completions bundle:
#    (Zenodo DOI TODO → outputs/completions/)
# 2. Regenerate personas (deterministic, no API):
python scripts/01_build_personas.py

# 3. Recompute metrics and tables:
python scripts/03_parse_completions.py
python scripts/04_compute_metrics.py
python scripts/06_experiment_analysis.py
python scripts/07_make_tables.py
python scripts/08_make_figures.py
```

### Verification

The `tests/` directory contains sanity checks against the paper's headline
numbers. Run:

```bash
pytest tests/
```

Successful output confirms the four checks below match to within ±0.005:

- Table 2 · pacdemons final protocol: **recall_minority = 0.512, JSD = 0.0011**
- Table 2 · womenwork final protocol: **Wasserstein = 0.181, κw = 0.312**
- Experiment · legitimacy: **Δ = −0.41, 95% CI [−0.47, −0.35], d_z = −0.26**
- Experiment · behavioral intent: **Δ = −15.76 pp, 95% CI [−17.15, −14.36]**

## Table / figure → script mapping

| Paper | Script | Output file |
|-------|--------|-------------|
| Table 1 (persona configurations) | `07_make_tables.py --table 1` | `config/ablations.py` (source of truth) |
| Table 2 (final calibration metrics) | `07_make_tables.py --table 2` | `outputs/results/pacdemons_all_phases.xlsx` sheet "All cells" · `outputs/results/womenwork_all_phases.xlsx` |
| Table 3–4 (Stage 1 screening) | `04_compute_metrics.py --stage 1` | `outputs/results/pacdemons_all_phases.xlsx` (Phase 1 rows) |
| Table 5–7 (Stage 2 sampling × T) | `04_compute_metrics.py --stage 2` | `outputs/results/*_all_phases.xlsx` (Phase 2–5 rows) |
| Table 8–10 (Stage 3 cross-model) | `04_compute_metrics.py --stage 3` | `outputs/results/*_all_phases.xlsx` (Phase 7–10 rows) |
| Table 11–13 (Stage 4 prompt) | `04_compute_metrics.py --stage 4` | `outputs/results/*_all_phases.xlsx` (Phase 15–16 rows) |
| Table 14–15 (seed stability) | `04_compute_metrics.py --stage 5` | `outputs/results/*_all_phases.xlsx` (Phase 12 rows) |
| **Table 16 (transfer, binary)** | `07_make_tables.py --table 16` | [`outputs/results/table_16_transfer_binary.csv`](outputs/results/table_16_transfer_binary.csv) |
| **Table 17 (transfer, ordinal)** | `07_make_tables.py --table 17` | [`outputs/results/table_17_transfer_ordinal.csv`](outputs/results/table_17_transfer_ordinal.csv) |
| Figure 1 (calibration funnel) | `08_make_figures.py --fig 1` | `figures/fig_calibration_funnel.pdf` (TODO — script) |
| Figure 2 (Stage 1 JSD) | `plot_step2a_jsd.py` | `figures/calibration/fig_step2a_jsd.pdf` |
| Figure 3 (final picks vs GT) | `plot_final_picks_vs_gt.py` | `figures/calibration/*.pdf` |
| Figure 4 (effect summary) | `08_make_figures.py --fig 4` | [`figures/fig_effect_summary.pdf`](figures/fig_effect_summary.pdf) |
| Figure 5 (legitimacy distribution) | `08_make_figures.py --fig 5` | [`figures/fig_legitimacy_distribution.pdf`](figures/fig_legitimacy_distribution.pdf) |
| Figure 6 (behavioral distribution) | `08_make_figures.py --fig 6` | [`figures/fig_behavioral_distribution.pdf`](figures/fig_behavioral_distribution.pdf) |
| Figure 7 (subgroup forest) | `08_make_figures.py --fig 7` | [`figures/fig_subgroup_forest.pdf`](figures/fig_subgroup_forest.pdf) |
| Figure 8 (within-condition contrasts) | `08_make_figures.py --fig 8` | [`figures/fig_within_condition_forest.pdf`](figures/fig_within_condition_forest.pdf) |
| Figure 9 (transitions heatmap) | `08_make_figures.py --fig 9` | [`figures/fig_within_persona_transitions.pdf`](figures/fig_within_persona_transitions.pdf) |
| Figure 10 (NUTS-1 map) | `08_make_figures.py --fig 10` | [`figures/fig_regional_legitimacy_map.pdf`](figures/fig_regional_legitimacy_map.pdf) · [`figures/fig_regional_behavioral_map.pdf`](figures/fig_regional_behavioral_map.pdf) |

**TODO — cross-reference these mappings against the final manuscript once the
JCSS version is fixed. Placeholder script `07_make_tables.py` and figure
dispatchers `08_make_figures.py` still need to be written; the underlying
CSVs and per-figure scripts already exist. Some renumbering may be needed
once JCSS sends galley proofs.**

## Model table

| Provider | Model identifier / snapshot | Access dates | Temperature | top_p | max_tokens (direct / VS-CoT / CoT) | Seeds used |
|----------|-----------------------------|--------------|------------:|------:|------------------------------------|------------|
| OpenAI | `gpt-4o-mini` | 2026-05 → 2026-08 (TODO exact) | 0.0, 0.4, 0.8 | 1.0 (default) | 10 / 250 / 200 | {0, 1, 2} |
| OpenAI | `gpt-5.4-mini` | 2026-06 → 2026-08 (TODO) | 0.8 | 1.0 | via `max_completion_tokens` | {0, 1, 2} |
| OpenAI | `gpt-5-mini` | 2026-07 (TODO) | 0.8 | 1.0 | 400 (VS-CoT + CoT) | {0} |
| Anthropic | `claude-haiku-4-5-20251001` | 2026-07 → 2026-08 (TODO) | 0.8 | 1.0 | 250 (VS-CoT) | {0} |
| Google AI Studio | `gemini-2.5-flash-lite` | 2026-06 → 2026-08 (TODO) | 0.0, 0.4, 0.8 | 1.0 | 250 (VS-CoT) | {0, 1, 2} |
| OpenRouter → Meta | `meta-llama/llama-3.3-70b-instruct` | 2026-07 (TODO) | 0.8 | 1.0 | 250 (VS-CoT) | {0, 1, 2} |

`top_p` was never explicitly set; provider defaults (`1.0`) apply. **TODO —
Kına/Ekmen: replace access-date placeholders with actual date ranges from
your API dashboards.**

## Seed handling

- **Direct answering** (Stages 1, 3, most of 4): the seed is passed to the
  provider's `seed` argument when `temperature = 0`. For `T > 0`, the seed
  argument has no effect (commercial API behavior) — replication under
  T > 0 is therefore only within-drift.
- **Verbalized sampling with chain-of-thought** (VS-CoT): the model returns
  a probability distribution over answer categories. A **client-side**
  numpy RNG (`np.random.default_rng(base_seed)`) draws the categorical
  sample from that distribution. Under the base protocol the RNG is
  reinitialised once per base seed and consumed sequentially across
  respondents; `07_resample_vs_cot.py` post-hoc corrects an earlier bug
  where the RNG was reinitialised per respondent (see Known deviations).
- **Experiment seeds**: fixed per respondent-condition pair via the same
  API-side seed argument when `T = 0`, or via a client-side hash of
  `(respondent_id, condition, base_seed)` when `T > 0`.

## Known deviations

- **Llama-3.3-70B C3 cell is incomplete** (1,783 of 2,588 respondents).
  Downstream aggregations for this cell subset accordingly.
- **VS-CoT client-side RNG bug fix**: in the initial screening runs the
  numpy RNG was reinitialised per respondent (rather than once per base
  seed), which inflated per-respondent stability metrics. All final metrics
  are computed from the corrected reprocessing performed in
  `07_resample_vs_cot.py`, which does *not* require additional API calls.
  See `outputs/results/methodology_note.md`.
- **Invalid completions are dropped, not imputed**: parse failures (rare,
  <1%) and API errors are excluded from metric computations. Sample-size
  columns in the results tables reflect this filter.
- **Paper labels "C6" == internal "C7"**: the paper renumbers the eleven
  ablations contiguously (C0–C10) for readability; the internal registry
  (`config/ablations.py`) preserves the historical numbering that skips
  C2. Concretely: paper C6 ↔ internal C7 (= "all persona groups except
  Social-Psychological").

## Contact

For replication questions: Helin Yaren Ekmen &lt;helinyarenekmen@gmail.com&gt;.
For substantive questions: TODO co-author contact.
