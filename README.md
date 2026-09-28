# Calibrating Synthetic Respondents

Replication materials for:

> Kına, M. F. & Ekmen, H. Y. (2026). *Calibrating Synthetic Respondents:
> A Within-Persona Experiment on Public Reactions to Protest in Turkey.*
> SocArXiv, v1. DOI: [10.31235/osf.io/hp7gz_v1](https://doi.org/10.31235/osf.io/hp7gz_v1).
> Peer-reviewed version submitted to the *Journal of Computational Social Science*.

## Summary

Persona-conditioned synthetic respondents are constructed for all 2,615
participants of the 2024 Turkish General Social Survey (TGSS 2024). A
sequential calibration funnel selects one inference protocol per outcome
against two held-out TGSS items — `pacdemons` (binary, rare positive) and
`womenwork` (5-point Likert). Six transfer items (`pacvolunteer`,
`paccontact`, `paccompl`, `famroles`, `satdem`, `polint`) are then used to
externally validate the calibrated protocols in Appendix G. Finally, the
calibrated protocols are applied to a within-persona synthetic experiment
that varies the political context (threat versus opportunity) around a
protest for Kurdish-language education in Ankara.

## Data & licenses

* Code — **MIT** (see `LICENSE`)
* Derived data (persona corpora, cleaned TGSS extract, model completions,
  metric tables) — **CC BY-NC 4.0** (see `LICENSE-DATA`).
* TGSS 2024 microdata — the source dataset is distributed under the
  producers' own **CC BY-NC 4.0** licence and is **not** redistributed
  here. Download from Zenodo (Kına 2026, [10.5281/zenodo.18721350][tgss])
  and place the .sav file at `data/raw/TGSS2024.sav`.

[tgss]: https://doi.org/10.5281/zenodo.18721350

If you use TGSS-derived material from this repository, cite the TGSS 2024
data producers **and** this replication archive.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your keys — see the model table below for which providers
```

By default the pipeline reads and writes caches inside the repo:

* Raw JSONL completions      → `./cache/calibration/`
* Rendered persona markdown  → `./data/derived/personas/`

Both can be relocated with `DT_CACHE_DIR` and `DT_PERSONA_DIR`.

## Reproduction

There are two reproduction paths. Path B is fast and free; Path A rerun
every API call from scratch.

### Path B — Reproduce every table and figure from archived outputs

No API calls, ~5 minutes on a laptop.

```bash
# 1. Download the Zenodo completions bundle into cache/calibration/
#    (Zenodo DOI to be added on release)

# 2. Rebuild persona corpora deterministically (no API):
python scripts/00_prepare_data.py             # TGSS .sav → cleaned CSV
python src/personas/01_persona_engine.py      # 2,615 respondents × 11 configs

# 3. Regenerate tables and figures:
python scripts/make_tables.py                 # outputs/tables/Table*.csv
python scripts/make_figures.py                # figures/Fig*.pdf

# 4. Run the verification suite:
pytest tests/
```

### Path A — Re-run inference from scratch

Approximately US$40 in API costs and ~4 hours on 8-way concurrency.
Numbers will differ slightly because commercial models drift.

```bash
python scripts/00_prepare_data.py
python src/personas/01_persona_engine.py
python scripts/04_run_calibration.py \
    --configs C0,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10 \
    --outcomes pacdemons,womenwork \
    --sampling direct --temperature 0.0

# Then Stages 2-5 per the paper's funnel (each takes ~30-60 min):
python scripts/04_run_calibration.py --sampling vs_cot --temperature 0.4
python scripts/09_run_calibration_gemini.py    # Stage 3 (Gemini)
python scripts/10_run_calibration_openrouter.py --model llama-3.3-70b   # Stage 3 (Llama)
python scripts/04_run_calibration.py --political_context ...            # Stage 4 (ideology paragraph)
python scripts/04_run_calibration.py --seed 1  # seed stability
python scripts/04_run_calibration.py --seed 2

# Section 4 experiment:
python scripts/run_context_experiment_v2.py --peace v3_final_noPC --no-pc

# Aggregate + figures:
python scripts/05_compute_metrics.py
python scripts/07_resample_vs_cot.py           # apply RNG correction (see docs/sampling_correction.md)
python scripts/make_tables.py
python scripts/make_figures.py
```

## Configurations — paper labels vs internal ids

The paper uses contiguous labels **C0-C10** (Table 4, Appendix B). The
internal id used in archived JSONL filenames is one step out at C2 because
the historical numbering skipped C2:

| Paper id | Internal id (`legacy_id`) |
|----------|---------------------------|
| C0-C1    | C0-C1 |
| C2       | C3 |
| C3       | C4 |
| C4       | C5 |
| C5       | C6 |
| **C6**   | **C7** (the paper's calibrated protocol config) |
| C7       | C8 |
| C8       | C9 |
| C9       | C10 |
| C10      | C11 |

`config/ablations.py` encodes both ids on every entry, and `config/config_id_mapping.csv`
is a human-readable dump of the mapping. Every user-facing artefact (Excel
`Config` column, table CSVs, figure captions) uses **paper labels**.

## Paper table & figure ↔ script mapping

| Paper | Script | Output |
|-------|--------|--------|
| Table 1 | `make_tables.py --table 1` | `outputs/tables/Table1_cross_model.csv` |
| Table 2 | `make_tables.py --table 2` | `outputs/tables/Table2_final_protocols.csv` |
| Table 3 | `make_tables.py --table 3` | `outputs/tables/Table3_persona_variables.csv` |
| Table 4 | `make_tables.py --table 4` | `outputs/tables/Table4_persona_configurations.csv` |
| Tables 5-6 (Stage 1) | `make_tables.py --table 5` | `outputs/tables/Table5_stage1_*.csv` |
| Tables 7-9 (Stage 2) | `make_tables.py --table 7` | `outputs/tables/Table7-9_stage2_*.csv` |
| Tables 10-11 (Stage 3) | `make_tables.py --table 10` | `outputs/tables/Table10-11_stage3_*.csv` |
| Tables 12-13 (Stage 4) | `make_tables.py --table 12` | `outputs/tables/Table12-13_stage4_*.csv` |
| Tables 14 (seed stability) | `make_tables.py --table 14` | `outputs/tables/Table14_stage5_*.csv` |
| **Table 15** (seed-pair agreement) | `make_tables.py --table 15` | `outputs/tables/Table15_seed_pair_agreement.csv` |
| Table 16 (Appendix G, binary) | `make_tables.py --table 16` | `outputs/tables/Table16_transfer_binary.csv` |
| Table 17 (Appendix G, ordinal) | `make_tables.py --table 17` | `outputs/tables/Table17_transfer_ordinal.csv` |
| Fig 1 (calibration funnel) | `make_figures.py --fig 1` | `figures/Fig01_calibration_funnel.pdf` |
| Fig 2 (effect summary) | `make_figures.py --fig 2` | `figures/Fig02_effect_summary.pdf` |
| Fig 3 (between-group forest) | `make_figures.py --fig 3` | `figures/Fig03_between_group_forest.pdf` |
| Fig 4 (subgroup context-effect forest) | `make_figures.py --fig 4` | `figures/Fig04_subgroup_context_forest.pdf` |
| Figs 5-6 (distributions) | `make_figures.py --fig 5` | `figures/Fig05_legitimacy_distribution.pdf`, `Fig06_behavioral_distribution.pdf` |
| Figs 7-8 (subgroup interactions) | `make_figures.py --fig 7` | `figures/Fig07_subgroup_interaction_behavioral.pdf`, `Fig08_subgroup_interaction_legitimacy.pdf` |
| Figs 9-10 (NUTS-1 maps) | `make_figures.py --fig 9` | `figures/Fig09_regional_legitimacy_map.pdf`, `Fig10_regional_behavioral_map.pdf` |

## Model table

| Provider | Model identifier | Access dates | Temperature | top_p | max_tokens (direct / VS-CoT / +CoT) | Seeds |
|----------|------------------|--------------|-------------|-------|-------------------------------------|-------|
| OpenAI | `gpt-4o-mini` | 2026-05 → 2026-08 | 0.0, 0.4, 0.8 | 1.0 (default) | 10 / 250 / 200 | 0, 1, 2 |
| OpenAI | `gpt-5.4-mini` | 2026-06 → 2026-08 | 0.8 | 1.0 | 250 (VS-CoT) | 0, 1, 2 |
| Anthropic | `claude-haiku-4-5-20251001` | 2026-07 → 2026-08 | 0.8 | 1.0 | 250 (VS-CoT) | 0 |
| Google | `gemini-2.5-flash-lite` | 2026-06 → 2026-08 | 0.0, 0.4, 0.8 | 1.0 | 250 (VS-CoT) | 0, 1, 2 |
| OpenRouter → Meta | `meta-llama/llama-3.3-70b-instruct` | 2026-07 | 0.8 | 1.0 | 250 (VS-CoT) | 0, 1, 2 |

`top_p` is provider default (`1.0`) throughout. Access dates are the range
across which cells for a given model were populated in the JSONL cache;
Kına/Ekmen (authors) can replace them with more precise dates from the API
dashboards.

## Seed handling

* **Direct answering** (Stages 1, 3, most of 4). At `T=0` the seed is
  passed to the provider `seed` field so responses are reproducible.
  At `T>0` the provider seed has no effect and replication is only within
  drift.
* **Verbalized sampling with chain-of-thought** (VS-CoT). The model
  returns a probability distribution over answer categories. A single
  `numpy.random.Generator` is initialised at the base seed for the run
  and advanced across respondents (see `docs/sampling_correction.md`).
* **Section 4 experiment**. `run_context_experiment_v2.py` fixes a
  per-(respondent, condition) draw seed via a hash so that the same pair
  is reproducible across reruns even at `T>0`.

## Known deviations

* The Llama 3.3 70B run at paper config C2 is incomplete (1,783 of 2,588
  respondents in the archived cache).
* Invalid completions are dropped, not imputed. Sample-size columns
  reflect the filter.
* Stage 3 and Stage 4 VS-CoT metric CSVs are the corrected output of
  `07_resample_vs_cot.py`. See `docs/sampling_correction.md`.

## How to cite

```
@article{kina_ekmen_2026,
  author  = {Kına, M. Fuat and Ekmen, Helin Yaren},
  title   = {Calibrating Synthetic Respondents: A Within-Persona Experiment
             on Public Reactions to Protest in Turkey},
  year    = {2026},
  month   = {9},
  journal = {SocArXiv},
  doi     = {10.31235/osf.io/hp7gz_v1},
  url     = {https://osf.io/preprints/socarxiv/hp7gz_v1}
}
```

## Contact

Helin Yaren Ekmen — <helinyarenekmen@gmail.com>
