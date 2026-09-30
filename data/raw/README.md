# TGSS 2024 raw microdata

## Licence

The 2024 Turkish General Social Survey (TGSS 2024) is distributed under
**Creative Commons Attribution-NonCommercial 4.0 International** (CC BY-NC 4.0),
the same licence used for all TGSS-derived material in this repository. See
`../../LICENSE-DATA`.

## Redistribution

The raw SPSS `.sav` file is **not** redistributed here. A cleaned extract of
the full microdata is included under `../derived/tgss2024_clean.csv` under
the same CC BY-NC 4.0 terms, so Path B ("reproduce every table and figure
from archived outputs") does **not** need the SPSS file.

## Downloading the SPSS file (only for Path A)

If you want to regenerate `data/derived/tgss2024_clean.csv` from source,
download the SPSS file from Zenodo and save it here as
`data/raw/TGSS2024.sav`:

  * DOI: [10.5281/zenodo.18721350](https://doi.org/10.5281/zenodo.18721350)
  * Cite exactly as printed on the Zenodo record "Cite as" box (see the
    project's `../../CITATION.cff` under `references`).

Then run `python scripts/00_prepare_data.py`.
