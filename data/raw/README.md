# TGSS 2024 microdata — retrieval instructions

The raw TGSS 2024 microdata **is not redistributed in this repository**. It is
covered by a separate license held by the survey producers.

## How to obtain

Download the SPSS file `TGSS2024.sav` from the Zenodo record:

**DOI**: [10.5281/zenodo.18721350](https://doi.org/10.5281/zenodo.18721350)

Save it here as:

```
data/raw/TGSS2024.sav
```

The first step of the pipeline (`scripts/00_prepare_data.py`) reads this file
and produces the cleaned CSV / meta JSON already shipped under
`data/derived/`, so if you only intend to reproduce results from the archived
model completions you do **not** need to redownload the SPSS file — the
derived tables suffice.

## Citation

If you use the TGSS 2024 microdata, please cite the original TGSS producers
per the license terms available on the Zenodo record.
