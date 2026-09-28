"""
00_prepare_data.py — One-time SAV → CSV conversion for TGSS 2024.

Outputs:
  data/tgss2024_clean.csv   — UTF-8 CSV, all 2615 rows × 664 cols
  data/tgss2024_meta.json   — variable labels, value labels, missing codes

Run once; commit both outputs. All downstream scripts read the CSV.
See PERSONA_SPEC.md §15 Step 0.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
SAV_PATH = ROOT / "data" / "TGSS2024.sav"
CSV_PATH = ROOT / "data" / "tgss2024_clean.csv"
META_PATH = ROOT / "data" / "tgss2024_meta.json"
LOG_PATH = ROOT / "logs" / f"prepare_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

EXPECTED_ROWS = 2615
EXPECTED_COLS = 664

# Turkish spot-check strings — if any appear garbled, encoding failed.
TURKISH_SPOT_VARS = ["gender", "degree", "degurba", "eidfinal", "marital"]


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH)],
    )
    return logging.getLogger(__name__)


def load_sav(log: logging.Logger) -> tuple[pd.DataFrame, pyreadstat.metadata_container]:
    log.info(f"Reading SAV: {SAV_PATH}")
    df, meta = pyreadstat.read_sav(str(SAV_PATH))
    log.info(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df, meta


def validate_shape(df: pd.DataFrame, log: logging.Logger) -> None:
    errors = []
    if df.shape[0] != EXPECTED_ROWS:
        errors.append(f"Row count {df.shape[0]} ≠ expected {EXPECTED_ROWS}")
    if df.shape[1] != EXPECTED_COLS:
        errors.append(f"Column count {df.shape[1]} ≠ expected {EXPECTED_COLS}")
    if errors:
        for e in errors:
            log.error(e)
        raise ValueError("Shape validation failed — check SAV file version.")
    log.info(f"Shape OK: {EXPECTED_ROWS} rows × {EXPECTED_COLS} cols")


def spot_check_turkish(df: pd.DataFrame, meta: pyreadstat.metadata_container, log: logging.Logger) -> None:
    """Verify Turkish characters in value labels aren't garbled."""
    log.info("Spot-checking Turkish character encoding in value labels...")
    turkish_chars = set("çşğüöıİÇŞĞÜÖ")
    found_turkish = False
    for var in TURKISH_SPOT_VARS:
        label_map = meta.variable_value_labels.get(var, {})
        for label in label_map.values():
            if any(c in label for c in turkish_chars):
                log.info(f"  {var}: '{label}' — Turkish chars OK")
                found_turkish = True
                break
    if not found_turkish:
        log.warning("No Turkish characters found in spot-check vars — verify encoding manually.")


def print_nan_summary(df: pd.DataFrame, log: logging.Logger) -> None:
    nan_counts = df.isna().sum()
    n_with_nan = (nan_counts > 0).sum()
    log.info(f"NaN summary: {n_with_nan} variables have at least one missing value")
    top_nan = nan_counts[nan_counts > 0].sort_values(ascending=False).head(10)
    if not top_nan.empty:
        log.info("Top 10 variables by NaN count:")
        for var, n in top_nan.items():
            log.info(f"  {var}: {n} ({n/len(df)*100:.1f}%)")


def build_meta(df: pd.DataFrame, meta: pyreadstat.metadata_container) -> dict:
    return {
        "generated_at": datetime.now().isoformat(),
        "source_file": str(SAV_PATH),
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "column_labels": meta.column_names_to_labels,
        "value_labels": {
            var: {str(int(k)): v for k, v in labels.items()}
            for var, labels in meta.variable_value_labels.items()
        },
        "variable_measure": meta.variable_measure,
        "missing_user_values": meta.missing_user_values,
        "missing_ranges": meta.missing_ranges,
    }


def export_csv(df: pd.DataFrame, log: logging.Logger) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(CSV_PATH), index=False, encoding="utf-8")
    log.info(f"Wrote CSV: {CSV_PATH}")


def verify_roundtrip(log: logging.Logger) -> None:
    """Read the CSV back and spot-check 10 variables against in-memory df."""
    df_check = pd.read_csv(str(CSV_PATH), encoding="utf-8")
    if df_check.shape[0] != EXPECTED_ROWS or df_check.shape[1] != EXPECTED_COLS:
        raise ValueError(f"Roundtrip shape mismatch: {df_check.shape}")
    log.info(f"Roundtrip OK: CSV reads back as {df_check.shape[0]} × {df_check.shape[1]}")


def export_meta(meta_dict: dict, log: logging.Logger) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)
    log.info(f"Wrote metadata: {META_PATH}")


def main() -> None:
    log = setup_logging()
    log.info("=== 00_prepare_data.py start ===")

    df, meta = load_sav(log)
    validate_shape(df, log)
    spot_check_turkish(df, meta, log)
    print_nan_summary(df, log)

    meta_dict = build_meta(df, meta)
    export_csv(df, log)
    verify_roundtrip(log)
    export_meta(meta_dict, log)

    log.info("=== Done. Commit data/tgss2024_clean.csv and data/tgss2024_meta.json ===")


if __name__ == "__main__":
    main()
