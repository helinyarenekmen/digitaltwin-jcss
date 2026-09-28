"""
01_persona_engine.py — Layer A persona generation.

Produces 11 × 2,615 = 28,765 structured Markdown files, one per
respondent × ablation config. Zero API calls — all local.

Usage:
    python scripts/01_persona_engine.py           # full run
    python scripts/01_persona_engine.py --dry-run # first 5 respondents, C0 + C1 only

Resume: skips files that already exist and are non-empty.
Logging: progress every 100 respondents; final manifest appended per file written.

See PERSONA_SPEC.md §7 Step 2, §8, §9.
"""

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.paths import PERSONA_DIR as _PERSONA

CSV_PATH       = ROOT / "data" / "derived" / "tgss2024_clean.csv"
EXCEL_PATH     = ROOT / "data" / "derived" / "TGSS2024_Persona_Variables.xlsx"
LABELS_PATH    = ROOT / "data" / "derived" / "display_labels.json"
LAYER_A_DIR    = _PERSONA
MANIFEST_PATH  = ROOT / "cache" / "layer_a_manifest.csv"

VALID_GROUP_PREFIXES = ("1.", "2.", "3.", "4.", "5.", "6.")

# §5 — render order (alias → position). Demographics first, social_psych last.
GROUP_ORDER: dict[str, int] = {
    "demographics": 0,
    "belief":       1,
    "political":    2,
    "social":       3,
    "economic":     4,
    "social_psych": 5,
}

# Turkish section headers for each alias
GROUP_HEADERS: dict[str, str] = {
    "demographics": "DEMOGRAFİ",
    "belief":       "İNANÇ, İDEOLOJİ, KİMLİK",
    "political":    "SİYASİ DEĞERLER VE TUTUMLAR",
    "social":       "SOSYAL DEĞERLER VE TUTUMLAR",
    "economic":     "EKONOMİK DURUM VE TUTUMLAR",
    "social_psych": "SOSYAL-PSİKOLOJİK",
}

# Outcome variables — must never appear in any persona file (leakage guard)
OUTCOME_VARS = {"pacdemons", "womenwork", "neilang"}


# ---------------------------------------------------------------------------
# Excel loading
# ---------------------------------------------------------------------------

def parse_options(raw: str) -> dict[int, str] | None:
    """
    Parse Options cell into {numeric_code: label} dict.
    Returns None for open-ended numeric variables ('[Açık uçlu / sayısal]').
    Handles both semicolon-separated and newline-separated formats.
    """
    if not isinstance(raw, str):
        return {}
    raw = raw.strip()
    if raw.startswith("[Açık uçlu"):
        return None  # continuous — render raw value

    result: dict[int, str] = {}
    # Split on semicolons or newlines, whichever appears
    parts = re.split(r"[;\n]", raw)
    for part in parts:
        part = part.strip()
        if "=" not in part:
            continue
        key_str, _, val = part.partition("=")
        key_str = key_str.strip()
        val = val.strip()
        try:
            result[int(float(key_str))] = val
        except ValueError:
            continue
    return result


def load_excel() -> tuple[dict[str, str], dict[str, dict[int, str] | None]]:
    """
    Returns:
      alias_by_varcode  — {var_code: group_alias}
      options_by_varcode — {var_code: {code: label} | None}
    """
    df = pd.read_excel(EXCEL_PATH, sheet_name="Variables")

    # Import here to avoid circular at module level
    from config.ablations import GROUP_ALIASES
    # Invert GROUP_ALIASES: excel_string → alias
    excel_to_alias = {v: k for k, v in GROUP_ALIASES.items()}

    alias_by_varcode: dict[str, str] = {}
    options_by_varcode: dict[str, dict[int, str] | None] = {}

    for _, row in df.iterrows():
        group_raw = row["Variable Group"]
        code = row["Variable Code"]
        if not isinstance(group_raw, str) or not group_raw.startswith(VALID_GROUP_PREFIXES):
            continue
        if not isinstance(code, str):
            continue
        code = code.strip()
        if code in OUTCOME_VARS:
            continue  # hard leakage guard

        alias = excel_to_alias.get(group_raw)
        if alias is None:
            print(f"WARNING: unmapped group {group_raw!r} for variable {code} — skipping")
            continue

        alias_by_varcode[code] = alias
        options_by_varcode[code] = parse_options(row.get("Options", ""))

    return alias_by_varcode, options_by_varcode


def load_display_labels() -> dict[str, str]:
    import json
    if not LABELS_PATH.exists():
        print(f"WARNING: {LABELS_PATH} not found — using variable codes as labels.")
        return {}
    with open(LABELS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_value(raw_value, options: dict[int, str] | None) -> str:
    """Convert a raw CSV value to its display string."""
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return "bilgi yok"
    if options is None:
        # Continuous numeric — render as integer if whole number, else float
        try:
            n = float(raw_value)
            return str(int(n)) if n == int(n) else str(n)
        except (ValueError, TypeError):
            return str(raw_value)
    # Categorical — look up label
    try:
        code = int(float(raw_value))
        return options.get(code, "bilgi yok")
    except (ValueError, TypeError):
        return "bilgi yok"


def render_persona(
    respondent_id: str,
    config_id: str,
    row: pd.Series,
    group_aliases: list[str],
    alias_by_varcode: dict[str, str],
    options_by_varcode: dict[str, dict[int, str] | None],
    display_labels: dict[str, str],
    drop_vars: tuple[str, ...] = (),
) -> tuple[str, int]:
    """
    Build the full Markdown text for one respondent × one config.
    Returns (markdown_text, n_variables_rendered).
    """
    included_aliases = set(group_aliases)

    # Collect lines per alias
    lines_by_alias: dict[str, list[str]] = {a: [] for a in included_aliases}

    drop_set = set(drop_vars)
    for var_code, alias in alias_by_varcode.items():
        if alias not in included_aliases:
            continue
        if var_code in drop_set:
            continue
        if var_code not in row.index:
            continue
        label = display_labels.get(var_code, var_code)
        value_str = render_value(row[var_code], options_by_varcode[var_code])
        lines_by_alias[alias].append(f"- {label}: {value_str}")

    # Sort groups by §5 render order
    ordered_aliases = sorted(included_aliases, key=lambda a: GROUP_ORDER.get(a, 99))

    sections: list[str] = [f"# Persona — {respondent_id} — Config {config_id}\n"]
    n_vars = 0
    for alias in ordered_aliases:
        group_lines = lines_by_alias[alias]
        if not group_lines:
            continue
        header = GROUP_HEADERS[alias]
        sections.append(f"## {header}")
        sections.extend(group_lines)
        sections.append("")  # blank line between sections
        n_vars += len(group_lines)

    return "\n".join(sections), n_vars


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def init_manifest() -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["respondent_id", "config_id", "path", "n_variables", "generated_at"])


def append_manifest(respondent_id: str, config_id: str, path: Path, n_vars: int) -> None:
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([respondent_id, config_id, str(path), n_vars, datetime.now().isoformat()])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 5 respondents under C0 and C1 only.")
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    print(f"  {len(df)} respondents, {len(df.columns)} columns")

    print("Loading Excel variable metadata...")
    alias_by_varcode, options_by_varcode = load_excel()
    print(f"  {len(alias_by_varcode)} persona variables mapped")

    print("Loading display labels...")
    display_labels = load_display_labels()

    from config.ablations import CONFIGS, CONFIGS_BY_ID, RUN_PRIORITY
    configs = CONFIGS

    if args.dry_run:
        configs = [CONFIGS_BY_ID["C0"], CONFIGS_BY_ID["C1"]]
        df = df.head(5)
        print(f"DRY RUN: 5 respondents × {len(configs)} configs")

    init_manifest()

    total = len(df) * len(configs)
    processed = 0
    skipped = 0

    for config in configs:
        config_dir = LAYER_A_DIR / config.config_id
        config_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{config.config_id}] {config.name}")

        for i, (_, row) in enumerate(df.iterrows()):
            respondent_id = f"TGSS_{int(row['id']):04d}"
            out_path = config_dir / f"{respondent_id}.md"

            # Resume: skip existing non-empty files
            if out_path.exists() and out_path.stat().st_size > 0:
                skipped += 1
                processed += 1
                continue

            markdown, n_vars = render_persona(
                respondent_id=respondent_id,
                config_id=config.config_id,
                row=row,
                group_aliases=config.group_aliases,
                alias_by_varcode=alias_by_varcode,
                options_by_varcode=options_by_varcode,
                display_labels=display_labels,
                drop_vars=getattr(config, "drop_vars", ()),
            )

            out_path.write_text(markdown, encoding="utf-8")
            append_manifest(respondent_id, config.config_id, out_path, n_vars)

            processed += 1
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(df)} respondents done")

        print(f"  {len(df)} respondents complete")

    print(f"\nDone. {processed} total ({skipped} skipped/resumed).")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
