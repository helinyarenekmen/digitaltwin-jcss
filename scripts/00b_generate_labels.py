"""
00b_generate_labels.py — One-time display label generation for persona variables.

Reads Question column from TGSS2024_Persona_Variables.xlsx; produces short
Turkish labels (2-4 words) for each persona variable via GPT-4o-mini, with a
≤5-word shortcut that skips the API.

Output: outputs/display_labels.json  (committed; never regenerated at runtime)

After running, open the JSON, fix any odd labels (expect 5-15 corrections),
then commit the final version. 01_persona_engine.py reads this file only.

See PERSONA_SPEC.md §7 Step 1.
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = ROOT / "TGSS2024_Persona_Variables.xlsx"
OUTPUT_PATH = ROOT / "outputs" / "display_labels.json"

VALID_GROUP_PREFIXES = ("1.", "2.", "3.", "4.", "5.", "6.")

SYSTEM_PROMPT = (
    "Sen bir araştırma asistanısın. Sana Türkçe bir anket sorusu verilecek. "
    "Bu soruyu, bir persona profilinde kısa etiket olarak kullanılmak üzere "
    "2-4 kelimelik Türkçe bir etikete dönüştür. "
    "Anlam kaybı olmadan özetle. Sadece etiketi döndür, başka hiçbir şey yazma."
)


def load_persona_variables() -> pd.DataFrame:
    df = pd.read_excel(EXCEL_PATH, sheet_name="Variables")
    mask = df["Variable Group"].apply(
        lambda g: isinstance(g, str) and g.startswith(VALID_GROUP_PREFIXES)
    )
    return df[mask][["Variable Code", "Question"]].dropna(
        subset=["Variable Code", "Question"]
    )


def is_short(question: str) -> bool:
    """≤5 words → use as-is, skipping the API call."""
    return len(question.split()) <= 5


def clean_label(label: str) -> str:
    label = label.strip().strip('"').strip("'").strip()
    label = label.rstrip(".:")
    return label


def call_gpt(client: OpenAI, question: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=30,
            )
            return clean_label(response.choices[0].message.content)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  ✗ API error after {retries} attempts: {e}")
                return f"REVIEW_{question[:40]}"
    return f"REVIEW_{question[:40]}"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing labels so we don't re-call for already-done variables.
    existing: dict[str, str] = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Resuming — {len(existing)} labels already present.")

    df = load_persona_variables()
    print(f"Persona variables to label: {len(df)}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        import getpass
        api_key = getpass.getpass("OpenAI API key: ").strip()
    if not api_key:
        print("ERROR: API key boş.", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    labels: dict[str, str] = dict(existing)
    shortcut_count = 0
    api_count = 0

    for _, row in df.iterrows():
        code = str(row["Variable Code"]).strip()
        question = str(row["Question"]).strip()

        if code in labels:
            continue  # already labeled

        if is_short(question):
            labels[code] = question
            shortcut_count += 1
            print(f"  [shortcut] {code}: {question!r}")
        else:
            label = call_gpt(client, question)
            labels[code] = label
            api_count += 1
            print(f"  [gpt]      {code}: {label!r}")

        # Save after every variable — safe resume if interrupted.
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {shortcut_count} shortcuts, {api_count} API calls.")
    print(f"Output: {OUTPUT_PATH}")
    print("\nNext step: open outputs/display_labels.json, fix any odd labels, commit.")


if __name__ == "__main__":
    main()
