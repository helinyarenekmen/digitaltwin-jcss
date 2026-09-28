"""
09_run_calibration_gemini.py — Gemini (Google) modelleri için calibration runner.

08_run_calibration_claude.py'nin Gemini versiyonu. Aynı CLI flag'leri:
  --mode screening
  --sampling direct|vs_cot
  --configs C1,C3,C4,C7
  --outcomes pacdemons,womenwork,neilang
  --temperature 0.8
  --address_mode ben|sen
  --seed 0
  --model gemini-2.5-flash-lite  (default)

Output dosyaları aynı klasörlere yazılır (screening/, screening_vs_cot_T08/, vb.)
ama filename'de model alias 'geminiflashlite' geçer:
  C7_geminiflashlite_T08_ben_direct_nocot_s0.jsonl

Bu sayede 05_compute_metrics.py ve 06_plot_screening.py
--model gemini-2.5-flash-lite ile direkt çalışır (alias'ı eklediğimizde).

Usage:
  python scripts/09_run_calibration_gemini.py --mode screening \\
      --sampling direct --configs C7 --outcomes pacdemons \\
      --temperature 0.8 --model gemini-2.5-flash-lite
"""

import argparse
import asyncio
import getpass
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CALIB_DIR     = Path("/Users/helinekmen/Library/Caches/digitaltwin_calibration")
SCREENING_DIR = CALIB_DIR / "screening"
LAYER_A_DIR   = Path("/Users/helinekmen/Library/Caches/digitaltwin_layer_a")
CSV_PATH      = ROOT / "data" / "tgss2024_clean.csv"

OUTCOME_SPECS: dict[str, dict] = {
    "pacdemons": {
        "question": (
            "Son 12 ay içinde Türkiye'de işlerin düzelmesini sağlamak veya "
            "kötüye gitmesini önlemek için sokak eylemlerine katıldınız mı?"
        ),
        "options": {1: "Evet", 2: "Hayır"},
        "valid_range": (1, 2),
        "minority_code": 1,
    },
    "womenwork": {
        "question": "Kadın tam zamanlı çalıştığında aile hayatı olumsuz etkilenir.",
        "options": {
            1: "Hiç katılmıyorum",
            2: "Katılmıyorum",
            3: "Ne katılıyorum ne katılmıyorum",
            4: "Katılıyorum",
            5: "Tamamen katılıyorum",
        },
        "valid_range": (1, 5),
        "minority_code": None,
    },
    "neilang": {
        "question": (
            "Bir yeni komşu evinizin yanına taşınacak olsa, farklı dili konuşan "
            "insanların komşunuz olmasından ne kadar rahatsız olur veya olmazsınız?"
        ),
        "options": {
            1: "Hiç rahatsız olmazdım",
            2: "Biraz rahatsız olurdum",
            3: "Çok rahatsız olurdum",
        },
        "valid_range": (1, 3),
        "minority_code": None,
    },
}

SCREENING_DEFAULT = dict(
    model="gemini-2.5-flash-lite",
    temperature=0.0,
    address_mode="ben",
    sampling="direct",
    cot=False,
    seed=0,
)

MODEL_ALIASES = {
    "gemini-2.5-flash-lite": "geminiflashlite",
    "gemini-2.5-flash":      "geminiflash",
    "gemini-2.5-pro":        "geminipro",
}

_TEMP_LABELS = {0.0: "T0", 0.3: "T03", 0.4: "T04", 0.7: "T07", 0.8: "T08", 1.0: "T1"}

SYSTEM_BEN = (
    "Aşağıda Türkiye'de yaşayan birinin kişisel profili yer almaktadır. "
    "Bu kişinin yerine geçerek bir ankete cevap vereceksin. "
    "Cevabını sadece verilen seçenekler arasından seç. "
    "Sayı olarak cevap ver, açıklama ekleme."
)

SYSTEM_SEN = (
    "Aşağıda Türkiye'de yaşayan birinin kişisel profili yer almaktadır. "
    "Onun bakış açısından bir ankete cevap vereceksin. "
    "Sen, aşağıda tanımlanan kişisin. "
    "Cevabını sadece verilen seçenekler arasından seç. "
    "Sayı olarak cevap ver, açıklama ekleme."
)

VS_COT_SUFFIX = (
    "\n\nBu kişinin sana sorulan soruya nasıl cevap vereceğini düşün. "
    "Gerçek hayatta aynı profile sahip insanların cevapları çeşitlilik gösterir; "
    "kişinin gerçek cevabı hakkında bir miktar belirsizlik vardır. "
    "Önce kısa bir analiz yap (1-2 cümle), sonra bu kişinin "
    "**her bir seçeneği seçme olasılığını** ver.\n\n"
    "Format:\n"
    "ANALİZ: <kısa analiz>\n"
    "DAĞILIM: {\"1\": 0.X, \"2\": 0.X, ...}\n\n"
    "Önemli:\n"
    "- TÜM seçeneklere bir olasılık ata (0.0 da olabilir).\n"
    "- Olasılıkların toplamı 1.0 olmalı.\n"
    "- Aşırı güvenli olma; belirsizlik varsa olasılıkları dağıt."
)

# Explicit CoT — added to top of VS or Direct system prompt to force extended reasoning.
EXPLICIT_COT_SUFFIX_DIRECT = (
    "\n\nCevabını vermeden ÖNCE bu kişinin profilindeki ilgili sinyalleri "
    "adım adım kısaca değerlendir (2-3 cümle). Sonra cevabını "
    "şu formatta ver: 'CEVAP: X' — burada X seçenek numarasıdır."
)
EXPLICIT_COT_SUFFIX_VSCOT = (
    "\n\nÖnemli: ANALİZ kısmında profildeki ilgili her sinyali (demografi, "
    "siyasi yönelim, sosyo-ekonomik durum, vb.) adım adım değerlendir, "
    "sonra DAĞILIM'ı bu çıkarsama temelinde ver."
)

# Political-context prepended paragraph for Phase 15 ablation.
POLITICAL_CONTEXT_TR = (
    "Türkiye'nin ideolojik yapısı iki temel eksen üzerinde şekillenir: "
    "sol-sağ siyasi ekseni ve seküler-dindar kültürel ekseni. "
    "Sol tarafta sosyalizm, feminizm, sosyal demokrasi ve Kürt ulusal "
    "hareketi birbirine yakın konumlanır. Sağ tarafta Türk milliyetçiliği "
    "ve Kemalizm yer alır. Muhafazakârlık ve İslamcılık kültürel eksenin "
    "dindar ucunu oluşturur."
)


def build_prompt(
    layer_a_text: str,
    outcome: str,
    address_mode: str,
    sampling: str,
    cot: bool = False,
    political_context: bool = False,
) -> tuple[str, str]:
    """Returns (system_instruction, user_content) for Gemini API."""
    spec = OUTCOME_SPECS[outcome]
    options_str = "\n".join(f"{k} = {v}" for k, v in spec["options"].items())
    system = SYSTEM_BEN if address_mode == "ben" else SYSTEM_SEN
    if political_context:
        system = POLITICAL_CONTEXT_TR + "\n\n" + system
    if sampling == "vs_cot":
        system = system + VS_COT_SUFFIX
        if cot:
            system = system + EXPLICIT_COT_SUFFIX_VSCOT
    elif cot:
        system = system + EXPLICIT_COT_SUFFIX_DIRECT
    ending = "Cevap (sadece sayı):" if sampling != "vs_cot" else ""
    user = (
        f"{layer_a_text}\n\n"
        f"---\n\n"
        f"Soru: {spec['question']}\n\n"
        f"Seçenekler:\n{options_str}"
        + (f"\n\n{ending}" if ending else "")
    )
    return system, user


def parse_direct(raw: str, valid_range: tuple[int, int]) -> int | None:
    m = re.search(r"\b(\d+)\b", raw.strip())
    if not m:
        return None
    val = int(m.group(1))
    lo, hi = valid_range
    return val if lo <= val <= hi else None


def parse_vs_cot(raw: str, valid_range: tuple[int, int], seed: int) -> int | None:
    import numpy as np
    m = re.search(r"DAĞILIM:\s*(\{[^}]+\})", raw, re.DOTALL)
    if not m:
        return None
    try:
        dist_raw = json.loads(m.group(1))
        options = {int(k): float(v) for k, v in dist_raw.items()}
        lo, hi = valid_range
        options = {k: v for k, v in options.items() if lo <= k <= hi}
        if not options:
            return None
        total = sum(options.values())
        keys = list(options.keys())
        probs = [options[k] / total for k in keys]
        rng = np.random.default_rng(seed)
        return int(rng.choice(keys, p=probs))
    except Exception:
        return None


def make_filename(config_id: str, model: str, temperature: float,
                  address_mode: str, sampling: str, cot: bool, seed: int,
                  political_context: bool = False) -> str:
    model_alias = MODEL_ALIASES.get(model, model.replace("-", "").replace(".", ""))
    temp_str = _TEMP_LABELS.get(temperature, f"T{temperature}".replace(".", ""))
    cot_str = "cot" if cot else "nocot"
    pc_token = "_pc" if political_context else ""
    return (f"{config_id}_{model_alias}_{temp_str}_{address_mode}_"
            f"{sampling}_{cot_str}{pc_token}_s{seed}.jsonl")


def load_done_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["respondent_id"])
            except Exception:
                pass
    return done


def append_result(jsonl_path: Path, record: dict) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def predict_one(
    client: genai.Client,
    semaphore: asyncio.Semaphore,
    respondent_id: str,
    config_id: str,
    outcome: str,
    layer_a_text: str,
    settings: dict,
) -> dict:
    spec = OUTCOME_SPECS[outcome]
    system, user = build_prompt(
        layer_a_text, outcome, settings["address_mode"], settings["sampling"],
        cot=settings.get("cot", False),
        political_context=settings.get("political_context", False),
    )
    record_base = dict(
        respondent_id=respondent_id,
        config_id=config_id,
        outcome=outcome,
        model=settings["model"],
        temperature=settings["temperature"],
        address_mode=settings["address_mode"],
        sampling=settings["sampling"],
        cot=settings["cot"],
        seed=settings["seed"],
        timestamp=datetime.now().isoformat(),
    )
    # Gemini analizleri OpenAI/Claude'tan uzun yazıyor → VS-CoT için
    # 400 token (150 yetmiyordu, %9 cevap DAĞILIM bitmeden kesiliyordu).
    max_out = 400 if settings["sampling"] == "vs_cot" else 10
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=settings["temperature"],
        max_output_tokens=max_out,
    )

    raw = ""
    prompt_tokens = 0
    completion_tokens = 0
    for attempt in range(2):
        for backoff_attempt in range(5):
            try:
                async with semaphore:
                    resp = await asyncio.wait_for(
                        client.aio.models.generate_content(
                            model=settings["model"],
                            contents=user,
                            config=config,
                        ),
                        timeout=90.0,
                    )
                raw = (resp.text or "").strip()
                if resp.usage_metadata:
                    prompt_tokens = resp.usage_metadata.prompt_token_count or 0
                    completion_tokens = resp.usage_metadata.candidates_token_count or 0
                break
            except Exception as e:
                err_str = str(e)
                rate_limited = ("429" in err_str
                                or "RESOURCE_EXHAUSTED" in err_str
                                or "rate" in err_str.lower())
                if rate_limited and backoff_attempt < 4:
                    print(f"    [{respondent_id}] rate-limit, backing off "
                          f"{2**backoff_attempt}s...")
                    await asyncio.sleep(2 ** backoff_attempt)
                else:
                    print(f"    [{respondent_id}] API error (attempt {attempt},"
                          f"{backoff_attempt}): {err_str[:140]}")
                    return {**record_base, "predicted_value": None,
                            "raw_response": f"API_ERROR: {e}",
                            "prompt_tokens": 0, "completion_tokens": 0,
                            "parse_status": "api_error"}

        if settings["sampling"] == "vs_cot":
            val = parse_vs_cot(raw, spec["valid_range"], settings["seed"])
        else:
            val = parse_direct(raw, spec["valid_range"])
        if val is not None:
            return {**record_base, "predicted_value": val, "raw_response": raw,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "parse_status": "ok"}
        if attempt == 0:
            continue

    return {**record_base, "predicted_value": None, "raw_response": raw,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "parse_status": "parse_failure"}


async def run_cell(
    client: genai.Client,
    semaphore: asyncio.Semaphore,
    config_id: str,
    outcome: str,
    respondents: pd.DataFrame,
    settings: dict,
    out_path: Path,
) -> None:
    done_ids = load_done_ids(out_path)
    remaining = respondents[~respondents["respondent_id"].isin(done_ids)]

    if remaining.empty:
        print(f"  [{config_id}/{outcome}] already complete — skipping")
        return

    print(f"  [{config_id}/{outcome}] {len(remaining)} respondents remaining "
          f"({len(done_ids)} already done)")

    respondent_ids: list[str] = remaining["respondent_id"].tolist()
    t0 = time.perf_counter()
    items: list[tuple[str, str]] = []
    for rid in respondent_ids:
        p = LAYER_A_DIR / config_id / f"{rid}.md"
        if not p.exists():
            print(f"  WARNING: missing {p.name}")
            continue
        items.append((rid, p.read_text(encoding="utf-8")))
    print(f"  [{config_id}/{outcome}] {len(items)} files loaded in "
          f"{time.perf_counter()-t0:.1f}s, starting API calls...")

    tasks = [
        predict_one(client, semaphore, rid, config_id, outcome, text, settings)
        for rid, text in items
    ]

    completed = 0
    parse_failures = 0
    for coro in asyncio.as_completed(tasks):
        record = await coro
        append_result(out_path, record)
        completed += 1
        if record["parse_status"] == "parse_failure":
            parse_failures += 1
        if completed % 100 == 0:
            print(f"    {completed}/{len(tasks)} done "
                  f"({parse_failures} parse failures so far)")

    pct_fail = parse_failures / completed * 100 if completed else 0
    status = "⚠ HIGH FAILURE RATE" if pct_fail > 2 else "OK"
    print(f"  [{config_id}/{outcome}] complete — {parse_failures}/{completed} "
          f"parse failures ({pct_fail:.1f}%) {status}")


async def screening_mode(client: genai.Client, args) -> None:
    from configs.ablations import CONFIGS

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df["respondent_id"] = df["id"].apply(lambda x: f"TGSS_{int(x):04d}")

    configs = CONFIGS
    all_outcomes = ["pacdemons", "womenwork", "neilang"]
    if args.outcomes:
        wanted = {o.strip() for o in args.outcomes.split(",")}
        outcomes = [o for o in all_outcomes if o in wanted]
        print(f"Outcomes filtered: {outcomes}")
    else:
        outcomes = all_outcomes

    settings = SCREENING_DEFAULT.copy()
    settings["sampling"] = args.sampling
    if args.temperature is not None:
        settings["temperature"] = args.temperature
    if args.seed is not None:
        settings["seed"] = args.seed
    if args.address_mode is not None:
        settings["address_mode"] = args.address_mode
    if args.model is not None:
        settings["model"] = args.model
    if args.cot:
        settings["cot"] = True
    if args.political_context:
        settings["political_context"] = True

    temp_str = _TEMP_LABELS.get(settings["temperature"],
                                f"T{settings['temperature']}".replace(".", ""))
    if args.sampling == "direct" and settings["temperature"] == 0.0:
        base_dir = SCREENING_DIR
    elif args.sampling == "direct":
        base_dir = SCREENING_DIR.parent / f"screening_{temp_str}"
    elif settings["temperature"] == 0.0:
        base_dir = SCREENING_DIR.parent / "screening_vs_cot"
    else:
        base_dir = SCREENING_DIR.parent / f"screening_vs_cot_{temp_str}"

    if args.configs:
        wanted = {c.strip() for c in args.configs.split(",")}
        configs = [c for c in CONFIGS if c.config_id in wanted]
        print(f"Configs filtered: {[c.config_id for c in configs]}")

    print(f"=== Gemini calibration | model={settings['model']} | "
          f"sampling={args.sampling} | T={settings['temperature']} | "
          f"output_dir={base_dir} ===")

    semaphore = asyncio.Semaphore(args.concurrency)

    for outcome in outcomes:
        lo, hi = OUTCOME_SPECS[outcome]["valid_range"]
        outcome_df = df[df[outcome].between(lo, hi)].copy()
        out_dir = base_dir / outcome
        out_dir.mkdir(parents=True, exist_ok=True)

        for config in configs:
            filename = make_filename(
                config.config_id, settings["model"], settings["temperature"],
                settings["address_mode"], settings["sampling"],
                settings["cot"], settings["seed"],
                political_context=settings.get("political_context", False),
            )
            out_path = out_dir / filename
            await run_cell(client, semaphore, config.config_id, outcome,
                           outcome_df, settings, out_path)


def get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Google AI Studio (Gemini) API key: ").strip()
    if not api_key:
        print("ERROR: API key boş.", file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=api_key)


async def amain(args) -> None:
    client = get_client()
    print(f"=== 09_run_calibration_gemini.py | mode={args.mode} | "
          f"concurrency={args.concurrency} ===")
    if args.mode == "screening":
        await screening_mode(client, args)
    else:
        print(f"Mode '{args.mode}' not implemented for Gemini script.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["screening"], default="screening")
    parser.add_argument("--sampling", choices=["direct", "vs_cot"], default="direct")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Gemini Flash Lite için varsayılan 5.")
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--outcomes", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--address_mode", choices=["ben", "sen"], default=None)
    parser.add_argument("--cot", action="store_true",
                        help="Add explicit chain-of-thought suffix (Phase 13/14 ablation).")
    parser.add_argument("--political_context", action="store_true",
                        help="Prepend Turkish political-ideology context (Phase 15 ablation). "
                             "Filename'de _pc_ olarak görünür.")
    parser.add_argument("--model", type=str, default=None,
                        help="Default: gemini-2.5-flash-lite")
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
