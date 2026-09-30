"""
10_run_calibration_openrouter.py — OpenRouter via OpenAI-compatible API.

Llama 3.3 70B Instruct (ve diğer OpenRouter modelleri) için runner.
OpenRouter OpenAI SDK ile çalışır → 04_run_calibration.py'nin paralel versiyonu,
sadece base_url ve API key değişir.

Aynı CLI flag'leri:
  --mode screening
  --sampling direct|vs_cot
  --configs C1,C3,C4,C7
  --outcomes pacdemons,womenwork,neilang
  --temperature 0.8
  --address_mode ben|sen
  --seed 0
  --model meta-llama/llama-3.3-70b-instruct  (default)

Filename alias: 'llama33-70b' → C7_llama3370b_T08_ben_direct_nocot_s0.jsonl

Usage:
  python3 scripts/10_run_calibration_openrouter.py --sampling direct \\
      --configs C7 --outcomes pacdemons --temperature 0.8
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import CACHE_DIR as _CACHE, PERSONA_DIR as _PERSONA

import pandas as pd
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CALIB_DIR     = Path(str(_CACHE))
SCREENING_DIR = CALIB_DIR / "screening"
LAYER_A_DIR   = Path(str(_PERSONA))
CSV_PATH      = ROOT / "data" / "tgss2024_clean.csv"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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
    model="meta-llama/llama-3.3-70b-instruct",
    temperature=0.0,
    address_mode="ben",
    sampling="direct",
    cot=False,
    seed=0,
)

MODEL_ALIASES = {
    "meta-llama/llama-3.3-70b-instruct":     "llama3370b",
    "meta-llama/llama-3.1-70b-instruct":     "llama3170b",
    "meta-llama/llama-3.1-8b-instruct":      "llama318b",
    "mistralai/mistral-large-2411":          "mistrallarge",
    "qwen/qwen-2.5-72b-instruct":            "qwen2572b",
    "deepseek/deepseek-chat":                "deepseekchat",
}

_TEMP_LABELS = {0.0: "T0", 0.3: "T03", 0.4: "T04", 0.7: "T07", 0.8: "T08", 1.0: "T1"}

SYSTEM_BEN = (
    "Aşağıda Türkiye'de yaşayan birinin kişisel profili yer almaktadır. "
    "Bu kişinin instead of geçerek bir ankete cevap vereceksin. "
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


def build_prompt(layer_a_text, outcome, address_mode, sampling):
    spec = OUTCOME_SPECS[outcome]
    options_str = "\n".join(f"{k} = {v}" for k, v in spec["options"].items())
    system = SYSTEM_BEN if address_mode == "ben" else SYSTEM_SEN
    if sampling == "vs_cot":
        system = system + VS_COT_SUFFIX
    ending = "Cevap (sadece sayı):" if sampling != "vs_cot" else ""
    user = (
        f"{layer_a_text}\n\n---\n\nSoru: {spec['question']}\n\n"
        f"Seçenekler:\n{options_str}"
        + (f"\n\n{ending}" if ending else "")
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_direct(raw, valid_range):
    m = re.search(r"\b(\d+)\b", raw.strip())
    if not m:
        return None
    val = int(m.group(1))
    lo, hi = valid_range
    return val if lo <= val <= hi else None


def parse_vs_cot(raw, valid_range, rng):
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
        return int(rng.choice(keys, p=probs))
    except Exception:
        return None


def make_filename(config_id, model, temperature, address_mode, sampling, cot, seed):
    model_alias = MODEL_ALIASES.get(model, model.split("/")[-1].replace("-", "").replace(".", ""))
    temp_str = _TEMP_LABELS.get(temperature, f"T{temperature}".replace(".", ""))
    cot_str = "cot" if cot else "nocot"
    return f"{config_id}_{model_alias}_{temp_str}_{address_mode}_{sampling}_{cot_str}_s{seed}.jsonl"


def load_done_ids(jsonl_path):
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


def append_result(jsonl_path, record):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def predict_one(client, semaphore, respondent_id, config_id, outcome,
                      layer_a_text, settings):
    spec = OUTCOME_SPECS[outcome]
    messages = build_prompt(layer_a_text, outcome,
                            settings["address_mode"], settings["sampling"])
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

    raw = ""
    prompt_tokens = 0
    completion_tokens = 0
    # Llama analizleri uzun olabiliyor  VS-CoT iin Gemini gibi 400 token
    max_out = 400 if settings["sampling"] == "vs_cot" else 10

    for attempt in range(2):
        for backoff_attempt in range(5):
            try:
                async with semaphore:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=settings["model"],
                            messages=messages,
                            temperature=settings["temperature"],
                            max_tokens=max_out,
                            seed=settings["seed"] if settings["temperature"] == 0 else None,
                            timeout=60.0,
                        ),
                        timeout=120.0,  # OpenRouter relay biraz yavaş
                    )
                raw = resp.choices[0].message.content or ""
                if resp.usage:
                    prompt_tokens = resp.usage.prompt_tokens
                    completion_tokens = resp.usage.completion_tokens
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str and backoff_attempt < 4:
                    print(f"    [{respondent_id}] 429 rate-limit, "
                          f"backing off {2**backoff_attempt}s...")
                    await asyncio.sleep(2 ** backoff_attempt)
                else:
                    print(f"    [{respondent_id}] API error: {err_str[:140]}")
                    return {**record_base, "predicted_value": None,
                            "raw_response": f"API_ERROR: {e}",
                            "prompt_tokens": 0, "completion_tokens": 0,
                            "parse_status": "api_error"}

        if settings["sampling"] == "vs_cot":
            val = parse_vs_cot(raw, spec["valid_range"], rng)
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


async def run_cell(client, semaphore, config_id, outcome, respondents,
                   settings, out_path):
    import numpy as np
    rng = np.random.default_rng(settings[\"seed\"])
    done_ids = load_done_ids(out_path)
    remaining = respondents[~respondents["respondent_id"].isin(done_ids)]

    if remaining.empty:
        print(f"  [{config_id}/{outcome}] already complete — skipping")
        return

    print(f"  [{config_id}/{outcome}] {len(remaining)} respondents remaining "
          f"({len(done_ids)} already done)")

    respondent_ids = remaining["respondent_id"].tolist()
    t0 = time.perf_counter()
    items = []
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
    print(f"  [{config_id}/{outcome}] complete — "
          f"{parse_failures}/{completed} parse failures ({pct_fail:.1f}%) {status}")


async def screening_mode(client, args):
    from config.ablations import CONFIGS

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

    print(f"=== OpenRouter calibration | model={settings['model']} | "
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
                settings["cot"], settings["seed"]
            )
            out_path = out_dir / filename
            await run_cell(client, semaphore, config.config_id, outcome,
                           outcome_df, settings, out_path)


def get_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        api_key = getpass.getpass("OpenRouter API key (sk-or-...): ").strip()
    if not api_key:
        print("ERROR: API key boş.", file=sys.stderr)
        sys.exit(1)
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            # OpenRouter expects app identification (optional but recommended)
            "HTTP-Referer": "https://github.com/local/digitaltwin",
            "X-Title": "TGSS Digital Twin Calibration",
        },
    )


async def amain(args):
    client = get_client()
    print(f"=== 10_run_calibration_openrouter.py | mode={args.mode} | "
          f"concurrency={args.concurrency} ===")
    if args.mode == "screening":
        await screening_mode(client, args)
    else:
        print(f"Mode '{args.mode}' not implemented.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["screening"], default="screening")
    parser.add_argument("--sampling", choices=["direct", "vs_cot"], default="direct")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="OpenRouter free tier için 5 önerilir.")
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--outcomes", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--address_mode", choices=["ben", "sen"], default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="OpenRouter model ID. "
                             "Default: meta-llama/llama-3.3-70b-instruct")
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
