"""
04_run_calibration.py — Phase 2 calibration inference runner.

Modes:
  screening   — Step 2A: all 11 configs × 3 outcomes, fixed default setting
  robustness  — Step 2C: shortlist configs × inference grid (requires shortlist.json
                and configs/inference_grid.json; run after Step 2B)

Usage:
  # Step 2A — screening (dry run: first 5 respondents)
  python scripts/04_run_calibration.py --mode screening --dry_run

  # Step 2A — full
  python scripts/04_run_calibration.py --mode screening

  # Step 2C — robustness (after 2B shortlist is ready)
  python scripts/04_run_calibration.py --mode robustness

Resume: each (config × outcome) cell appends to one JSONL. Already-scored
respondents are skipped. Crash-safe.

See CALIBRATION_SPEC.md §2, §3, §4, §6.
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

# layer_a klasörü iCloud-senkronize Desktop'tan ~/Library/Caches'e taşındı
# (28K küçük dosya iCloud'da sürekli offload edilip okuma 1500ms'ye çıkıyordu).
# outputs/layer_a symlink'i iCloud sürekli "layer_a 2" diye rename ediyor,
# bu yüzden script direkt gerçek yolu kullanıyor.
LAYER_A_DIR   = Path(str(_PERSONA))
# outputs/calibration de iCloud-dışına taşındı: 6MB+'lik JSONL'ler
# sürekli iCloud upload'a uğrayıp load_done_ids'i kilitliyordu.
CALIB_DIR     = Path(str(_CACHE))
SCREENING_DIR = CALIB_DIR / "screening"
ROBUST_DIR    = CALIB_DIR / "robustness"
SHORTLIST_PATH = CALIB_DIR / "shortlist.json"
GRID_PATH     = ROOT / "configs" / "inference_grid.json"
CSV_PATH      = ROOT / "data" / "tgss2024_clean.csv"

# ---------------------------------------------------------------------------
# Outcome specs — verbatim TGSS questions and valid response ranges
# See CALIBRATION_SPEC.md §3
# ---------------------------------------------------------------------------

OUTCOME_SPECS: dict[str, dict] = {
    "pacdemons": {
        "question": (
            "Son 12 ay içinde Türkiye'de işlerin düzelmesini sağlamak veya "
            "kötüye gitmesini önlemek için sokak eylemlerine katıldınız mı?"
        ),
        "options": {1: "Evet", 2: "Hayır"},
        "valid_range": (1, 2),
        "minority_code": 1,  # 1=Evet is the rare class (~5%)
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
    # ---- Robustness / leakage-guarded probes (config_robustness) ----
    "pacparty": {
        "question": (
            "Son 12 ay içinde Türkiye'de işlerin düzelmesini sağlamak veya "
            "kötüye gitmesini önlemek için bir partinin siyasi faaliyetlerine "
            "katıldınız mı?"
        ),
        "options": {1: "Evet", 2: "Hayır"},
        "valid_range": (1, 2),
        "minority_code": 1,
    },
    "pacboycott": {
        "question": (
            "Son 12 ay içinde Türkiye'de işlerin düzelmesini sağlamak veya "
            "kötüye gitmesini önlemek için belirli ürünleri boykot ettiniz mi?"
        ),
        "options": {1: "Evet", 2: "Hayır"},
        "valid_range": (1, 2),
        "minority_code": 1,
    },
    "paconline": {
        "question": (
            "Son 12 ay içinde Türkiye'de işlerin düzelmesini sağlamak veya "
            "kötüye gitmesini önlemek için internette (örneğin; bloglarda, "
            "e-posta yoluyla veya Facebook, Twitter gibi sosyal medya "
            "platformlarında) politika hakkında herhangi bir şey paylaştınız mı?"
        ),
        "options": {1: "Evet", 2: "Hayır"},
        "valid_range": (1, 2),
        "minority_code": 1,
    },
    "polint": {
        "question": "Siyaset konuları ile ne kadar ilgilisiniz?",
        "options": {
            1: "Hiç ilgili değilim",
            2: "Biraz ilgiliyim",
            3: "Oldukça ilgiliyim",
            4: "Çok ilgiliyim",
        },
        "valid_range": (1, 4),
        "minority_code": None,
    },
    "thimmig": {
        "question": "Sizce, göçmenler Türkiye için ne derecede tehdit oluşturmaktadır?",
        "options": {
            0: "0 (Hiç tehdit değil)",
            1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
            6: "6", 7: "7", 8: "8", 9: "9",
            10: "10 (Çok ciddi tehdit)",
        },
        "valid_range": (0, 10),
        "minority_code": None,
    },
    "polminor": {
        "question": (
            "Aşağıdaki ifadeye katılıp katılmama düzeyiniz nedir?\n"
            "Türkiye'de azınlık gruplarının hakları korunmaktadır."
        ),
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
    "polfrlim": {
        "question": (
            "Aşağıdaki ifadeye katılıp katılmama düzeyiniz nedir?\n"
            "Terörle mücadele amacıyla kişisel özgürlüklerin sınırlandırılması "
            "kabul edilebilir."
        ),
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
    "satdem": {
        "question": "Genel olarak, Türkiye'de demokrasinin işleyişinden ne kadar memnunsunuz?",
        "options": {
            1: "Hiç memnun değilim",
            2: "Memnun değilim",
            3: "Ne memnunum ne değilim",
            4: "Memnunum",
            5: "Tamamen memnunum",
        },
        "valid_range": (1, 5),
        "minority_code": None,
    },
    "famroles": {
        "question": (
            "Aşağıdaki ifadeye katılıp katılmama düzeyiniz nedir?\n"
            "Bir erkeğin görevi para kazanmaktır; bir kadının görevi ise eve "
            "ve aileye bakmaktır."
        ),
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
    "paccontact": {
        "question": (
            "Son 12 ay içinde Türkiye'de işlerin düzelmesini sağlamak veya "
            "kötüye gitmesini önlemek için bir politikacıyla, hükümetle veya "
            "yerel yönetim yetkilisiyle iletişime geçtiniz mi?"
        ),
        "options": {1: "Evet", 2: "Hayır"},
        "valid_range": (1, 2),
        "minority_code": 1,
    },
    "paccompl": {
        "question": (
            "Son 12 ay içinde Türkiye'de işlerin düzelmesini sağlamak veya "
            "kötüye gitmesini önlemek için çözüm merkezlerine talep veya "
            "şikayette bulundunuz mu?"
        ),
        "options": {1: "Evet", 2: "Hayır"},
        "valid_range": (1, 2),
        "minority_code": 1,
    },
    "paccimer": {
        "question": (
            "Son 12 ay içinde Türkiye'de işlerin düzelmesini sağlamak veya "
            "kötüye gitmesini önlemek için CİMER'e şikayette bulundunuz mu?"
        ),
        "options": {1: "Evet", 2: "Hayır"},
        "valid_range": (1, 2),
        "minority_code": 1,
    },
    "pacvolunteer": {
        "question": (
            "Son 12 ay içinde Türkiye'de işlerin düzelmesini sağlamak veya "
            "kötüye gitmesini önlemek için kar amacı gütmeyen veya hayır "
            "kurumu için gönüllü olarak çalıştınız mı?"
        ),
        "options": {1: "Evet", 2: "Hayır"},
        "valid_range": (1, 2),
        "minority_code": 1,
    },
    "paccult": {
        "question": (
            "Son 12 ay içinde Türkiye'de işlerin düzelmesini sağlamak veya "
            "kötüye gitmesini önlemek için sosyal ve kültürel aktivitelerde "
            "gönüllü oldunuz mu?"
        ),
        "options": {1: "Evet", 2: "Hayır"},
        "valid_range": (1, 2),
        "minority_code": 1,
    },
    "paccharity": {
        "question": (
            "Son 12 ay içinde Türkiye'de işlerin düzelmesini sağlamak veya "
            "kötüye gitmesini önlemek için hayır kurumlarına veya yardım "
            "faaliyetlerine katıldınız mı?"
        ),
        "options": {1: "Evet", 2: "Hayır"},
        "valid_range": (1, 2),
        "minority_code": 1,
    },
}

# Step 2A fixed default — locked before any prediction (CALIBRATION_SPEC.md §2)
SCREENING_DEFAULT = dict(
    model="gpt-4o-mini",
    temperature=0.0,
    address_mode="ben",
    sampling="direct",
    cot=False,
    seed=0,
)

MODEL_ALIASES = {
    "gpt-4o-mini": "gpt4omini",
    "gpt-5-mini": "gpt5mini",
    "gpt-5.4-mini": "gpt54mini",
    "claude-haiku": "claudehaiku",
}

# ---------------------------------------------------------------------------
# Prompt construction (§3)
# ---------------------------------------------------------------------------

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

# Explicit Chain-of-Thought addition (separate factor from VS-CoT).
# Asks model to reason step-by-step before answering.
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

# Political context: Türkiye ideolojik yapısı hakkında domain knowledge.
# Phase 15 testi: bu context'in calibration performansına etkisini ölçer.
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
) -> list[dict]:
    spec = OUTCOME_SPECS[outcome]
    options_str = "\n".join(f"{k} = {v}" for k, v in spec["options"].items())
    system = SYSTEM_BEN if address_mode == "ben" else SYSTEM_SEN

    # Political context is PREPENDED to the system prompt (background knowledge).
    if political_context:
        system = POLITICAL_CONTEXT_TR + "\n\n" + system

    if sampling == "vs_cot":
        system = system + VS_COT_SUFFIX
        if cot:
            system = system + EXPLICIT_COT_SUFFIX_VSCOT
    elif cot:
        # Direct + explicit CoT: ask for reasoning then CEVAP: X format
        system = system + EXPLICIT_COT_SUFFIX_DIRECT

    # Direct (no CoT) → "Cevap (sadece sayı):" tail.
    # Direct + CoT → tail removed (format is enforced by CEVAP: X marker).
    # VS-CoT → ANALİZ+DAĞILIM format takes over, no tail.
    if sampling == "vs_cot" or cot:
        ending = ""
    else:
        ending = "Cevap (sadece sayı):"
    user = (
        f"{layer_a_text}\n\n"
        f"---\n\n"
        f"Soru: {spec['question']}\n\n"
        f"Seçenekler:\n{options_str}"
        + (f"\n\n{ending}" if ending else "")
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Response parsing (§3)
# ---------------------------------------------------------------------------

def parse_direct(raw: str, valid_range: tuple[int, int]) -> int | None:
    """Extract integer from response.
    Priority: "CEVAP: X" marker (from explicit CoT), else first integer.
    """
    # CoT format: look for "CEVAP: X" first
    m_cevap = re.search(r"CEVAP\s*:\s*(\d+)", raw, re.IGNORECASE)
    if m_cevap:
        val = int(m_cevap.group(1))
        lo, hi = valid_range
        return val if lo <= val <= hi else None
    # Fallback: first integer in response
    m = re.search(r"\b(\d+)\b", raw.strip())
    if not m:
        return None
    val = int(m.group(1))
    lo, hi = valid_range
    return val if lo <= val <= hi else None


def parse_vs_cot(raw: str, valid_range: tuple[int, int], rng) -> int | None:
    """Extract the verbalized distribution and draw one categorical sample.

    Fixed protocol (paper Section 3.3): a single client-side RNG is created
    per run at the base seed and advanced across respondents. `rng` must be a
    numpy.random.Generator instance shared by the caller.
    """
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


# ---------------------------------------------------------------------------
# File naming and JSONL I/O
# ---------------------------------------------------------------------------

_TEMP_LABELS = {0.0: "T0", 0.3: "T03", 0.4: "T04", 0.7: "T07", 0.8: "T08", 1.0: "T1"}


def make_filename(config_id: str, model: str, temperature: float,
                  address_mode: str, sampling: str, cot: bool, seed: int,
                  political_context: bool = False) -> str:
    model_alias = MODEL_ALIASES.get(model, model.replace("-", ""))
    temp_str = _TEMP_LABELS.get(temperature, f"T{temperature}".replace(".", ""))
    cot_str = "cot" if cot else "nocot"
    # Political context inserts "_pc" token before seed; default empty (no change to legacy names).
    pc_token = "_pc" if political_context else ""
    return f"{config_id}_{model_alias}_{temp_str}_{address_mode}_{sampling}_{cot_str}{pc_token}_s{seed}.jsonl"


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


# ---------------------------------------------------------------------------
# Single prediction (with retry logic)
# ---------------------------------------------------------------------------

async def predict_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    respondent_id: str,
    config_id: str,
    outcome: str,
    layer_a_text: str,
    settings: dict,
    rng=None,
) -> dict:
    """Single-respondent prediction. `rng` is required for verbalized sampling
    and must be a numpy.random.Generator shared across the whole run (created
    once at the base seed; see run_cell / amain)."""
    spec = OUTCOME_SPECS[outcome]
    messages = build_prompt(
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

    for attempt in range(2):  # attempt 0 = first try, attempt 1 = single retry on parse fail
        for backoff_attempt in range(5):
            try:
                # GPT-5 ailesi 'max_completion_tokens' istiyor; eski modeller
                # 'max_tokens' ile çalışır. Modele göre doğru parametreyi seç.
                # CoT eklenince reasoning için daha çok token gerekir.
                # VS-CoT 150 -> 250 değiştirildi (2026-06-27): zengin persona'larda
                # (C11 gibi) ANALİZ uzun yazılıyor, DAĞILIM JSON yarıda kesiliyordu
                # → parse_failure. 250 token uzun analiz + tam JSON için yeterli.
                if settings["sampling"] == "vs_cot":
                    max_out = 400 if settings.get("cot", False) else 250
                else:
                    max_out = 200 if settings.get("cot", False) else 10
                tokens_key = ("max_completion_tokens"
                              if settings["model"].startswith("gpt-5")
                              else "max_tokens")
                kwargs = dict(
                    model=settings["model"],
                    messages=messages,
                    temperature=settings["temperature"],
                    seed=settings["seed"] if settings["temperature"] == 0 else None,
                    timeout=60.0,
                )
                kwargs[tokens_key] = max_out
                async with semaphore:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(**kwargs),
                        timeout=90.0,  # asyncio-level hard cap
                    )
                raw = resp.choices[0].message.content or ""
                prompt_tokens = resp.usage.prompt_tokens
                completion_tokens = resp.usage.completion_tokens
                break  # success
            except Exception as e:
                err_str = str(e)
                if "429" in err_str and backoff_attempt < 4:
                    print(f"    [{respondent_id}] 429 rate-limit, backing off {2**backoff_attempt}s...")
                    await asyncio.sleep(2 ** backoff_attempt)
                else:
                    print(f"    [{respondent_id}] API error (attempt {attempt},{backoff_attempt}): {err_str[:120]}")
                    return {**record_base, "predicted_value": None,
                            "raw_response": f"API_ERROR: {e}",
                            "prompt_tokens": 0, "completion_tokens": 0,
                            "parse_status": "api_error"}
        else:
            continue

        # Parse
        if settings["sampling"] == "vs_cot":
            val = parse_vs_cot(raw, spec["valid_range"], rng)
        else:
            val = parse_direct(raw, spec["valid_range"])

        if val is not None:
            return {**record_base, "predicted_value": val, "raw_response": raw,
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "parse_status": "ok"}

        # First attempt failed to parse — retry once (same settings, per §3)
        if attempt == 0:
            continue

    # Both attempts failed
    return {**record_base, "predicted_value": None, "raw_response": raw,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "parse_status": "parse_failure"}


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

async def run_cell(
    client: AsyncOpenAI,
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

    # Iterate over a plain ID list — avoids ~0.4ms/row overhead of
    # iterrows() on a 664-column DataFrame. File reads are sequential
    # (macOS throttles parallel small-file reads under contention).
    respondent_ids: list[str] = remaining["respondent_id"].tolist()
    t0 = time.perf_counter()
    items: list[tuple[str, str]] = []
    for rid in respondent_ids:
        p = LAYER_A_DIR / config_id / f"{rid}.md"
        if not p.exists():
            print(f"  WARNING: missing {p.name}")
            continue
        items.append((rid, p.read_text(encoding="utf-8")))
    elapsed = time.perf_counter() - t0
    print(f"  [{config_id}/{outcome}] {len(items)} files loaded in {elapsed:.1f}s, starting API calls...")

    # One numpy RNG per run, initialised with the base seed. The same instance
    # is passed into every predict_one call; parse_vs_cot advances it across
    # respondents. This fixes the earlier per-respondent reinit bug that
    # collapsed VS-CoT sampling into fixed-quantile inverse-CDF sampling.
    import numpy as np
    rng = np.random.default_rng(settings["seed"])

    tasks = [
        predict_one(client, semaphore, rid, config_id, outcome, text, settings, rng)
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


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

async def screening_mode(client: AsyncOpenAI, args) -> None:
    from config.ablations import CONFIGS

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df["respondent_id"] = df["id"].apply(lambda x: f"TGSS_{int(x):04d}")

    # Only respondents with a valid ground truth for this outcome (NaN → skip)
    configs = CONFIGS
    all_outcomes = list(OUTCOME_SPECS.keys())
    if args.outcomes:
        wanted = {o.strip() for o in args.outcomes.split(",")}
        outcomes = [o for o in all_outcomes if o in wanted]
        unknown = wanted - set(all_outcomes)
        if unknown:
            print(f"WARNING: bilinmeyen outcome: {sorted(unknown)}")
        print(f"Outcomes filtered: {outcomes}")
    else:
        outcomes = all_outcomes

    settings = SCREENING_DEFAULT.copy()
    settings["sampling"] = args.sampling  # "direct" or "vs_cot"
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

    # Output klasörleri: canonical Step 2A (direct + T=0) → screening/.
    # Diğer her kombinasyon ayrı klasöre yazılır, mevcut sonuçlar dokunulmaz.
    temp_str = _TEMP_LABELS.get(settings["temperature"],
                                f"T{settings['temperature']}".replace(".", ""))
    if args.sampling == "direct" and settings["temperature"] == 0.0:
        base_dir = SCREENING_DIR  # canonical Step 2A
    elif args.sampling == "direct":
        base_dir = SCREENING_DIR.parent / f"screening_{temp_str}"
    elif settings["temperature"] == 0.0:
        base_dir = SCREENING_DIR.parent / "screening_vs_cot"
    else:
        base_dir = SCREENING_DIR.parent / f"screening_vs_cot_{temp_str}"

    if args.configs:
        wanted = {c.strip() for c in args.configs.split(",")}
        configs = [c for c in CONFIGS if c.config_id in wanted]
        missing = wanted - {c.config_id for c in configs}
        if missing:
            print(f"WARNING: unknown configs ignored: {sorted(missing)}")
        print(f"Filtered to {len(configs)} configs: {[c.config_id for c in configs]}")

    if args.dry_run:
        configs = [c for c in CONFIGS if c.config_id in ("C0", "C1")]
        df = df.head(5)
        print(f"DRY RUN ({args.sampling}): 5 respondents × C0,C1 × 3 outcomes")

    print(f"=== sampling={args.sampling} | output_dir={base_dir} ===")
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
                political_context=settings.get("political_context", False)
            )
            out_path = out_dir / filename
            await run_cell(
                client, semaphore, config.config_id, outcome,
                outcome_df, settings, out_path
            )


async def robustness_mode(client: AsyncOpenAI, args) -> None:
    if not SHORTLIST_PATH.exists():
        print(f"ERROR: {SHORTLIST_PATH} not found — run Step 2B first.")
        sys.exit(1)
    if not GRID_PATH.exists():
        print(f"ERROR: {GRID_PATH} not found — create after budget decision.")
        sys.exit(1)

    with open(SHORTLIST_PATH) as f:
        shortlist = json.load(f)["shortlist"]
    with open(GRID_PATH) as f:
        grid = json.load(f)

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df["respondent_id"] = df["id"].apply(lambda x: f"TGSS_{int(x):04d}")

    semaphore = asyncio.Semaphore(args.concurrency)

    for cell in grid["cells"]:
        config_id = cell["config_id"]
        if config_id not in shortlist:
            continue
        outcome = cell["outcome"]
        settings = {k: cell[k] for k in
                    ["model", "temperature", "address_mode", "sampling", "cot", "seed"]}

        lo, hi = OUTCOME_SPECS[outcome]["valid_range"]
        outcome_df = df[df[outcome].between(lo, hi)].copy()
        out_dir = ROBUST_DIR / outcome
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = make_filename(
            config_id, settings["model"], settings["temperature"],
            settings["address_mode"], settings["sampling"],
            settings["cot"], settings["seed"],
            political_context=settings.get("political_context", False),
        )
        out_path = out_dir / filename
        await run_cell(client, semaphore, config_id, outcome,
                       outcome_df, settings, out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

OPENAI_MODELS = {"gpt-4o-mini", "gpt-4o"}
ANTHROPIC_MODELS = {"claude-haiku", "claude-haiku-4-5-20251001"}


def get_client(model: str) -> AsyncOpenAI:
    if model in ANTHROPIC_MODELS:
        # TODO Step 2C: replace with anthropic.AsyncAnthropic client.
        # Anthropic SDK uses a different call signature (client.messages.create,
        # not client.chat.completions.create). predict_one() will need a model
        # dispatch branch before Step 2C runs with Claude-Haiku.
        # Tracking issue: CALIBRATION_SPEC.md §15 "Claude-Haiku API access".
        raise NotImplementedError(
            f"Model '{model}' requires the Anthropic SDK. "
            "Wire up anthropic.AsyncAnthropic before running Step 2C. "
            "See CALIBRATION_SPEC.md §15."
        )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("OpenAI API key: ").strip()
    if not api_key:
        print("ERROR: API key boş.", file=sys.stderr)
        sys.exit(1)
    return AsyncOpenAI(api_key=api_key)


async def amain(args) -> None:
    # Screening always uses gpt-4o-mini; robustness may include claude-haiku
    if args.model is not None:
        model = args.model
    else:
        model = SCREENING_DEFAULT["model"] if args.mode == "screening" else "gpt-4o-mini"
    client = get_client(model)

    print(f"=== 04_run_calibration.py | mode={args.mode} | "
          f"concurrency={args.concurrency} ===")

    if args.mode == "screening":
        await screening_mode(client, args)
    elif args.mode == "robustness":
        await robustness_mode(client, args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["screening", "robustness"],
                        default="screening")
    parser.add_argument("--sampling", choices=["direct", "vs_cot"],
                        default="direct",
                        help="screening only: 'direct' (Step 2A) or 'vs_cot'. "
                             "vs_cot writes to outputs/calibration/screening_vs_cot/")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config IDs to run "
                             "(e.g. 'C1,C3,C4,C7'). Default: all configs.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override default temperature (Step 2A=0.0). "
                             "Filename encodes temperature; new runs do not clobber prior.")
    parser.add_argument("--outcomes", type=str, default=None,
                        help="Comma-separated outcomes (default: tüm 3'ü). "
                             "Örn: 'pacdemons' veya 'pacdemons,neilang'.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override default seed (Step 2A=0). "
                             "T>0 cell'lerinde stokastik kararlılık için "
                             "farklı seed'ler kullanılır (s0,s1,s2...).")
    parser.add_argument("--address_mode", choices=["ben", "sen"], default=None,
                        help="Override default address mode (Step 2A=ben). "
                             "Filename'de _ben_/_sen_ olarak görünür, "
                             "sen-dili sonuçları ben-dili olanları clobber etmez.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override default model (Step 2A=gpt-4o-mini). "
                             "Örn: gpt-5.4-mini, gpt-5-mini.")
    parser.add_argument("--cot", action="store_true",
                        help="Add explicit chain-of-thought prompting. "
                             "Direct + CoT: model reasons then outputs 'CEVAP: X'. "
                             "VS-CoT + CoT: emphasizes step-by-step ANALİZ. "
                             "Filename'de _cot_ olarak görünür.")
    parser.add_argument("--political_context", action="store_true",
                        help="Prepend Turkish political-ideology context paragraph to "
                             "system prompt (Phase 15 test). Filename'de _pc_ olarak görünür.")
    parser.add_argument("--concurrency", type=int, default=15,
                        help="Max in-flight API calls")
    parser.add_argument("--dry_run", action="store_true",
                        help="5 respondents, C0+C1 only")
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
