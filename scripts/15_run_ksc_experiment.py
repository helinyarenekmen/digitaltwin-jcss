"""
15_run_ksc_experiment.py — Kurmanci/Zazaki eğitim dili sorusu (KSC).

Single-question, no-vignette experiment. All 2,615 C7 personas answer one
4-option nominal question about mother-tongue instruction policy, using
the calibrated womenwork protocol:

    C7 · Gemini 2.5 Flash Lite · VS-CoT · T=0.8 · ben · seed 0

Cevap seçenekleri (1–4):
    1 Eğitim iki dilli olmalı, hem Türkçe hem de ana dilde eğitim verilmeli
    2 Eğitim dili Türkçe olmalı, ana dil okulda ayrıca öğretilmeli
    3 Eğitim dili Türkçe olmalı, okulda ana dili öğretmeye de gerek yok
    4 Eğitim dili sadece ana dil (Kurmanci/Zazaki) olmalı

Outputs
-------
outputs/experiments/exp_ksc/
    exp_ksc_C7_gemini25flashlite_T08_ben_vs_cot_s0.jsonl
    csv/ksc_wide.csv
    csv/ksc_long.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv as csv_lib
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types


ROOT = Path(__file__).resolve().parents[1]
LAYER_A_DIR = Path(os.environ.get(
    "LAYER_A_DIR",
    str(Path.home() / "Library" / "Caches" / "digitaltwin_layer_a"),
))
EXPERIMENT_OUT = ROOT / "outputs" / "experiments"
EXP_ID = "exp_ksc"

MODEL       = "gemini-flash-lite-latest"
MODEL_ALIAS = "gemini25flashlite"
CONFIG      = "C7"
TEMP        = 0.8
SAMPLING    = "vs_cot"
ADDRESS     = "ben"

ITEM_ID = "ksc_medium_of_instruction"
QUESTION = (
    "Sizce ana dili Türkçe olmayanlar için, ana dili Kurmanci/Zazaki olanlar "
    "için okullarda eğitim dili nasıl olmalı?"
)
OPTIONS: dict[int, str] = {
    1: "Eğitim iki dilli olmalı, hem Türkçe hem de ana dilde eğitim verilmeli",
    2: "Eğitim dili Türkçe olmalı, ana dil okulda ayrıca öğretilmeli",
    3: "Eğitim dili Türkçe olmalı, okulda ana dili öğretmeye de gerek yok",
    4: "Eğitim dili sadece ana dil (Kurmanci/Zazaki) olmalı",
}
VALID_RANGE = (min(OPTIONS), max(OPTIONS))


SYSTEM_BEN = (
    "Aşağıda Türkiye'de yaşayan birinin kişisel profili yer almaktadır. "
    "Bu kişinin yerine geçerek bir ankete cevap vereceksin. "
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


def build_prompt(layer_a_text: str) -> tuple[str, str]:
    system = SYSTEM_BEN + VS_COT_SUFFIX
    opts = "\n".join(f"{k} = {v}" for k, v in OPTIONS.items())
    user = (
        f"{layer_a_text}\n\n"
        f"---\n\n"
        f"Soru: {QUESTION}\n\n"
        f"Seçenekler:\n{opts}"
    )
    return system, user


def per_respondent_seed(rid: str, base_seed: int) -> int:
    payload = f"{base_seed}|{rid}|{ITEM_ID}".encode("utf-8")
    return int.from_bytes(hashlib.md5(payload).digest()[:8], "big") & 0x7FFFFFFF


def parse_vs_cot(raw: str, seed: int) -> tuple[int | None, dict[int, float] | None]:
    import numpy as np
    m = re.search(r"DAĞILIM:\s*(\{[^}]+\})", raw or "", re.DOTALL)
    if not m:
        return None, None
    try:
        dist_raw = json.loads(m.group(1))
        options = {int(k): float(v) for k, v in dist_raw.items()}
        lo, hi = VALID_RANGE
        options = {k: v for k, v in options.items() if lo <= k <= hi}
        if not options:
            return None, None
        total = sum(options.values())
        norm = {k: v / total for k, v in options.items()}
        keys = list(options.keys())
        probs = [norm[k] for k in keys]
        rng = np.random.default_rng(seed)
        return int(rng.choice(keys, p=probs)), norm
    except Exception:
        return None, None


def cell_filename() -> str:
    return f"{EXP_ID}_{CONFIG}_{MODEL_ALIAS}_T08_{ADDRESS}_{SAMPLING}_s0.jsonl"


async def _gemini_call(client, *, system: str, user: str) -> tuple[str, int, int]:
    cfg = genai_types.GenerateContentConfig(
        system_instruction=system, temperature=TEMP, max_output_tokens=800,
    )
    resp = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=MODEL, contents=user, config=cfg,
        ),
        timeout=120.0,
    )
    raw = (resp.text or "").strip()
    pt = resp.usage_metadata.prompt_token_count or 0 if resp.usage_metadata else 0
    ct = resp.usage_metadata.candidates_token_count or 0 if resp.usage_metadata else 0
    return raw, pt, ct


async def _call_one(client, *, rid: str, layer_a: str, base_seed: int,
                    max_retries: int = 5) -> dict:
    system, user = build_prompt(layer_a)
    seed = per_respondent_seed(rid, base_seed)
    base = dict(
        experiment_id=EXP_ID, respondent_id=rid, item_id=ITEM_ID,
        persona_config=CONFIG, model=MODEL, temperature=TEMP,
        sampling=SAMPLING, address_mode=ADDRESS, base_seed=base_seed,
        draw_seed=seed,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    backoff = 1
    for attempt in range(max_retries):
        try:
            raw, pt, ct = await _gemini_call(client, system=system, user=user)
            break
        except Exception as e:
            err = str(e)
            if ("429" in err or "rate" in err.lower() or "RESOURCE_EXHAUSTED" in err) \
                    and attempt < max_retries - 1:
                await asyncio.sleep(2 ** backoff); backoff += 1
                continue
            return {**base, "predicted_value": None, "dist": None,
                    "raw_response": f"API_ERROR: {err}",
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "parse_status": "api_error"}
    val, dist = parse_vs_cot(raw, seed)
    return {**base, "predicted_value": val, "dist": dist,
            "raw_response": raw, "prompt_tokens": pt, "completion_tokens": ct,
            "parse_status": "ok" if val is not None else "parse_failure"}


def list_respondents() -> list[str]:
    d = LAYER_A_DIR / CONFIG
    return sorted(p.stem for p in d.glob("TGSS_*.md"))


def preload(rids: list[str]) -> dict[str, str]:
    return {rid: (LAYER_A_DIR / CONFIG / f"{rid}.md").read_text(encoding="utf-8")
            for rid in rids}


async def run(client, rids: list[str], out_dir: Path, base_seed: int,
              concurrency: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / cell_filename()
    done: set[str] = set()
    if fp.exists():
        for line in fp.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("parse_status") == "ok":
                    done.add(r["respondent_id"])
            except Exception:
                continue
        print(f"  resume: {len(done)} respondents already ok")

    personas = preload(rids)
    fh = fp.open("a", encoding="utf-8")
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    async def worker(rid: str) -> None:
        if rid in done:
            return
        async with sem:
            rec = await _call_one(client, rid=rid, layer_a=personas[rid],
                                  base_seed=base_seed)
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            async with write_lock:
                fh.write(line); fh.flush()

    tasks = [asyncio.create_task(worker(rid)) for rid in rids]
    total = len(tasks)
    print(f"  scheduling {total} API calls  concurrency={concurrency}")
    completed = 0
    t0 = time.time()
    log_every = max(50, total // 200)
    try:
        for fut in asyncio.as_completed(tasks):
            await fut
            completed += 1
            if completed % log_every == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total - completed) / rate if rate > 0 else 0
                print(f"    progress: {completed}/{total} "
                      f"({completed/total*100:.1f}%)  rate={rate:.1f}/s  "
                      f"ETA={int(remaining)//60}m{int(remaining)%60:02d}s")
    finally:
        fh.close()


def export_csvs(out_dir: Path) -> None:
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for fp in sorted(out_dir.glob("*.jsonl")):
        for line in fp.open(encoding="utf-8"):
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    if not records:
        print("  no records"); return

    long_path = csv_dir / "ksc_long.csv"
    cols = ["experiment_id", "respondent_id", "item_id", "persona_config",
            "model", "temperature", "sampling", "address_mode",
            "base_seed", "draw_seed", "predicted_value", "parse_status"]
    with long_path.open("w", newline="", encoding="utf-8") as f:
        w = csv_lib.writer(f); w.writerow(cols)
        for r in records:
            w.writerow([r.get(c) for c in cols])
    print(f"  ✓ {long_path.relative_to(ROOT)}  ({len(records)} rows)")

    wide_path = csv_dir / "ksc_wide.csv"
    with wide_path.open("w", newline="", encoding="utf-8") as f:
        w = csv_lib.writer(f)
        w.writerow(["respondent_id", f"{ITEM_ID}_pred", "parse_status"])
        for r in records:
            w.writerow([r["respondent_id"], r.get("predicted_value"),
                        r.get("parse_status")])
    print(f"  ✓ {wide_path.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--random-sample", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--export-csvs", action="store_true")
    args = p.parse_args()

    rids = list_respondents()
    if args.random_sample:
        rng = random.Random(args.sample_seed)
        rids = sorted(rng.sample(rids, args.random_sample))

    out_dir = EXPERIMENT_OUT / EXP_ID
    print("=== 15_run_ksc_experiment.py ===")
    print(f"  experiment_id  : {EXP_ID}")
    print(f"  personas       : n={len(rids)}  (config={CONFIG})")
    print(f"  protocol       : {CONFIG} · {MODEL} · {SAMPLING} · T={TEMP} · {ADDRESS}")
    print(f"  base_seed      : {args.seed}")
    print(f"  concurrency    : {args.concurrency}")
    print(f"  output dir     : {out_dir.relative_to(ROOT)}")
    print(f"  total API calls: {len(rids)}")
    # rough cost estimate
    in_rate, out_rate = 0.10/1_000_000, 0.40/1_000_000
    cost = len(rids) * (1000 * in_rate + 400 * out_rate)
    print(f"  est cost       : ~${cost:.2f}")

    if args.export_csvs:
        export_csvs(out_dir); return
    if args.dry_run:
        print("\n--dry-run: not calling the API."); return

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        import getpass
        key = getpass.getpass("Google / Gemini API key: ").strip()
    if not key:
        raise SystemExit("ERROR: GEMINI_API_KEY missing.")
    client = genai.Client(api_key=key)

    asyncio.run(run(client, rids, out_dir, args.seed, args.concurrency))
    print("\n✓ Done. Exporting CSVs …")
    export_csvs(out_dir)


if __name__ == "__main__":
    main()
