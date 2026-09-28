"""
run_calibration_natural.py — Best-setup calibration with natural-language personas.

Runs the two final-pick calibration cells against the natural-language
persona rewrites produced by `natural_rewrite_personas.py`. Structured-
persona results in the main calibration cache are not touched.

Cells covered (best-setup, C7):
  pacdemons  — C7 · gpt-4o-mini · Direct · T=0.8 · ben · seed 0
  womenwork  — C7 · Gemini 2.5 Flash Lite · VS-CoT · T=0.8 · ben · seed 0

Paths
-----
Personas (natural, produced by natural_rewrite_personas.py):
  ~/Library/Caches/digitaltwin_layer_a_natural/C7/TGSS_*.md

Outputs (kept separate from the structured cache):
  ~/Library/Caches/digitaltwin_calibration_natural/
    screening_T08/pacdemons/C7_gpt4omini_T08_ben_direct_nocot_s0.jsonl
    screening_vs_cot_T08/womenwork/C7_geminiflashlite_T08_ben_vs_cot_nocot_s0.jsonl

Usage
-----
    export OPENAI_API_KEY=...  GEMINI_API_KEY=...
    python scripts/run_calibration_natural.py                  # both cells
    python scripts/run_calibration_natural.py --only pacdemons
    python scripts/run_calibration_natural.py --only womenwork --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore
    genai_types = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
NAT_LAYER_A_DIR = Path.home() / "Library" / "Caches" / "digitaltwin_layer_a_natural"
NAT_CALIB_DIR   = Path.home() / "Library" / "Caches" / "digitaltwin_calibration_natural"
BACKUP_DIR      = ROOT / "outputs" / "calibration_backup" / "natural_rewrite" / "calibration"


CELLS: dict[str, dict] = {
    "pacdemons": dict(
        outcome="pacdemons",
        subdir="screening_T08",
        filename="C7_gpt4omini_T08_ben_direct_nocot_s0.jsonl",
        model="gpt-4o-mini",
        temperature=0.8,
        sampling="direct",
        question=(
            "Son 12 ay içinde Türkiye'de işlerin düzelmesini sağlamak veya "
            "kötüye gitmesini önlemek için sokak eylemlerine katıldınız mı?"
        ),
        options={1: "Evet", 2: "Hayır"},
        valid_range=(1, 2),
    ),
    "womenwork": dict(
        outcome="womenwork",
        subdir="screening_vs_cot_T08",
        filename="C7_geminiflashlite_T08_ben_vs_cot_nocot_s0.jsonl",
        model="gemini-2.5-flash-lite",
        temperature=0.8,
        sampling="vs_cot",
        question="Kadın tam zamanlı çalıştığında aile hayatı olumsuz etkilenir.",
        options={
            1: "Hiç katılmıyorum",
            2: "Katılmıyorum",
            3: "Ne katılıyorum ne katılmıyorum",
            4: "Katılıyorum",
            5: "Tamamen katılıyorum",
        },
        valid_range=(1, 5),
    ),
    "womenwork_gpt54mini": dict(
        outcome="womenwork",
        subdir="screening_vs_cot_T08",
        filename="C7_gpt54mini_T08_ben_vs_cot_nocot_s0.jsonl",
        model="gpt-5.4-mini",
        temperature=0.8,
        sampling="vs_cot",
        question="Kadın tam zamanlı çalıştığında aile hayatı olumsuz etkilenir.",
        options={
            1: "Hiç katılmıyorum",
            2: "Katılmıyorum",
            3: "Ne katılıyorum ne katılmıyorum",
            4: "Katılıyorum",
            5: "Tamamen katılıyorum",
        },
        valid_range=(1, 5),
    ),
}

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


def build_prompt(layer_a_text: str, cell: dict) -> tuple[str, str]:
    system = SYSTEM_BEN + (VS_COT_SUFFIX if cell["sampling"] == "vs_cot" else "")
    options_str = "\n".join(f"{k} = {v}" for k, v in cell["options"].items())
    user = (
        f"{layer_a_text}\n\n"
        f"---\n\n"
        f"Soru: {cell['question']}\n\n"
        f"Seçenekler:\n{options_str}"
    )
    if cell["sampling"] != "vs_cot":
        user += "\n\nCevap (sadece sayı):"
    return system, user


def per_respondent_seed(rid: str, outcome: str, base_seed: int) -> int:
    payload = f"{base_seed}|{rid}|{outcome}".encode("utf-8")
    return int.from_bytes(hashlib.md5(payload).digest()[:8], "big") & 0x7FFFFFFF


def parse_direct(raw: str, valid_range: tuple[int, int]) -> int | None:
    lo, hi = valid_range
    m = re.search(r"(\d+)", raw or "")
    if not m: return None
    v = int(m.group(1))
    return v if lo <= v <= hi else None


def parse_vs_cot(raw: str, valid_range: tuple[int, int],
                 seed: int) -> tuple[int | None, dict[int, float] | None]:
    import numpy as np
    m = re.search(r"DAĞILIM:\s*(\{[^}]+\})", raw or "", re.DOTALL)
    if not m: return None, None
    try:
        dist_raw = json.loads(m.group(1))
        options = {int(k): float(v) for k, v in dist_raw.items()}
        lo, hi = valid_range
        options = {k: v for k, v in options.items() if lo <= k <= hi}
        if not options: return None, None
        total = sum(options.values())
        norm = {k: v / total for k, v in options.items()}
        rng = np.random.default_rng(seed)
        return int(rng.choice(list(norm), p=list(norm.values()))), norm
    except Exception:
        return None, None


async def _openai_call(client, *, model, system, user, temperature, max_out, seed):
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=temperature,
        seed=seed if temperature == 0 else None,
    )
    # GPT-5 family uses 'max_completion_tokens'; older models use 'max_tokens'.
    tokens_key = "max_completion_tokens" if model.startswith("gpt-5") else "max_tokens"
    kwargs[tokens_key] = max_out
    resp = await asyncio.wait_for(client.chat.completions.create(**kwargs),
                                   timeout=120.0)
    raw = resp.choices[0].message.content or ""
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0
    return raw, pt, ct


async def _gemini_call(client, *, model, system, user, temperature, max_out):
    cfg = genai_types.GenerateContentConfig(
        system_instruction=system, temperature=temperature, max_output_tokens=max_out,
    )
    resp = await asyncio.wait_for(
        client.aio.models.generate_content(model=model, contents=user, config=cfg),
        timeout=120.0,
    )
    raw = (resp.text or "").strip()
    pt = resp.usage_metadata.prompt_token_count or 0 if resp.usage_metadata else 0
    ct = resp.usage_metadata.candidates_token_count or 0 if resp.usage_metadata else 0
    return raw, pt, ct


async def _call_one(clients, *, rid, layer_a, cell, base_seed, max_retries=5):
    system, user = build_prompt(layer_a, cell)
    seed = per_respondent_seed(rid, cell["outcome"], base_seed)
    max_out = 800 if cell["sampling"] == "vs_cot" else 10

    base = dict(
        respondent_id=rid, outcome=cell["outcome"],
        persona_config="C7", persona_source="natural",
        model=cell["model"], temperature=cell["temperature"],
        sampling=cell["sampling"], address_mode="ben", cot=False,
        base_seed=base_seed, draw_seed=seed,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    backoff = 1
    raw, pt, ct = "", 0, 0
    for attempt in range(max_retries):
        try:
            if cell["model"].startswith("gpt-"):
                raw, pt, ct = await _openai_call(
                    clients["openai"], model=cell["model"], system=system, user=user,
                    temperature=cell["temperature"], max_out=max_out, seed=base_seed,
                )
            else:
                raw, pt, ct = await _gemini_call(
                    clients["gemini"], model=cell["model"], system=system, user=user,
                    temperature=cell["temperature"], max_out=max_out,
                )
            break
        except Exception as e:
            err = str(e)
            if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < max_retries - 1:
                await asyncio.sleep(2 ** backoff); backoff += 1
                continue
            return {**base, "predicted_value": None, "dist": None,
                    "raw_response": f"API_ERROR: {err}", "prompt_tokens": 0,
                    "completion_tokens": 0, "parse_status": "api_error"}

    if cell["sampling"] == "vs_cot":
        val, dist = parse_vs_cot(raw, cell["valid_range"], seed)
    else:
        val, dist = parse_direct(raw, cell["valid_range"]), None
    return {**base, "predicted_value": val, "dist": dist, "raw_response": raw,
            "prompt_tokens": pt, "completion_tokens": ct,
            "parse_status": "ok" if val is not None else "parse_failure"}


async def run_cell(clients, cell_name: str, cell: dict, rids: list[str],
                    concurrency: int, base_seed: int) -> None:
    out_dir = NAT_CALIB_DIR / cell["subdir"] / cell["outcome"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / cell["filename"]

    done: set[str] = set()
    if fp.exists():
        for line in fp.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("parse_status") == "ok":
                    done.add(r["respondent_id"])
            except Exception:
                continue
        if done:
            print(f"  [{fp.name}] resume: {len(done)} already ok")

    personas = {rid: (NAT_LAYER_A_DIR / "C7" / f"{rid}.md").read_text(encoding="utf-8")
                for rid in rids
                if (NAT_LAYER_A_DIR / "C7" / f"{rid}.md").exists()}
    missing = [rid for rid in rids if rid not in personas]
    if missing:
        print(f"  ⚠  {len(missing)} personas missing natural rewrite (skipping)")

    fh = fp.open("a", encoding="utf-8")
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def worker(rid: str) -> None:
        if rid in done or rid not in personas:
            return
        async with sem:
            rec = await _call_one(clients, rid=rid, layer_a=personas[rid],
                                  cell=cell, base_seed=base_seed)
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            async with write_lock:
                fh.write(line); fh.flush()

    tasks = [asyncio.create_task(worker(rid)) for rid in rids]
    total = len(tasks)
    print(f"  [{cell_name}] scheduling {total} API calls  concurrency={concurrency}")
    completed = 0
    t0 = time.time()
    log_every = max(50, total // 100)
    try:
        for fut in asyncio.as_completed(tasks):
            await fut
            completed += 1
            if completed % log_every == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total - completed) / rate if rate > 0 else 0
                print(f"    {cell_name}: {completed}/{total} "
                      f"({completed/total*100:.1f}%)  rate={rate:.1f}/s  "
                      f"ETA={int(remaining)//60}m{int(remaining)%60:02d}s")
    finally:
        fh.close()


def list_respondents() -> list[str]:
    d = NAT_LAYER_A_DIR / "C7"
    if not d.exists():
        raise SystemExit(f"Natural personas not found under {d}.\n"
                         "Run scripts/natural_rewrite_personas.py first.")
    return sorted(p.stem for p in d.glob("TGSS_*.md"))


def _make_clients(need_openai: bool, need_gemini: bool) -> dict:
    clients: dict = {}
    if need_openai:
        if AsyncOpenAI is None: raise SystemExit("openai SDK not installed.")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            import getpass; key = getpass.getpass("OpenAI API key: ").strip()
        clients["openai"] = AsyncOpenAI(api_key=key)
    if need_gemini:
        if genai is None: raise SystemExit("google-genai SDK not installed.")
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            import getpass; key = getpass.getpass("Google / Gemini API key: ").strip()
        clients["gemini"] = genai.Client(api_key=key)
    return clients


def main() -> None:
    p = argparse.ArgumentParser(description="Run best-setup calibration with natural personas.")
    p.add_argument("--only", choices=list(CELLS), default=None,
                   help="Run only one cell (default: both).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rids = list_respondents()
    cell_names = [args.only] if args.only else list(CELLS)

    print("=== run_calibration_natural.py ===")
    print(f"  natural personas    : n={len(rids)} under {NAT_LAYER_A_DIR}/C7")
    print(f"  output cache        : {NAT_CALIB_DIR}")
    print(f"  cells to run        : {cell_names}")
    for name in cell_names:
        c = CELLS[name]
        print(f"     - {name:10s}  {c['model']:26s}  T={c['temperature']}  {c['sampling']}")

    n_calls = sum(len(rids) for _ in cell_names)
    print(f"  total API calls     : ~{n_calls}")
    if args.dry_run:
        print("\n--dry-run: not calling the API."); return

    clients = _make_clients(
        need_openai=any(CELLS[n]["model"].startswith("gpt-") for n in cell_names),
        need_gemini=any(CELLS[n]["model"].startswith("gemini") for n in cell_names),
    )

    async def go():
        for name in cell_names:
            await run_cell(clients, name, CELLS[name], rids,
                           args.concurrency, args.seed)

    asyncio.run(go())

    # Mirror calibration results to durable backup location
    import shutil
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_fp in NAT_CALIB_DIR.rglob("*.jsonl"):
        rel = src_fp.relative_to(NAT_CALIB_DIR)
        dst_fp = BACKUP_DIR / rel
        dst_fp.parent.mkdir(parents=True, exist_ok=True)
        if not dst_fp.exists() or dst_fp.read_bytes() != src_fp.read_bytes():
            shutil.copy2(src_fp, dst_fp); copied += 1
    print(f"✓ Mirrored to backup: {BACKUP_DIR}  ({copied} file(s))")
    print("\n✓ Done. Analyze with compare_natural_vs_structured.py.")


if __name__ == "__main__":
    main()
