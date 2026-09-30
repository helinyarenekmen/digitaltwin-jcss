"""
run_context_experiment_v2.py — S-vs-P context experiment (final-pick protocols).

Uses the FINAL protocols validated in calibration_final_v2:

  Likert DVs → C7 + GPT-5.4-mini + VS-CoT + T=0.8 + ben + political context prepend
  Binary DV  → C7 + GPT-4o-mini  + Direct + T=0.8 + ben + political context prepend

Differences from the v1 runner (run_context_experiment.py):
  - Likert model: gemini-2.5-flash-lite → gpt-5.4-mini
  - Binary temperature: 0.0 → 0.8
  - Political context paragraph prepended to system prompt for ALL DVs
  - New experiment folder: outputs/experiments/exp12_PC_gpt5.4mini_gpt4omini/

Design
------
Two vignettes:
  S — security context   (Suriye operasyonları) — same across versions
  P — peace context      (Newroz variants; see PEACE_VIGNETTES)

Two dependent variables, each with its OWN inference protocol:

  dv_legitimacy       5-pt Likert   → C7 + GPT-5.4-mini  + VS-CoT + T=0.8 + ben + PC
  dv_behavioral_intent  BINARY      → C7 + GPT-4o-mini   + Direct + T=0.8 + ben + PC
      1 = Katılırdım, 2 = Katılmazdım

Usage
-----
    # full run with the new final-pick protocols (uses v3 vignette):
    python scripts/run_context_experiment_v2.py --peace v3_final

    # only Peace condition (fast; if you already have the S half elsewhere):
    python scripts/run_context_experiment_v2.py --peace v3_final --conditions P

    # regenerate CSVs from cached jsonl without hitting the API:
    python scripts/run_context_experiment_v2.py --peace v3_final --export-csvs

Peace versions and their experiment folders
-------------------------------------------
  v1        → outputs/experiments/exp08_context_only/           (legacy protocols)
  v3        → outputs/experiments/exp10_context_peacev3/        (legacy protocols)
  v4        → outputs/experiments/exp11_context_peacev4/        (legacy protocols)
  v3_final  → outputs/experiments/exp12_PC_gpt5.4mini_gpt4omini/     (NEW final-pick protocols)
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None  # type: ignore
    genai_types = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
LAYER_A_DIR = Path(os.environ.get(
    "LAYER_A_DIR",
    str(Path.home() / "Library" / "Caches" / "digitaltwin_layer_a"),
))
EXPERIMENT_OUT = ROOT / "outputs" / "experiments"

GEMINI_MODEL = "gemini-2.5-flash-lite"       # kept for compatibility, unused in v2
GPT54MINI_MODEL = "gpt-5.4-mini"             # new Likert model (final pick)
GEMINI_ALIAS = "gemini25flashlite"

MODEL_ALIASES = {
    "gpt-5.4-mini":             "gpt54mini",
    "gpt-4o-mini":              "gpt4omini",
    "gemini-flash-lite-latest": GEMINI_ALIAS,
    "gemini-2.5-flash-lite":    GEMINI_ALIAS,
    "gemini-2.5-flash":         "gemini25flash",
}
_TEMP_LABELS = {0.0: "T0", 0.3: "T03", 0.4: "T04", 0.7: "T07", 0.8: "T08", 1.0: "T1"}


# ---------------------------------------------------------------------------
# Peace vignette bank + experiment mapping
# ---------------------------------------------------------------------------
S_VIGNETTE = (
    "Milli Savunma Bakanlığı, Suriye'nin kuzeyinde Kürt silahlı gruplarıyla "
    "sürdürülen çatışmalara ilişkin açıklama yaptı. Açıklamada, son haftalarda "
    "bölgede silahlı çatışmaların yoğunlaştığı ve Türk Silahlı Kuvvetleri'nin "
    "operasyonlarını sürdürdüğü bildirildi. Kayıpların yaşandığı ve bölgede "
    "güvenlik önlemlerinin artırıldığı aktarıldı.\n\n"
    "Bu politik bağlamda Ankara'da düzenlenen yürüyüşte bir grup, Kürtçe "
    "anadilde eğitim talebini dile getirdi."
)

P_VIGNETTE_V1 = (
    "Bu yıl Newroz kutlamaları farklı bir atmosferde geçti. Süregelen barış "
    "ve müzakere sürecinin yarattığı iyimserlik ortamında pek çok şehirde "
    "Kürt ve Türk vatandaşlar bir arada kutlama yaptı. Gözlemciler, "
    "kutlamaların bu yıl önceki yıllara kıyasla daha kapsayıcı ve şenlikli "
    "bir havada ilerlediğini aktardı.\n\n"
    "Bu politik bağlamda Ankara'da düzenlenen yürüyüşte bir grup, Kürtçe "
    "anadilde eğitim talebini dile getirdi."
)

P_VIGNETTE_V3 = (
    "Mart ayında, hükümet ile Kürt siyasi temsilcileri arasındaki görüşmelerin "
    "devam ettiği ve tarafların çözüm sürecine ilişkin olumlu açıklamalar yaptığı "
    "bir dönemde Newroz kutlamaları geniş katılımla gerçekleşti. Diyarbakır ve "
    "İstanbul'daki meydanlar gün boyunca kalabalıktı. Cumhurbaşkanı ile çeşitli "
    "siyasi partilerin liderleri Newroz mesajları yayımladı.\n\n"
    "Bu dönemde Ankara'da düzenlenen bir yürüyüşte bir grup, Kürtçe anadilde "
    "eğitim talebini dile getirdi."
)

P_VIGNETTE_V4 = (
    "Mart ayında, barış ve müzakere sürecinin devam ettiği bir ortamda Newroz "
    "kutlamaları geniş katılımla gerçekleşti. Diyarbakır ve İstanbul'daki meydanlar "
    "gün boyunca kalabalıktı; kutlamalara çok sayıda aile ve çocuk da katıldı. "
    "Cumhurbaşkanı ile çeşitli siyasi partilerin liderleri Newroz mesajları yayımladı.\n\n"
    "Bu bağlamda Ankara'da düzenlenen yürüyüşte bir grup, Kürtçe "
    "anadilde eğitim talebini dile getirdi."
)

PEACE_VIGNETTES: dict[str, tuple[str, str, str]] = {
    # peace_version : (experiment_id, condition_label, vignette_text)
    "v1":              ("exp08_context_only",                 "Peace context",           P_VIGNETTE_V1),
    "v3":              ("exp10_context_peacev3",              "Peace context (v3)",      P_VIGNETTE_V3),
    "v4":              ("exp11_context_peacev4",              "Peace context (v4)",      P_VIGNETTE_V4),
    # v2-final-pick uses the v3 vignette content but a NEW output folder and NEW protocols
    "v3_final":        ("exp12_PC_gpt5.4mini_gpt4omini",      "Peace context (v3)",      P_VIGNETTE_V3),
    # v3_final_noPC: same as v3_final but political-context prepend disabled (companion to exp13)
    "v3_final_noPC":   ("exp13_noPC_gpt5.4mini_gpt4omini",    "Peace context (v3)",      P_VIGNETTE_V3),
}


# ---------------------------------------------------------------------------
# Survey items
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SurveyItem:
    item_id: str
    label: str
    question: str
    options: dict[int, str]
    scale_type: str
    persona_config: str
    model: str
    sampling: str
    temperature: float

    @property
    def valid_range(self) -> tuple[int, int]:
        ks = sorted(self.options.keys())
        return ks[0], ks[-1]


_LIKERT5 = {
    1: "Kesinlikle katılmıyorum",
    2: "Katılmıyorum",
    3: "Ne katılıyorum ne katılmıyorum",
    4: "Katılıyorum",
    5: "Kesinlikle katılıyorum",
}

ITEM_DV_LEGITIMACY = SurveyItem(
    item_id="dv_legitimacy",
    label="Movement legitimacy (5-pt Likert)",
    question=(
        "Tüm bunları göz önünde bulundurduğunuzda \"Kürtçe anadilde eğitim "
        "eylemi yapan bu grup meşru bir hak mücadelesi yürütüyor.\" ifadesine "
        "ne ölçüde katılıyorsunuz?"
    ),
    options=_LIKERT5, scale_type="likert",
    persona_config="C7", model=GPT54MINI_MODEL, sampling="vs_cot", temperature=0.8,
)

ITEM_DV_POLICY_SUPPORT = SurveyItem(
    item_id="dv_policy_support",
    label="Policy support (5-pt Likert)",
    question=(
        "Tüm bunları göz önünde bulundurduğunuzda \"Devlet okullarında Kürtçe "
        "anadilde eğitime yasal olarak izin verilmelidir.\" ifadesine ne "
        "ölçüde katılıyorsunuz?"
    ),
    options=_LIKERT5, scale_type="likert",
    persona_config="C7", model=GPT54MINI_MODEL, sampling="vs_cot", temperature=0.8,
)

ITEM_DV_BEHAVIORAL_INTENT = SurveyItem(
    item_id="dv_behavioral_intent",
    label="Behavioral intent (binary; 1 = Katılırdım, 2 = Katılmazdım)",
    question=(
        "Tüm bunları göz önünde bulundurduğunuzda, böyle bir yürüyüş "
        "düzenlense yürüyüşe bizzat katılır mıydınız?"
    ),
    options={1: "Katılırdım", 2: "Katılmazdım"}, scale_type="binary",
    persona_config="C7", model="gpt-4o-mini", sampling="direct", temperature=0.8,
)

SURVEY_ITEMS: list[SurveyItem] = [
    ITEM_DV_LEGITIMACY, ITEM_DV_BEHAVIORAL_INTENT,
]


# ---------------------------------------------------------------------------
# Conditions (peace_version chosen at runtime)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    code: str
    factor: str
    label: str
    vignette: str


def make_conditions(peace_version: str) -> list[Condition]:
    _, peace_label, peace_text = PEACE_VIGNETTES[peace_version]
    return [
        Condition("S", "context", "Security context", S_VIGNETTE),
        Condition("P", "context", peace_label,        peace_text),
    ]


# ---------------------------------------------------------------------------
# Prompts (identical across peace versions)
# ---------------------------------------------------------------------------
SYSTEM_BEN = (
    "Aşağıda Türkiye'de yaşayan birinin kişisel profili yer almaktadır. "
    "Bu kişinin instead of geçerek bir ankete cevap vereceksin. "
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


# Political context prepended to the system prompt for the v2 final protocol
# (adopted in calibration_final_v2 as part of the Stage 4 ablation study).
POLITICAL_CONTEXT_TR = (
    "Türkiye'nin ideolojik yapısı iki temel eksen üzerinde şekillenir: "
    "sol-sağ siyasi ekseni ve seküler-dindar kültürel ekseni. "
    "Sol tarafta sosyalizm, feminizm, sosyal demokrasi ve Kürt ulusal "
    "hareketi birbirine yakın konumlanır. Sağ tarafta Türk milliyetçiliği "
    "ve Kemalizm yer alır. Muhafazakârlık ve İslamcılık kültürel eksenin "
    "dindar ucunu oluşturur."
)


# Module-level flag toggled by the --no-pc CLI switch (default: PC prepend ON).
_INCLUDE_PC = True


def set_include_pc(flag: bool) -> None:
    global _INCLUDE_PC
    _INCLUDE_PC = flag


def build_prompt(layer_a_text: str, condition: Condition,
                 item: SurveyItem) -> tuple[str, str]:
    # Prepend political context iff the flag is set (default ON for v3_final)
    pc_prefix = (POLITICAL_CONTEXT_TR + "\n\n") if _INCLUDE_PC else ""
    system = (
        pc_prefix +
        SYSTEM_BEN + (VS_COT_SUFFIX if item.sampling == "vs_cot" else "")
    )
    options_str = "\n".join(f"{k} = {v}" for k, v in item.options.items())
    user = (
        f"{layer_a_text}\n\n"
        f"---\n\n"
        f"{condition.vignette}\n\n"
        f"---\n\n"
        f"Soru: {item.question}\n\n"
        f"Seçenekler:\n{options_str}"
    )
    if item.sampling != "vs_cot":
        user += "\n\nCevap (sadece sayı):"
    return system, user


# ---------------------------------------------------------------------------
# Parsing / seeding / naming
# ---------------------------------------------------------------------------
def per_respondent_seed(rid: str, cond: str, item_id: str, base_seed: int) -> int:
    payload = f"{base_seed}|{rid}|{cond}|{item_id}".encode("utf-8")
    return int.from_bytes(hashlib.md5(payload).digest()[:8], "big") & 0x7FFFFFFF


def parse_direct(raw: str, valid_range: tuple[int, int]) -> int | None:
    lo, hi = valid_range
    m = re.search(r"(\d+)", raw or "")
    if not m:
        return None
    v = int(m.group(1))
    return v if lo <= v <= hi else None


def parse_vs_cot(raw: str, valid_range: tuple[int, int],
                 seed: int) -> tuple[int | None, dict[int, float] | None]:
    import numpy as np
    m = re.search(r"DAĞILIM:\s*(\{[^}]+\})", raw or "", re.DOTALL)
    if not m:
        return None, None
    try:
        dist_raw = json.loads(m.group(1))
        options = {int(k): float(v) for k, v in dist_raw.items()}
        lo, hi = valid_range
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


def make_cell_filename(experiment_id: str, persona_config: str,
                       model: str, temperature: float, sampling: str,
                       address_mode: str, seed: int) -> str:
    alias = MODEL_ALIASES.get(model, model.replace("-", "").replace(".", ""))
    tcode = _TEMP_LABELS.get(temperature, f"T{temperature}".replace(".", ""))
    return (f"{experiment_id}_{persona_config}_{alias}_{tcode}_"
            f"{address_mode}_{sampling}_s{seed}.jsonl")


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------
def load_persona(config_id: str, respondent_id: str) -> str:
    return (LAYER_A_DIR / config_id / f"{respondent_id}.md").read_text(encoding="utf-8")


def list_respondents(config_id: str) -> list[str]:
    d = LAYER_A_DIR / config_id
    return sorted(p.stem for p in d.glob("TGSS_*.md"))


def preload_personas(config_id: str, rids: list[str]) -> dict[str, str]:
    return {rid: load_persona(config_id, rid) for rid in rids}


# ---------------------------------------------------------------------------
# Provider dispatchers
# ---------------------------------------------------------------------------
async def _openai_call(client, *, model: str, system: str, user: str,
                       temperature: float, max_out: int,
                       seed: int | None) -> tuple[str, int, int]:
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=temperature,
        seed=seed if temperature == 0 else None,
    )
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_out
    else:
        kwargs["max_tokens"] = max_out
    resp = await asyncio.wait_for(client.chat.completions.create(**kwargs),
                                  timeout=120.0)
    raw = resp.choices[0].message.content or ""
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0
    return raw, pt, ct


async def _gemini_call(client, *, model: str, system: str, user: str,
                       temperature: float, max_out: int) -> tuple[str, int, int]:
    cfg = genai_types.GenerateContentConfig(
        system_instruction=system, temperature=temperature,
        max_output_tokens=max_out,
    )
    resp = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=model, contents=user, config=cfg,
        ),
        timeout=120.0,
    )
    raw = (resp.text or "").strip()
    pt = resp.usage_metadata.prompt_token_count or 0 if resp.usage_metadata else 0
    ct = resp.usage_metadata.candidates_token_count or 0 if resp.usage_metadata else 0
    return raw, pt, ct


async def _call_one(clients: dict, *, respondent_id: str, condition: Condition,
                    item: SurveyItem, layer_a_text: str, address_mode: str,
                    base_seed: int, max_retries: int = 5) -> dict:
    system, user = build_prompt(layer_a_text, condition, item)
    seed = per_respondent_seed(respondent_id, condition.code,
                                item.item_id, base_seed)
    max_out = 800 if item.sampling == "vs_cot" else 10

    base = {
        "respondent_id":  respondent_id, "experiment_id": None,
        "condition":      condition.code, "factor": condition.factor,
        "condition_label": condition.label,
        "item_id":        item.item_id, "scale_type": item.scale_type,
        "persona_config": item.persona_config, "model": item.model,
        "temperature":    item.temperature, "address_mode": address_mode,
        "sampling":       item.sampling, "base_seed": base_seed,
        "draw_seed":      seed,
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    backoff = 1
    raw, pt, ct = "", 0, 0
    for attempt in range(max_retries):
        try:
            if item.model.startswith("gpt-"):
                raw, pt, ct = await _openai_call(
                    clients["openai"], model=item.model, system=system, user=user,
                    temperature=item.temperature, max_out=max_out, seed=base_seed,
                )
            elif item.model.startswith("gemini"):
                raw, pt, ct = await _gemini_call(
                    clients["gemini"], model=item.model, system=system, user=user,
                    temperature=item.temperature, max_out=max_out,
                )
            else:
                return {**base, "predicted_value": None, "dist": None,
                        "raw_response": f"API_ERROR: unknown provider for model={item.model}",
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "parse_status": "api_error"}
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

    if item.sampling == "vs_cot":
        val, dist = parse_vs_cot(raw, item.valid_range, seed)
    else:
        val, dist = parse_direct(raw, item.valid_range), None

    return {**base, "predicted_value": val, "dist": dist, "raw_response": raw,
            "prompt_tokens": pt, "completion_tokens": ct,
            "parse_status": "ok" if val is not None else "parse_failure"}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def run_cell(*, clients: dict, experiment_id: str,
                   respondents: list[str], conditions: list[Condition],
                   items: list[SurveyItem], address_mode: str,
                   base_seed: int, concurrency: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    grouped: dict[tuple, list[SurveyItem]] = defaultdict(list)
    for it in items:
        grouped[(it.persona_config, it.model, it.sampling, it.temperature)].append(it)

    personas_by_cfg = {cfg: preload_personas(cfg, respondents)
                       for cfg in {it.persona_config for it in items}}
    print(f"  preloaded personas for configs: {list(personas_by_cfg)}")

    file_handles: dict[Path, Any] = {}
    done_by_file: dict[Path, set[tuple[str, str, str]]] = {}
    for (cfg, model, samp, temp), _ in grouped.items():
        fname = make_cell_filename(experiment_id, cfg, model, temp,
                                    samp, address_mode, base_seed)
        fp = out_dir / fname
        done: set[tuple[str, str, str]] = set()
        if fp.exists():
            for line in fp.open(encoding="utf-8"):
                try:
                    r = json.loads(line)
                    if r.get("parse_status") == "ok":
                        done.add((r["respondent_id"], r["condition"], r["item_id"]))
                except Exception:
                    continue
            if done:
                print(f"  [{fp.name}] resume: {len(done)} triples already done")
        done_by_file[fp] = done
        file_handles[fp] = fp.open("a", encoding="utf-8")

    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    async def worker(rid: str, cond: Condition, item: SurveyItem, fp: Path) -> None:
        if (rid, cond.code, item.item_id) in done_by_file[fp]:
            return
        async with sem:
            layer_a = personas_by_cfg[item.persona_config][rid]
            rec = await _call_one(
                clients, respondent_id=rid, condition=cond, item=item,
                layer_a_text=layer_a, address_mode=address_mode,
                base_seed=base_seed,
            )
            rec["experiment_id"] = experiment_id
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            async with write_lock:
                file_handles[fp].write(line); file_handles[fp].flush()

    tasks: list[asyncio.Task] = []
    for rid in respondents:
        cond_order = list(conditions)
        random.Random(per_respondent_seed(rid, "ORDER_COND", "ORDER",
                                          base_seed)).shuffle(cond_order)
        for cond in cond_order:
            item_order = list(items)
            random.Random(per_respondent_seed(rid, cond.code, "ORDER_ITEM",
                                              base_seed)).shuffle(item_order)
            for item in item_order:
                fname = make_cell_filename(experiment_id, item.persona_config,
                                            item.model, item.temperature,
                                            item.sampling, address_mode, base_seed)
                fp = out_dir / fname
                tasks.append(asyncio.create_task(worker(rid, cond, item, fp)))

    total = len(tasks)
    print(f"  scheduling {total} API calls "
          f"({len(respondents)} resp × {len(conditions)} cond × {len(items)} items)  "
          f"concurrency={concurrency}")

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
        for fh in file_handles.values():
            fh.close()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
_META_COLS_WIDE = [
    "experiment_id", "respondent_id", "condition", "factor",
    "persona_config", "model", "temperature", "sampling",
    "address_mode", "base_seed",
]


def export_csvs(out_dir: Path, items: list[SurveyItem]) -> None:
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for fp in sorted(out_dir.glob("*.jsonl")):
        for line in fp.open(encoding="utf-8"):
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    if not records:
        print(f"  no records under {out_dir}"); return

    long_path = csv_dir / "all_items_long.csv"
    long_cols = _META_COLS_WIDE + ["item_id", "scale_type", "predicted_value",
                                    "parse_status", "draw_seed"]
    with long_path.open("w", newline="", encoding="utf-8") as f:
        w = csv_lib.writer(f); w.writerow(long_cols)
        for r in records:
            meta = [r.get(c) for c in _META_COLS_WIDE]
            w.writerow(meta + [r.get("item_id"), r.get("scale_type"),
                               r.get("predicted_value"), r.get("parse_status"),
                               r.get("draw_seed")])
    print(f"  ✓ {long_path.relative_to(ROOT)}  ({len(records)} rows)")

    for it in items:
        path = csv_dir / f"{it.item_id}.csv"
        wide_cols = list(_META_COLS_WIDE) + [f"{it.item_id}_pred"]
        item_records = [r for r in records if r.get("item_id") == it.item_id]
        by_key: dict[tuple[str, str], dict] = {}
        for r in item_records:
            key = (r["respondent_id"], r["condition"])
            row = by_key.setdefault(key, {c: r.get(c) for c in _META_COLS_WIDE})
            row[f"{it.item_id}_pred"] = r.get("predicted_value")
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv_lib.writer(f); w.writerow(wide_cols)
            for k in sorted(by_key):
                row = by_key[k]
                w.writerow([row.get(c) for c in wide_cols])
        print(f"  ✓ {path.relative_to(ROOT)}  ({len(by_key)} rows)")


# ---------------------------------------------------------------------------
# Client factory / main
# ---------------------------------------------------------------------------
def _make_clients(need_openai: bool, need_gemini: bool) -> dict:
    clients: dict = {}
    if need_openai:
        if AsyncOpenAI is None:
            raise SystemExit("openai SDK not installed.")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            import getpass; key = getpass.getpass("OpenAI API key: ").strip()
        if not key:
            raise SystemExit("ERROR: OPENAI_API_KEY missing.")
        clients["openai"] = AsyncOpenAI(api_key=key)
    if need_gemini:
        if genai is None:
            raise SystemExit("google-genai SDK not installed. `pip install google-genai`.")
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            import getpass; key = getpass.getpass("Google / Gemini API key: ").strip()
        if not key:
            raise SystemExit("ERROR: GEMINI_API_KEY missing.")
        clients["gemini"] = genai.Client(api_key=key)
    return clients


def parse_conditions_arg(s: str, all_conditions: list[Condition]) -> list[Condition]:
    if not s or s == "all":
        return list(all_conditions)
    codes = {c.strip() for c in s.split(",") if c.strip()}
    chosen = [c for c in all_conditions if c.code in codes]
    if not chosen:
        raise SystemExit(f"--conditions matched nothing; valid: "
                         f"{[c.code for c in all_conditions]}")
    return chosen


def parse_items_arg(s: str) -> list[SurveyItem]:
    if not s or s == "all":
        return list(SURVEY_ITEMS)
    ids = {x.strip() for x in s.split(",") if x.strip()}
    chosen = [it for it in SURVEY_ITEMS if it.item_id in ids]
    if not chosen:
        raise SystemExit(f"--items matched nothing; valid: "
                         f"{[it.item_id for it in SURVEY_ITEMS]}")
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description="Consolidated S vs P context experiment runner.")
    p.add_argument("--peace", required=True, choices=list(PEACE_VIGNETTES),
                   help="Which Peace vignette to use (selects target experiment folder).")
    p.add_argument("--address-mode", type=str, choices=["ben", "sen"], default="ben")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conditions", type=str, default="all")
    p.add_argument("--items", type=str, default="all")
    p.add_argument("--n-respondents", type=int, default=None)
    p.add_argument("--random-sample", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--export-csvs", action="store_true")
    p.add_argument("--no-pc", action="store_true",
                   help="Disable political-context prepend (default: ON). "
                        "Use with --peace v3_final_noPC for exp13.")
    args = p.parse_args()
    set_include_pc(not args.no_pc)  # PC prepend ON unless --no-pc

    experiment_id, _peace_label, _peace_text = PEACE_VIGNETTES[args.peace]
    conditions = parse_conditions_arg(args.conditions, make_conditions(args.peace))
    items = parse_items_arg(args.items)

    configs_needed = sorted({it.persona_config for it in items})
    respondent_sets = [set(list_respondents(cfg)) for cfg in configs_needed]
    respondents = sorted(set.intersection(*respondent_sets)) if respondent_sets else []

    if args.random_sample is not None:
        if args.random_sample > len(respondents):
            raise SystemExit(f"--random-sample {args.random_sample} > available {len(respondents)}")
        rng = random.Random(args.sample_seed)
        respondents = sorted(rng.sample(respondents, args.random_sample))
    elif args.n_respondents is not None:
        respondents = respondents[: args.n_respondents]

    out_dir = EXPERIMENT_OUT / experiment_id
    n_calls = len(respondents) * len(conditions) * len(items)

    print("=== run_context_experiment.py ===")
    print(f"  peace version : {args.peace}")
    print(f"  experiment_id : {experiment_id}")
    print(f"  personas       : n={len(respondents)}")
    print(f"  conditions     : {[c.code for c in conditions]}  ({len(conditions)})")
    print(f"  items          : {[it.item_id for it in items]}  ({len(items)})")
    for it in items:
        print(f"     - {it.item_id:24s} → {it.persona_config} + {it.model} + "
              f"{it.sampling} + T={it.temperature}")
    print(f"  P vignette     : {args.peace}  ·  "
          f"{PEACE_VIGNETTES[args.peace][2][:80]}...")
    print(f"  output dir     : {out_dir.relative_to(ROOT)}")
    print(f"  total API calls: {n_calls}")

    pricing = {
        "gpt-4o-mini":              (0.15 / 1_000_000, 0.60 / 1_000_000),
        "gpt-5.4-mini":             (0.25 / 1_000_000, 2.00 / 1_000_000),  # approximate
        "gemini-2.5-flash-lite":    (0.10 / 1_000_000, 0.40 / 1_000_000),
        "gemini-flash-lite-latest": (0.10 / 1_000_000, 0.40 / 1_000_000),
    }
    tot_cost = 0.0
    for it in items:
        calls = len(respondents) * len(conditions)
        pt, ct = 1000, (800 if it.sampling == "vs_cot" else 10)
        in_rate, out_rate = pricing.get(it.model, (0.0, 0.0))
        tot_cost += calls * (pt * in_rate + ct * out_rate)
    print(f"  est cost       : ~${tot_cost:.2f}")

    if args.export_csvs:
        if not out_dir.exists():
            raise SystemExit(f"No output dir {out_dir}.")
        export_csvs(out_dir, items); return
    if args.dry_run:
        print("\n--dry-run: not calling the API."); return

    clients = _make_clients(
        need_openai=any(it.model.startswith("gpt-") for it in items),
        need_gemini=any(it.model.startswith("gemini") for it in items),
    )

    asyncio.run(run_cell(
        clients=clients, experiment_id=experiment_id,
        respondents=respondents, conditions=conditions, items=items,
        address_mode=args.address_mode, base_seed=args.seed,
        concurrency=args.concurrency, out_dir=out_dir,
    ))

    print("\n✓ All calls done. Exporting per-item CSVs …")
    export_csvs(out_dir, items)


if __name__ == "__main__":
    main()
