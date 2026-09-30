"""
natural_rewrite_personas.py — Rewrite C7 Layer A personas as natural
Turkish biographies (3rd person), for the "natural persona" ablation.

For each structured C7 persona in
    ~/Library/Caches/digitaltwin_layer_a/C7/TGSS_*.md
this script asks GPT-4o-mini to produce a flowing 3rd-person biography
that preserves every fact from the structured labels but adds no new
information, no interpretation, no editorial voice. Output is saved to
    ~/Library/Caches/digitaltwin_layer_a_natural/C7/TGSS_*.md

The natural personas can then be fed to the calibration runner in place
of the structured Layer A markdown, to test how sensitive calibration
metrics are to persona format.

Usage
-----
    export OPENAI_API_KEY=...
    python scripts/natural_rewrite_personas.py                 # full run (2615)
    python scripts/natural_rewrite_personas.py --random-sample 20 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import random
import time
from pathlib import Path
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path.home() / "Library" / "Caches" / "digitaltwin_layer_a" / "C7"
DST_DIR = Path.home() / "Library" / "Caches" / "digitaltwin_layer_a_natural" / "C7"
# Durable backup mirror location (structured under outputs/):
BACKUP_DIR = ROOT / "outputs" / "calibration_backup" / "natural_rewrite" / "personas" / "C7"

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "Sen yapılandırılmış anket profillerini doğal Türkçe biyografilere "
    "dönüştüren bir yazarsın.\n\n"
    "Kurallar (KATI):\n"
    "1. Sana verilen tüm bilgileri koru; hiçbirini atlama.\n"
    "2. YENİ bilgi EKLEME. Verilmemiş bir detayı uydurma "
    "(memleket, iş yeri adı, aile üye adları, geçmiş olaylar, vb.).\n"
    "3. YORUM veya değer yargısı katma. Kişinin görüşlerini olduğu gibi aktar.\n"
    "4. Verideki 'bilgi yok' işaretlerini yeniden ifade etmeye gerek yok — "
    "o bilgiyi bilmiyorsan sadece bahsetme.\n"
    "5. 3. şahıs, biyografik ton kullan (\"O 45 yaşında bir kadındır. "
    "İstanbul'da yaşar…\" gibi). Cümlelere \"O\" ile başlamak zorunda değilsin; "
    "\"45 yaşında bir kadındır.\" veya \"Ali Bey 45 yaşındadır.\" gibi doğal "
    "başlangıçlar da uygundur.\n"
    "6. **Sayısal skorları asla ham haliyle aktarma.** Kaynakta 0-10 arası "
    "ölçek değerleri bulunabilir (kimlik yakınlığı, parti yakınlığı, "
    "memnuniyet, güven, tehdit algısı vb.). Bu değerleri asla \"AK Parti'ye "
    "yakınlık: 8\" veya \"Sünni kimlik yakınlığı 10\" gibi sayı olarak yazma; "
    "onun instead of ölçeğin anlamına uygun doğal ifadelere dönüştür: "
    "\"çok yakın hisseder\", \"oldukça yakın\", \"orta düzeyde yakın\", "
    "\"biraz yakın\", \"uzak\", \"hiç yakın hissetmez\" gibi. Aynısı Likert "
    "skorları için de geçerli. Gelir dilimleri (\"25.000-29.999 TL\") ve "
    "yaş gibi somut sayısal verileri ise olduğu gibi bırakabilirsin.\n"
    "7. Akıcı ve okunabilir bir paragraf üret; başlık, madde işareti veya "
    "tekrar eden kalıp kullanma. Uzunluk kaynaktaki bilgi miktarına göre "
    "orantılı olsun (genellikle 1–2 paragraf).\n"
    "8. Sadece biyografiyi ver, giriş cümlesi veya meta-yorum ekleme."
)

USER_TEMPLATE = (
    "Aşağıdaki yapılandırılmış anket profilini yukarıdaki kurallara göre "
    "doğal bir Türkçe biyografiye dönüştür:\n\n"
    "---\n"
    "{persona}\n"
    "---"
)


async def rewrite_one(client: AsyncOpenAI, src_fp: Path, dst_fp: Path,
                       *, max_retries: int = 5) -> tuple[Path, bool, str]:
    """Rewrite one persona. Returns (dst_fp, ok, note)."""
    if dst_fp.exists() and dst_fp.stat().st_size > 0:
        return dst_fp, True, "skip:exists"

    persona = src_fp.read_text(encoding="utf-8")
    if not persona.strip():
        return dst_fp, False, "empty source"

    backoff = 1
    for attempt in range(max_retries):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": USER_TEMPLATE.format(
                            persona=persona)},
                    ],
                    temperature=0.3,   # small — factual rewrite
                    max_tokens=800,
                ),
                timeout=90.0,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                return dst_fp, False, "empty completion"
            dst_fp.parent.mkdir(parents=True, exist_ok=True)
            dst_fp.write_text(text + "\n", encoding="utf-8")
            return dst_fp, True, f"ok ({len(text)} chars)"
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "rate" in msg.lower()) and attempt < max_retries - 1:
                await asyncio.sleep(2 ** backoff); backoff += 1
                continue
            return dst_fp, False, f"api_error: {msg[:120]}"
    return dst_fp, False, "gave up"


async def run(rids: list[str], concurrency: int) -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        import getpass; key = getpass.getpass("OpenAI API key: ").strip()
    if not key:
        raise SystemExit("ERROR: OPENAI_API_KEY missing.")
    client = AsyncOpenAI(api_key=key)

    DST_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    stats = {"ok": 0, "skip": 0, "fail": 0}

    async def worker(rid: str) -> None:
        src_fp = SRC_DIR / f"{rid}.md"
        dst_fp = DST_DIR / f"{rid}.md"
        async with sem:
            _, ok, note = await rewrite_one(client, src_fp, dst_fp)
            if not ok:
                stats["fail"] += 1
                if stats["fail"] <= 5:
                    print(f"  ✗ {rid}: {note}")
            elif note.startswith("skip"):
                stats["skip"] += 1
            else:
                stats["ok"] += 1

    tasks = [asyncio.create_task(worker(rid)) for rid in rids]
    total = len(tasks)
    log_every = max(50, total // 200)
    completed = 0
    t0 = time.time()

    for fut in asyncio.as_completed(tasks):
        await fut
        completed += 1
        if completed % log_every == 0 or completed == total:
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (total - completed) / rate if rate > 0 else 0
            print(f"    {completed}/{total}  ({completed/total*100:.1f}%)  "
                  f"rate={rate:.1f}/s  ok={stats['ok']} skip={stats['skip']} "
                  f"fail={stats['fail']}  ETA={int(remaining)//60}m"
                  f"{int(remaining)%60:02d}s")

    print(f"\n✓ Done. ok={stats['ok']}  skip(existed)={stats['skip']}  "
          f"fail={stats['fail']}")
    print(f"Working output: {DST_DIR}")

    # Mirror to durable backup location
    mirror_to_backup()


def mirror_to_backup() -> None:
    """Sync all natural personas from the cache to the durable
    outputs/calibration_backup/natural_rewrite/ mirror."""
    import shutil
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_fp in sorted(DST_DIR.glob("TGSS_*.md")):
        dst_fp = BACKUP_DIR / src_fp.name
        # Only copy if content differs (or dst doesn't exist)
        if not dst_fp.exists() or dst_fp.read_bytes() != src_fp.read_bytes():
            shutil.copy2(src_fp, dst_fp)
            copied += 1
    # README noting what's inside
    readme = BACKUP_DIR.parent / "README.md"
    if not readme.exists():
        readme.write_text(
            "# natural_rewrite/\n\n"
            "Durable mirror of the natural-language persona rewrites used "
            "for the 'natural vs structured persona' calibration ablation.\n\n"
            "## Layout\n"
            "- `personas/C7/TGSS_*.md` — GPT-4o-mini rewrites of the C7 "
            "Layer A structured personas (3rd person biography, no new "
            "info, no raw 0–10 scale scores).\n"
            "- `calibration/` — populated later by "
            "`run_calibration_natural.py --mirror` with best-setup cell "
            "results (pacdemons, womenwork) run against these personas.\n\n"
            "## Working locations (cache)\n"
            "- Personas: `~/Library/Caches/digitaltwin_layer_a_natural/C7/`\n"
            "- Calibration: `~/Library/Caches/digitaltwin_calibration_natural/`\n",
            encoding="utf-8",
        )
    print(f"✓ Mirrored to backup: {BACKUP_DIR}  ({copied} new/updated)")


def list_respondents() -> list[str]:
    return sorted(p.stem for p in SRC_DIR.glob("TGSS_*.md"))


def main() -> None:
    p = argparse.ArgumentParser(description="Rewrite structured personas as natural biographies.")
    p.add_argument("--random-sample", type=int, default=None,
                   help="Rewrite only N randomly-sampled personas (for testing).")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rids = list_respondents()
    if args.random_sample:
        rng = random.Random(args.sample_seed)
        rids = sorted(rng.sample(rids, min(args.random_sample, len(rids))))

    # Rough cost estimate: 1000 in + 500 out at gpt-4o-mini rates
    cost = len(rids) * (1000 * 0.15/1_000_000 + 500 * 0.60/1_000_000)

    print("=== natural_rewrite_personas.py ===")
    print(f"  model         : {MODEL}")
    print(f"  personas      : n={len(rids)}  (config=C7)")
    print(f"  source dir    : {SRC_DIR}")
    print(f"  output dir    : {DST_DIR}")
    print(f"  concurrency   : {args.concurrency}")
    print(f"  est cost      : ~${cost:.2f}")
    print(f"  resume        : skips existing non-empty files")

    if args.dry_run:
        print("\n--dry-run: not calling the API."); return
    if AsyncOpenAI is None:
        raise SystemExit("openai SDK not installed.")

    asyncio.run(run(rids, args.concurrency))


if __name__ == "__main__":
    main()
