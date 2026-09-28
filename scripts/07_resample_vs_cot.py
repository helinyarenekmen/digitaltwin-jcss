"""
07_resample_vs_cot.py — VS-CoT JSONL'lerini per-respondent seed ile yeniden örnekle.

Mevcut sorun:
  Tüm 1692 respondent için tek bir rng (seed=0) kullanılıyor → numpy'nin
  sequence'i deterministik olduğu için tahminler dağılım çevresinde değil,
  rng'nin sırasında yığılıyor.

Çözüm:
  Her respondent için seed = md5(base_seed|respondent_id) → bağımsız
  örnekleme. raw_response saklı olduğu için API çağrısına gerek yok.

JSONL'lerin raw_response alanından DAĞILIM parse edilir, yeniden sample edilir,
`predicted_value` güncellenir. Dosya yerinde yazılır (raw_response korunur,
geri alınabilir).

Usage:
  python scripts/07_resample_vs_cot.py --temperature 0.8 --dry-run   # önizleme
  python scripts/07_resample_vs_cot.py --temperature 0.8             # yerinde yaz
  python scripts/07_resample_vs_cot.py --temperature 0.0 0.8         # ikisi de
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

CALIB = Path("/Users/helinekmen/Library/Caches/digitaltwin_calibration")

VALID_RANGES = {
    "pacdemons": (1, 2),
    "womenwork": (1, 5),
    "neilang":   (1, 3),
}

_TEMP_LABELS = {0.0: "T0", 0.3: "T03", 0.4: "T04", 0.7: "T07", 0.8: "T08", 1.0: "T1"}


def per_respondent_seed(respondent_id: str, base_seed: int = 0) -> int:
    """Stable, well-distributed seed per (base_seed, respondent_id)."""
    h = hashlib.md5(f"{base_seed}|{respondent_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFF  # positive int32


def parse_dist(raw: str, lo: int, hi: int) -> dict[int, float] | None:
    m = re.search(r"DAĞILIM:\s*(\{[^}]+\})", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
        opts = {int(k): float(v) for k, v in d.items() if lo <= int(k) <= hi}
        total = sum(opts.values())
        if total <= 0 or not opts:
            return None
        return {k: v / total for k, v in opts.items()}
    except Exception:
        return None


def resample_file(path: Path, outcome: str, base_seed: int, dry_run: bool) -> dict:
    lo, hi = VALID_RANGES[outcome]
    changes = {"total": 0, "ok": 0, "no_dist": 0, "changed": 0, "same": 0}
    new_lines = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            changes["total"] += 1

            if r.get("parse_status") != "ok":
                new_lines.append(line)
                continue
            changes["ok"] += 1

            dist = parse_dist(r.get("raw_response", ""), lo, hi)
            if dist is None:
                changes["no_dist"] += 1
                new_lines.append(line)
                continue

            seed = per_respondent_seed(r["respondent_id"], base_seed)
            rng = np.random.default_rng(seed)
            keys = list(dist.keys())
            probs = [dist[k] for k in keys]
            new_pred = int(rng.choice(keys, p=probs))

            old_pred = r.get("predicted_value")
            if new_pred != old_pred:
                changes["changed"] += 1
            else:
                changes["same"] += 1

            r["predicted_value"] = new_pred
            r["resample_seed"] = seed
            new_lines.append(json.dumps(r, ensure_ascii=False))

    if not dry_run:
        # Atomic write: tmp file then rename
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for ln in new_lines:
                f.write(ln + "\n")
        tmp.replace(path)

    return changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.0, 0.8],
                        help="Hangi temperature(lar)ı resample (default: 0.0 ve 0.8)")
    parser.add_argument("--base-seed", type=int, default=0,
                        help="Base seed (default 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Sadece preview, dosyaları değiştirme")
    args = parser.parse_args()

    if not CALIB.exists():
        print(f"ERROR: {CALIB} bulunamadı")
        sys.exit(1)

    print(f"{'DRY RUN' if args.dry_run else 'YAZILIYOR'} — base_seed={args.base_seed}")
    print()

    grand = {"total": 0, "ok": 0, "no_dist": 0, "changed": 0, "same": 0}

    for temp in args.temperature:
        temp_token = _TEMP_LABELS.get(temp, f"T{temp}".replace(".", ""))
        screening_dir = (CALIB / "screening_vs_cot" if temp == 0.0
                         else CALIB / f"screening_vs_cot_{temp_token}")
        if not screening_dir.exists():
            print(f"### Temperature {temp}: {screening_dir.name} yok, atlanıyor")
            continue
        print(f"### Temperature {temp} → {screening_dir.name} ###")
        for outcome_dir in sorted(screening_dir.iterdir()):
            if not outcome_dir.is_dir() or outcome_dir.name not in VALID_RANGES:
                continue
            outcome = outcome_dir.name
            pattern = f"*_gpt4omini_{temp_token}_*_vs_cot_*.jsonl"
            files = sorted(outcome_dir.glob(pattern))
            if not files:
                continue
            print(f"  [{outcome}] {len(files)} dosya")
            for path in files:
                c = resample_file(path, outcome, args.base_seed, args.dry_run)
                pct = c["changed"] / c["ok"] * 100 if c["ok"] else 0
                print(f"    {path.stem[:30]:<32} ok={c['ok']:<5} "
                      f"changed={c['changed']:<5} ({pct:5.1f}%) "
                      f"no_dist={c['no_dist']}")
                for k in grand:
                    grand[k] += c[k]
        print()

    print("=" * 60)
    print("ÖZET")
    print("=" * 60)
    total_ok = grand["ok"]
    if total_ok:
        print(f"Toplam ok satır: {total_ok:,}")
        print(f"  değişen: {grand['changed']:,} ({grand['changed']/total_ok*100:.1f}%)")
        print(f"  aynı:    {grand['same']:,}")
        print(f"  dağılım parse edilemedi: {grand['no_dist']:,}")
    if args.dry_run:
        print("\n(--dry-run: dosyalar değiştirilmedi)")
    else:
        print("\nBittikten sonra: python scripts/05_compute_metrics.py --variant vs_cot --temperature {0.0|0.8}")


if __name__ == "__main__":
    main()
