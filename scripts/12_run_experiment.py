"""
12_run_experiment.py — Within-subject factorial experiment runner.

One API call per (persona × condition × survey-item). For each turn the model
sees the vignette and exactly one question; this isolates each item's draw
from the others, makes parsing trivial (one DAĞILIM JSON per response), and
gives per-item resume granularity.

Study
-----
Digital Twins for Causal Inference: Social Movement Mobilization and Public
Opinion. Research question: how does support for Kurdish-language education
vary under combinations of political-context and protest-framing manipulations?

Design
------
2 × 3 full factorial, within-subject. Each persona is exposed to all six
vignette conditions; under each vignette every survey item is asked in its
own API call. Condition AND item order are independently randomized per
respondent (seed-controlled).

  Factor A — Political context (2 levels):  G (security)   B (peace)
  Factor B — Framing (3 levels):           C (none)  E (universal rights)  K (self-determination)
  Conditions: GC, GE, GK, BC, BE, BK

Survey items
------------
Three sections, each item gets its own API call:

  §6 manipulation_check : 6a context check, 6b framing check    (categorical)
  §7 dependent_variable : 7a legitimacy, 7b threat, ...         (Likert)
  §  additional         : TODO — extra DVs / moderators

Defaults
--------
C11 persona + VS-CoT + T=0.8 + GPT-4o-mini + first-person (ben).

Outputs
-------
outputs/experiments/<experiment_id>/
  <cell_id>.jsonl                       # one line per persona × condition × item
  csv/manipulation_checks.csv           # wide format, one row per persona × condition
  csv/dependent_variables.csv
  csv/additional_variables.csv          # only if ADDITIONAL_ITEMS is non-empty
  csv/all_items_long.csv                # tall format, one row per record

Use --export-csvs after a run to (re)generate the CSV exports without API calls.

Usage
-----
  python scripts/12_run_experiment.py                     # full run (defaults)
  python scripts/12_run_experiment.py --n-respondents 50  # pilot
  python scripts/12_run_experiment.py --conditions GC,BC  # subset of cells
  python scripts/12_run_experiment.py --items dv_legitimacy,dv_threat  # subset of items
  python scripts/12_run_experiment.py --dry-run           # plan only
  python scripts/12_run_experiment.py --export-csvs       # rebuild CSVs from JSONL
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


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
LAYER_A_DIR = Path(os.environ.get(
    "LAYER_A_DIR",
    str(Path.home() / "Library" / "Caches" / "digitaltwin_layer_a"),
))
EXPERIMENT_OUT = ROOT / "outputs" / "experiments"

MODEL_ALIASES = {
    "gpt-4o-mini":  "gpt4omini",
    "gpt-5.4-mini": "gpt54mini",
    "gpt-5-mini":   "gpt5mini",
}
_TEMP_LABELS = {0.0: "T0", 0.3: "T03", 0.4: "T04", 0.7: "T07", 0.8: "T08", 1.0: "T1"}


# ---------------------------------------------------------------------------
# Survey items
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurveyItem:
    """One question presented to the simulated respondent after the vignette."""
    item_id: str                 # short tag, e.g. "manip_context"
    section: str                 # "manipulation_check" / "dependent_variable" / "additional"
    label: str                   # human-readable label
    question: str                # Turkish question text
    options: dict[int, str]      # numeric code → Turkish label
    scale_type: str              # "categorical" or "likert"

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


# ---- §6 Manipülasyon Kontrolleri --------------------------------------------

ITEM_MANIP_CONTEXT = SurveyItem(
    item_id="manip_context",
    section="manipulation_check",
    label="6a Bağlam manipülasyonu kontrolü",
    # Revised 2026-06-30 (after V4 pilot showed the peace paragraph does not
    # mention Syria explicitly, causing the model to honestly answer
    # "Hatırlamıyorum" 100 % of the time on BE under V4's strict structural
    # separation). The question is now framed in terms of the general
    # Turkish-Kurdish peace climate so both context paragraphs (security and
    # peace) can answer it directly.
    question=(
        "Az önce okuduğunuz metne göre Türkiye'de Kürt-Türk ilişkilerinin ve "
        "barış sürecinin genel durumunu nasıl tanımlarsınız?"
    ),
    options={
        1: "Aktif gerilim ve silahlı çatışma ortamı var",
        2: "Barış süreci sürüyor, uzlaşma ortamı var",
        3: "Hatırlamıyorum",
    },
    scale_type="categorical",
)

ITEM_MANIP_FRAMING = SurveyItem(
    item_id="manip_framing",
    section="manipulation_check",
    label="6b Çerçeve manipülasyonu kontrolü",
    question=(
        "Az önce okuduğunuz yürüyüş metninde Kürtçe anadilde eğitim talebi "
        "en çok hangi gerekçeyle savunuluyordu?"
    ),
    options={
        1: "Ek bir gerekçe sunulmadan, doğrudan bir politika talebi olarak",
        2: "Eşit yurttaşlık, kültürel haklar ve demokratik kapsayıcılık temelinde",
        3: "Halkların kendi kaderini tayin hakkı temelinde",
        4: "Hatırlamıyorum",
    },
    scale_type="categorical",
)


# ---- §7 Bağımlı Değişkenler --------------------------------------------------

ITEM_DV_LEGITIMACY = SurveyItem(
    item_id="dv_legitimacy",
    section="dependent_variable",
    label="7a Hareket meşruiyeti",
    question="Bu grup meşru bir hak mücadelesi yürütüyor.",
    options=_LIKERT5,
    scale_type="likert",
)

ITEM_DV_THREAT = SurveyItem(
    item_id="dv_threat",
    section="dependent_variable",
    label="7b Tehdit algısı",
    question=(
        "Bu grubun talepleri toplumsal huzuru ve ulusal birliği tehdit edici "
        "nitelik taşıyor."
    ),
    options=_LIKERT5,
    scale_type="likert",
)

ITEM_DV_POLICY_SUPPORT = SurveyItem(
    item_id="dv_policy_support",
    section="dependent_variable",
    label="7c Politika desteği",
    question="Devlet okullarında Kürtçe anadilde eğitime yasal olarak izin verilmelidir.",
    options=_LIKERT5,
    scale_type="likert",
)

ITEM_DV_BEHAVIORAL_INTENT = SurveyItem(
    item_id="dv_behavioral_intent",
    section="dependent_variable",
    label="7d Hareket desteği (davranışsal eğilim)",
    question="Böyle bir yürüyüş düzenlendiğinde aşağıdakilerden hangisini yapardınız?",
    # 6-point ordinal spectrum from most-supportive (1) to most-opposed (6).
    # Kept as 'categorical' because it is not an agreement scale; analysts
    # who want an ordinal treatment can recode downstream.
    options={
        1: "Yürüyüşe bizzat katılırdım",
        2: "Sosyal medyada paylaşır, desteklerdim",
        3: "Destekler ama kamuoyu önünde belli etmezdim",
        4: "Karşı çıkar ama kamuoyu önünde belli etmezdim",
        5: "Sosyal medyada karşıt görüşlü paylaşım yapardım",
        6: "Karşıt görüşlü bir etkinliğe bizzat katılırdım",
    },
    scale_type="categorical",
)


# ---- §8 İlave Test Değişkenleri ---------------------------------------------

ITEM_ADD_SECURITY_PERCEPTION = SurveyItem(
    item_id="add_security_perception",
    section="additional",
    label="8a Genel güvenlik algısı",
    question="Şu anda Türkiye'nin güvenliği sizce ne kadar tehdit altında?",
    options={
        1: "Hiç tehdit altında değil",
        2: "Biraz tehdit altında",
        3: "Oldukça tehdit altında",
        4: "Çok ciddi tehdit altında",
    },
    scale_type="likert",   # 4-pt ordinal
)

ITEM_ADD_FRAMING_LEGITIMACY = SurveyItem(
    item_id="add_framing_legitimacy",
    section="additional",
    label="8b Çerçeve meşruiyeti",
    question=(
        "Az önce bahsi geçen grubun talebini sunma biçimini ne kadar meşru "
        "buluyorsunuz?"
    ),
    options={
        1: "Hiç meşru bulmuyorum",
        2: "Pek meşru bulmuyorum",
        3: "Kısmen meşru buluyorum",
        4: "Çok meşru buluyorum",
    },
    scale_type="likert",   # 4-pt ordinal
)


ADDITIONAL_ITEMS: list[SurveyItem] = [
    ITEM_ADD_SECURITY_PERCEPTION,
    ITEM_ADD_FRAMING_LEGITIMACY,
]


SURVEY_ITEMS: list[SurveyItem] = [
    # §6 manipulation checks
    ITEM_MANIP_CONTEXT,
    ITEM_MANIP_FRAMING,
    # §7 dependent variables
    ITEM_DV_LEGITIMACY,
    ITEM_DV_THREAT,
    ITEM_DV_POLICY_SUPPORT,
    ITEM_DV_BEHAVIORAL_INTENT,
    # §8 additional test variables
    *ADDITIONAL_ITEMS,
]


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    code: str
    factor_a: str
    factor_a_label: str
    factor_b: str
    factor_b_label: str
    vignette: str


_SECURITY_CONTEXT_PARA = (
    "Milli Savunma Bakanlığı, Suriye'nin kuzeyinde Kürt silahlı gruplarıyla "
    "sürdürülen çatışmalara ilişkin açıklama yaptı. Açıklamada, son haftalarda "
    "bölgede silahlı çatışmaların yoğunlaştığı ve Türk Silahlı Kuvvetleri'nin "
    "operasyonlarını sürdürdüğü bildirildi. Kayıpların yaşandığı ve bölgede "
    "güvenlik önlemlerinin artırıldığı aktarıldı."
)
_PEACE_CONTEXT_PARA = (
    "Bu yıl Nevroz kutlamaları farklı bir atmosferde geçti. Süregelen müzakere "
    "sürecinin yarattığı iyimserlik ortamında pek çok şehirde Kürt ve Türk "
    "vatandaşlar bir arada kutlama yaptı. Gözlemciler, kutlamaların bu yıl "
    "önceki yıllara kıyasla daha kapsayıcı ve şenlikli bir havada ilerlediğini "
    "aktardı."
)

GC_VIGNETTE = (
    _SECURITY_CONTEXT_PARA + "\n\n"
    "Aynı dönemde Ankara'da düzenlenen yürüyüşte bir grup, Kürtçe anadilde "
    "eğitim talebini dile getirdi. Katılımcıların pankartlar taşıdığı ve "
    "sloganlar attığı bildirildi."
)
GE_VIGNETTE = (
    _SECURITY_CONTEXT_PARA + "\n\n"
    "Aynı dönemde Ankara'da düzenlenen yürüyüşte bir grup, Kürtçe anadilde "
    "eğitim talebini dile getirdi. Katılımcılar, bu talebin eşit yurttaşlık, "
    "kültürel haklar ve demokratik kapsayıcılık açısından temel bir hak "
    "olduğunu vurguladı."
)
GK_VIGNETTE = (
    _SECURITY_CONTEXT_PARA + "\n\n"
    "Aynı dönemde Ankara'da düzenlenen yürüyüşte bir grup, Kürtçe anadilde "
    "eğitim talebini dile getirdi. Katılımcılar, "
    "\"Dil yasakları sömürge politikasıdır; halkların kendi kaderini tayin "
    "hakkı pazarlık konusu olamaz\" yazılı pankartlar taşıdı."
)
BC_VIGNETTE = (
    _PEACE_CONTEXT_PARA + "\n\n"
    "Bu süreçte Ankara'da düzenlenen yürüyüşte bir grup, Kürtçe anadilde "
    "eğitim talebini dile getirdi. Katılımcıların pankartlar taşıdığı ve "
    "sloganlar attığı bildirildi."
)
BE_VIGNETTE = (
    _PEACE_CONTEXT_PARA + "\n\n"
    "Bu süreçte Ankara'da düzenlenen yürüyüşte bir grup, Kürtçe anadilde "
    "eğitim talebini dile getirdi. Katılımcılar, bu talebin eşit yurttaşlık, "
    "kültürel haklar ve demokratik kapsayıcılık açısından temel bir hak "
    "olduğunu vurguladı."
)
BK_VIGNETTE = (
    _PEACE_CONTEXT_PARA + "\n\n"
    "Bu süreçte Ankara'da düzenlenen yürüyüşte bir grup, Kürtçe anadilde "
    "eğitim talebini dile getirdi. Katılımcılar, "
    "\"Dil yasakları sömürge politikasıdır; halkların kendi kaderini tayin "
    "hakkı pazarlık konusu olamaz\" yazılı pankartlar taşıdı."
)

# V1 conditions (Stages 1-4; preserved verbatim for replication).
CONDITIONS_V1: list[Condition] = [
    Condition("GC", "G", "security_context", "C", "no_framing",         GC_VIGNETTE),
    Condition("GE", "G", "security_context", "E", "universal_rights",   GE_VIGNETTE),
    Condition("GK", "G", "security_context", "K", "self_determination", GK_VIGNETTE),
    Condition("BC", "B", "peace_context",    "C", "no_framing",         BC_VIGNETTE),
    Condition("BE", "B", "peace_context",    "E", "universal_rights",   BE_VIGNETTE),
    Condition("BK", "B", "peace_context",    "K", "self_determination", BK_VIGNETTE),
]

# ---------------------------------------------------------------------------
# FRAMING TEXT — V2 (revised 2026-06-30)
# ---------------------------------------------------------------------------
# Substantially longer and more elaborated than V1. Both framings now anchor
# the protest as "barışçıl bir yürüyüş" and add explicit normative cues
# (UN human-rights norms for E; "asimilasyoncu politikalara karşı kolektif
# mücadele" and "ezilen bir halk" for K). The same V2 paragraph is reused
# under both security and peace contexts — context is set by the first
# paragraph only.
#
# C (no framing) is intentionally NOT defined in V2 — by the Stage-2
# manipulation-check finding, C was indistinguishable from E and has been
# dropped from the analysis design going forward.
# ---------------------------------------------------------------------------

_E_FRAMING_V2 = (
    "Ankara'da düzenlenen barışçıl bir yürüyüşte bir grup, Kürtçe ana dilde "
    "eğitim talebini dile getirdi. Katılımcılara göre Kürtçe ana dilde eğitim, "
    "eşit yurttaşlık, eğitim, ve kültürel haklar açısından temel bir insan "
    "hakkı olarak değerlendirilmelidir. Katılımcılar ayrıca bu talebin, "
    "Birleşmiş Milletler insan hakları normlarında yer alan eşitlik, ayrımcılık "
    "yasağı ve eğitim hakkı ilkeleriyle uyumlu olduğunu vurguladı. "
    "Açıklamalarda, ana dilde eğitim hakkının demokratik bir toplumda farklı "
    "kimliklerin eşit biçimde tanınması ve kamusal yaşama eşit katılımı "
    "açısından önemli olduğu ifade edildi."
)
_K_FRAMING_V2 = (
    "Ankara'da düzenlenen barışçıl bir yürüyüşte bir grup, Kürtçe ana dilde "
    "eğitim talebini dile getirdi. Katılımcılar, bu talebin Kürt halkının "
    "kendi kaderini tayin hakkının ve asimilasyoncu politikalara karşı "
    "verdiği kolektif mücadelenin parçası olduğunu vurguladı. Açıklamalarda, "
    "ana dilde eğitim hakkının ezilen bir halkın kendi dilini, kültürünü ve "
    "siyasal geleceğini özgürce belirleme hakkından ayrı düşünülemeyeceği "
    "ifade edildi. Katılımcılar, dil hakları mücadelesinin Kürt halkının "
    "özgürlük ve kendi kaderini tayin hakkı ile birlikte ele alınması "
    "gerektiğine dikkat çekti."
)

GE_VIGNETTE_V2 = _SECURITY_CONTEXT_PARA + "\n\n" + _E_FRAMING_V2
GK_VIGNETTE_V2 = _SECURITY_CONTEXT_PARA + "\n\n" + _K_FRAMING_V2
BE_VIGNETTE_V2 = _PEACE_CONTEXT_PARA    + "\n\n" + _E_FRAMING_V2
BK_VIGNETTE_V2 = _PEACE_CONTEXT_PARA    + "\n\n" + _K_FRAMING_V2

CONDITIONS_V2: list[Condition] = [
    Condition("GE", "G", "security_context", "E", "universal_rights",   GE_VIGNETTE_V2),
    Condition("GK", "G", "security_context", "K", "self_determination", GK_VIGNETTE_V2),
    Condition("BE", "B", "peace_context",    "E", "universal_rights",   BE_VIGNETTE_V2),
    Condition("BK", "B", "peace_context",    "K", "self_determination", BK_VIGNETTE_V2),
]

# ---------------------------------------------------------------------------
# V3 (revised 2026-06-30 after Stage 5).
# Two changes from V2:
#   (a) PEACE context paragraph rewritten to be longer and richer in
#       peace / brotherhood signal (Newroz "barış ve kardeşlik duygusu",
#       "birlikte yaşama iradesi", three sentences instead of two).
#       Security context paragraph is UNCHANGED.
#   (b) Each framing paragraph is prepended with the connector phrase
#       "Bu politik ortamda " to make the discourse link between context
#       and framing explicit. The framing text itself is V2 verbatim.
# Goal: rescue peace-context recognition (collapsed to 7-18% under V2)
# without weakening the strong V2 framing wording.
# ---------------------------------------------------------------------------

_PEACE_CONTEXT_PARA_V3 = (
    "Bu yıl Newroz kutlamaları, barış ve kardeşlik duygusunun daha güçlü "
    "hissedildiği, neşeli ve umutlu bir atmosferde geçti. Barış sürecinin "
    "yarattığı iyimserlikle birlikte, birçok şehirde Kürt ve Türk vatandaşlar "
    "kutlamalarda yan yana geldi. Meydanlarda oluşan kalabalıklar, Newroz'un "
    "birlikte yaşama iradesinin paylaşıldığı güçlü bir buluşmaya dönüştüğünü "
    "gösterdi. Gözlemciler de bu yılki kutlamaların önceki yıllara kıyasla "
    "daha canlı, daha renkli ve daha şenlikli bir atmosferde geçtiğini aktardı."
)

_CONNECTOR_V3 = "Bu politik ortamda "

_E_FRAMING_V3 = _CONNECTOR_V3 + _E_FRAMING_V2
_K_FRAMING_V3 = _CONNECTOR_V3 + _K_FRAMING_V2

# Security context is unchanged in V3; peace context is the V3 rewrite.
GE_VIGNETTE_V3 = _SECURITY_CONTEXT_PARA    + "\n\n" + _E_FRAMING_V3
GK_VIGNETTE_V3 = _SECURITY_CONTEXT_PARA    + "\n\n" + _K_FRAMING_V3
BE_VIGNETTE_V3 = _PEACE_CONTEXT_PARA_V3    + "\n\n" + _E_FRAMING_V3
BK_VIGNETTE_V3 = _PEACE_CONTEXT_PARA_V3    + "\n\n" + _K_FRAMING_V3

CONDITIONS_V3: list[Condition] = [
    Condition("GE", "G", "security_context", "E", "universal_rights",   GE_VIGNETTE_V3),
    Condition("GK", "G", "security_context", "K", "self_determination", GK_VIGNETTE_V3),
    Condition("BE", "B", "peace_context",    "E", "universal_rights",   BE_VIGNETTE_V3),
    Condition("BK", "B", "peace_context",    "K", "self_determination", BK_VIGNETTE_V3),
]

# ---------------------------------------------------------------------------
# V4 (revised 2026-06-30 after Stage 6).
# Goal: rescue peace-context recognition by structurally separating the
# two paragraphs so the model treats them as two distinct news items
# instead of one continuous text. Three changes from V3:
#   (a) A preamble sentence tells the model that what follows are two
#       short news items it should read carefully and base its answers on.
#   (b) Each paragraph gets a "HABER 1:" / "HABER 2:" label so the model
#       can treat them as separate items rather than running prose.
#   (c) "Bu politik ortamda" connector (added in V3) is REMOVED from the
#       framing paragraph — the explicit labels make the connector
#       redundant, and the V3 connector itself may have contributed to
#       the conflict-prior activation.
# Peace context paragraph uses the V3 stronger version
# (`_PEACE_CONTEXT_PARA_V3`). Framing text is the V2 wording verbatim
# (no "Bu politik ortamda" prefix).
# ---------------------------------------------------------------------------

_PREAMBLE_V4 = (
    "Aşağıdaki iki kısa haber metnini dikkatlice oku. Ardından soruları "
    "bu haber metinlerini dikkate alarak yanıtla."
)


def _wrap_v4(context_para: str, framing_para: str) -> str:
    return (
        f"{_PREAMBLE_V4}\n\n"
        f"HABER 1:\n{context_para}\n\n"
        f"HABER 2:\n{framing_para}"
    )


GE_VIGNETTE_V4 = _wrap_v4(_SECURITY_CONTEXT_PARA,    _E_FRAMING_V2)
GK_VIGNETTE_V4 = _wrap_v4(_SECURITY_CONTEXT_PARA,    _K_FRAMING_V2)
BE_VIGNETTE_V4 = _wrap_v4(_PEACE_CONTEXT_PARA_V3,    _E_FRAMING_V2)
BK_VIGNETTE_V4 = _wrap_v4(_PEACE_CONTEXT_PARA_V3,    _K_FRAMING_V2)

CONDITIONS_V4: list[Condition] = [
    Condition("GE", "G", "security_context", "E", "universal_rights",   GE_VIGNETTE_V4),
    Condition("GK", "G", "security_context", "K", "self_determination", GK_VIGNETTE_V4),
    Condition("BE", "B", "peace_context",    "E", "universal_rights",   BE_VIGNETTE_V4),
    Condition("BK", "B", "peace_context",    "K", "self_determination", BK_VIGNETTE_V4),
]

# Default exposed name — backwards-compatible with earlier callers.
CONDITIONS: list[Condition] = CONDITIONS_V1

FRAMES_BY_VERSION: dict[str, list[Condition]] = {
    "v1": CONDITIONS_V1,
    "v2": CONDITIONS_V2,
    "v3": CONDITIONS_V3,
    "v4": CONDITIONS_V4,
}


# ---------------------------------------------------------------------------
# Prompt construction (single-item per call)
# ---------------------------------------------------------------------------

SYSTEM_BEN = (
    "Aşağıda Türkiye'de yaşayan bir kişinin profili yer almaktadır. "
    "Bu kişinin yerine geçerek bir ankete cevap vereceksin.\n\n"
    "Sana iki kısa haber metni ve ardından bir anket sorusu verilecek. "
    "HABER 1 genel siyasi bağlamı anlatmaktadır. HABER 2 ise bu bağlam "
    "içinde Ankara'da gerçekleşen yürüyüşü ve yürüyüşte dile getirilen "
    "talebin nasıl gerekçelendirildiğini anlatmaktadır."
)

SYSTEM_SEN = (
    "Aşağıda Türkiye'de yaşayan bir kişinin profili yer almaktadır. "
    "Sen, aşağıda tanımlanan kişisin; onun bakış açısından bir ankete "
    "cevap vereceksin.\n\n"
    "Sana iki kısa haber metni ve ardından bir anket sorusu verilecek. "
    "HABER 1 genel siyasi bağlamı anlatmaktadır. HABER 2 ise bu bağlam "
    "içinde Ankara'da gerçekleşen yürüyüşü ve yürüyüşte dile getirilen "
    "talebin nasıl gerekçelendirildiğini anlatmaktadır."
)

VS_COT_SUFFIX = (
    "\n\nBu kişinin gerçek hayatta bu soruya nasıl cevap verebileceğini düşün. "
    "Aynı profile sahip insanlar arasında cevap çeşitliliği olabileceği için, "
    "tek bir kesin cevap vermek yerine her seçeneğin seçilme olasılığını "
    "tahmin et.\n\n"
    "Önce 1-2 cümlelik kısa bir analiz yap. Ardından her seçenek için "
    "olasılık dağılımı ver.\n\n"
    "Format:\n"
    "ANALİZ: <kısa analiz>\n"
    "DAĞILIM: {\"1\": 0.X, \"2\": 0.X, ...}\n\n"
    "Kurallar:\n"
    "* Tüm seçeneklere bir olasılık ata.\n"
    "* Olasılıkların toplamı 1.0 olmalı.\n"
    "* Açıklama kısmını kısa tut.\n"
    "* Nihai çıktı yalnızca ANALİZ ve DAĞILIM satırlarından oluşmalı."
)

DIRECT_FORMAT_SUFFIX = (
    "\n\nCevabını yalnızca verilen seçenekler arasından sayı olarak ver, "
    "açıklama ekleme."
)


def build_prompt(layer_a_text: str, condition: Condition, item: SurveyItem,
                 address_mode: str, sampling: str) -> list[dict]:
    """Construct system + user messages for one (persona × condition × item) call.

    Scale-aware question formatting (matches the original Turkish survey design):

    - Likert items: the question is presented as an agreement statement
      preceded by the survey-instrument's anchor phrase
      ``"Tüm bunları göz önünde bulundurarak aşağıdaki ifadeye ne ölçüde
        katılıyorsunuz?"``. This is the standard §7 wording from the
      experiment design document. Without it the model is given the
      Likert options without the explicit agreement framing.
    - Categorical items (manip checks, behavioral_intent): the question
      text is already a proper interrogative ("…nasıl tanımlarsınız?",
      "…hangisini yapardınız?") and needs no extra wrapper.
    """
    system = SYSTEM_BEN if address_mode == "ben" else SYSTEM_SEN
    if sampling == "vs_cot":
        system = system + VS_COT_SUFFIX
    else:
        system = system + DIRECT_FORMAT_SUFFIX

    options_str = "\n".join(f"{k} = {v}" for k, v in item.options.items())

    # Scale-aware question block.
    if item.scale_type == "likert":
        question_block = (
            "Soru: Tüm bunları göz önünde bulundurarak aşağıdaki ifadeye ne "
            "ölçüde katılıyorsunuz?\n\n"
            f"İfade: {item.question}"
        )
    else:
        question_block = f"Soru: {item.question}"

    user = (
        f"{layer_a_text}\n\n"
        f"---\n\n"
        f"{condition.vignette}\n\n"
        f"---\n\n"
        f"{question_block}\n\n"
        f"Seçenekler:\n{options_str}"
    )
    if sampling != "vs_cot":
        user += "\n\nCevap (sadece sayı):"

    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Output parsing — single-item DAĞILIM (identical to calibration runner).
# ---------------------------------------------------------------------------

def per_respondent_seed(respondent_id: str, condition_code: str,
                        item_id: str, base_seed: int) -> int:
    """Deterministic seed per (respondent, condition, item, base_seed)."""
    payload = f"{base_seed}|{respondent_id}|{condition_code}|{item_id}".encode("utf-8")
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
    """Return (predicted_value, normalized_distribution) from a VS-CoT response."""
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


# ---------------------------------------------------------------------------
# File naming + persona loading
# ---------------------------------------------------------------------------

def make_cell_filename(experiment_id: str, persona_config: str, model: str,
                       temperature: float, address_mode: str,
                       sampling: str, seed: int) -> str:
    alias = MODEL_ALIASES.get(model, model.replace("-", ""))
    tcode = _TEMP_LABELS.get(temperature, f"T{temperature}".replace(".", ""))
    return (f"{experiment_id}_{persona_config}_{alias}_{tcode}_"
            f"{address_mode}_{sampling}_s{seed}.jsonl")


def load_persona(config_id: str, respondent_id: str) -> str:
    return (LAYER_A_DIR / config_id / f"{respondent_id}.md").read_text(encoding="utf-8")


def list_respondents(config_id: str) -> list[str]:
    d = LAYER_A_DIR / config_id
    return sorted(p.stem for p in d.glob("TGSS_*.md"))


def preload_personas(config_id: str, respondent_ids: list[str]) -> dict[str, str]:
    """Read all persona Markdown files once into memory.

    With ~2,588 personas × 6 conditions × 8 items, the run would otherwise
    re-open the same file ~48 times per persona. Pre-loading is a few MB and
    eliminates repeated filesystem I/O on the hot path.
    """
    cache: dict[str, str] = {}
    for rid in respondent_ids:
        cache[rid] = load_persona(config_id, rid)
    return cache


# ---------------------------------------------------------------------------
# Async inference loop — one API call per (persona × condition × item)
# ---------------------------------------------------------------------------

async def _call_one(client, *, respondent_id: str, condition: Condition,
                    item: SurveyItem, layer_a_text: str,
                    model: str, temperature: float, address_mode: str,
                    sampling: str, base_seed: int,
                    max_retries: int = 5) -> dict:
    messages = build_prompt(layer_a_text, condition, item, address_mode, sampling)
    seed = per_respondent_seed(respondent_id, condition.code, item.item_id, base_seed)
    max_out = 250 if sampling == "vs_cot" else 10

    record_base = {
        "respondent_id":  respondent_id,
        "experiment_id":  None,                  # filled by caller
        "factor_a":       condition.factor_a,
        "factor_a_label": condition.factor_a_label,
        "factor_b":       condition.factor_b,
        "factor_b_label": condition.factor_b_label,
        "condition":      condition.code,
        "item_id":        item.item_id,
        "section":        item.section,
        "scale_type":     item.scale_type,
        "model":          model,
        "temperature":    temperature,
        "address_mode":   address_mode,
        "sampling":       sampling,
        "base_seed":      base_seed,
        "draw_seed":      seed,
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    backoff = 1
    raw, prompt_tokens, completion_tokens = "", 0, 0
    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = dict(
                model=model, messages=messages, temperature=temperature,
                seed=base_seed if temperature == 0 else None,
            )
            if model.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = max_out
            else:
                kwargs["max_tokens"] = max_out

            resp = await asyncio.wait_for(
                client.chat.completions.create(**kwargs), timeout=120.0
            )
            raw = resp.choices[0].message.content or ""
            if resp.usage:
                prompt_tokens = resp.usage.prompt_tokens
                completion_tokens = resp.usage.completion_tokens
            break
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < max_retries - 1:
                await asyncio.sleep(2 ** backoff); backoff += 1
                continue
            return {**record_base, "predicted_value": None, "dist": None,
                    "raw_response": f"API_ERROR: {err}",
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "parse_status": "api_error"}

    if sampling == "vs_cot":
        val, dist = parse_vs_cot(raw, item.valid_range, seed)
    else:
        val = parse_direct(raw, item.valid_range)
        dist = None

    return {**record_base,
            "predicted_value": val, "dist": dist, "raw_response": raw,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "parse_status": "ok" if val is not None else "parse_failure"}


async def run_cell(*, client, experiment_id: str, persona_config: str,
                   respondents: list[str], conditions: list[Condition],
                   items: list[SurveyItem],
                   model: str, temperature: float, address_mode: str,
                   sampling: str, base_seed: int, concurrency: int,
                   out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip (respondent, condition, item) triples that are already ok.
    done: set[tuple[str, str, str]] = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("parse_status") == "ok":
                    done.add((r["respondent_id"], r["condition"], r["item_id"]))
            except Exception:
                continue
    if done:
        print(f"  [{out_path.name}] resume: {len(done)} (rid, condition, item) triples already done")

    # Preload personas into memory once (124K calls × ~5KB read = ~600MB
    # filesystem traffic eliminated).
    print(f"  preloading {len(respondents)} persona files into memory...")
    persona_cache = preload_personas(persona_config, respondents)
    print(f"  ✓ {len(persona_cache)} personas cached")

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    # Single append-mode file handle reduces open/close churn under high concurrency.
    f_out = out_path.open("a", encoding="utf-8")

    async def worker(rid: str, cond: Condition, item: SurveyItem) -> None:
        if (rid, cond.code, item.item_id) in done:
            return
        async with sem:
            rec = await _call_one(
                client, respondent_id=rid, condition=cond, item=item,
                layer_a_text=persona_cache[rid], model=model,
                temperature=temperature, address_mode=address_mode,
                sampling=sampling, base_seed=base_seed,
            )
            rec["experiment_id"] = experiment_id
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            async with write_lock:
                f_out.write(line)
                f_out.flush()  # durable progress under crash

    # Randomize CONDITION order and ITEM order independently per respondent,
    # both seeded so the run is reproducible.
    tasks: list[asyncio.Task] = []
    for rid in respondents:
        cond_order = list(conditions)
        random.Random(per_respondent_seed(rid, "ORDER_COND", "ORDER", base_seed)).shuffle(cond_order)
        for cond in cond_order:
            item_order = list(items)
            random.Random(per_respondent_seed(rid, cond.code, "ORDER_ITEM", base_seed)).shuffle(item_order)
            for item in item_order:
                tasks.append(asyncio.create_task(worker(rid, cond, item)))

    total = len(tasks)
    print(f"  [{out_path.name}] scheduling {total} API calls "
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
                print(f"    progress: {completed}/{total}  "
                      f"({completed/total*100:.1f}%)  "
                      f"rate={rate:.1f}/s  "
                      f"ETA={int(remaining)//60}m{int(remaining)%60:02d}s")
    finally:
        f_out.close()


# ---------------------------------------------------------------------------
# CSV export — wide per section + tall long-format.
# ---------------------------------------------------------------------------

_META_COLS_WIDE = [
    "experiment_id", "respondent_id", "condition",
    "factor_a", "factor_a_label", "factor_b", "factor_b_label",
    "model", "temperature", "address_mode", "sampling", "base_seed",
]


def export_csvs(jsonl_path: Path, items: list[SurveyItem]) -> None:
    """Write per-section wide CSVs and one long-format CSV for analysis."""
    csv_dir = jsonl_path.parent / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for line in jsonl_path.open(encoding="utf-8"):
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    if not records:
        print(f"  no records in {jsonl_path}")
        return

    # Long-format CSV (one row per JSONL record).
    long_path = csv_dir / "all_items_long.csv"
    long_cols = _META_COLS_WIDE + ["item_id", "section", "scale_type",
                                   "predicted_value", "parse_status", "draw_seed"]
    with long_path.open("w", newline="", encoding="utf-8") as f:
        w = csv_lib.writer(f)
        w.writerow(long_cols)
        for r in records:
            meta = [r.get(c) for c in _META_COLS_WIDE]
            w.writerow(meta + [
                r.get("item_id"), r.get("section"), r.get("scale_type"),
                r.get("predicted_value"), r.get("parse_status"),
                r.get("draw_seed"),
            ])
    print(f"  ✓ {long_path.relative_to(ROOT)}  ({len(records)} rows)")

    # Wide-format CSVs (one per section). Pivot by (respondent × condition).
    by_section: dict[str, list[SurveyItem]] = {}
    for it in items:
        by_section.setdefault(it.section, []).append(it)

    for section, its in by_section.items():
        path = csv_dir / f"{section}.csv"
        wide_cols = list(_META_COLS_WIDE) + [f"{it.item_id}_pred" for it in its]
        # Build (rid, cond) → row dict
        groups: dict[tuple[str, str], dict] = {}
        for r in records:
            if r.get("section") != section:
                continue
            key = (r["respondent_id"], r["condition"])
            row = groups.setdefault(key, {c: r.get(c) for c in _META_COLS_WIDE})
            row[f"{r['item_id']}_pred"] = r.get("predicted_value")
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv_lib.writer(f)
            w.writerow(wide_cols)
            for key in sorted(groups):
                row = groups[key]
                w.writerow([row.get(c) for c in wide_cols])
        print(f"  ✓ {path.relative_to(ROOT)}  ({len(groups)} rows, {len(its)} items)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_conditions_arg(s: str | None, active: list[Condition] | None = None) -> list[Condition]:
    pool = active if active is not None else CONDITIONS
    if not s or s == "all":
        return list(pool)
    codes = {c.strip() for c in s.split(",") if c.strip()}
    chosen = [c for c in pool if c.code in codes]
    if not chosen:
        raise SystemExit(f"--conditions matched nothing; got {s!r}. "
                         f"Valid codes for this frame version: {[c.code for c in pool]}")
    return chosen


def parse_items_arg(s: str | None) -> list[SurveyItem]:
    if not s or s == "all":
        return list(SURVEY_ITEMS)
    ids = {x.strip() for x in s.split(",") if x.strip()}
    chosen = [it for it in SURVEY_ITEMS if it.item_id in ids]
    if not chosen:
        raise SystemExit(f"--items matched nothing; got {s!r}. "
                         f"Valid ids: {[it.item_id for it in SURVEY_ITEMS]}")
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description="Within-subject factorial experiment runner.")
    p.add_argument("--experiment-id", type=str, default="exp01_kurd_education")
    p.add_argument("--persona-config", type=str, default="C11")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--sampling", type=str, choices=["direct", "vs_cot"], default="vs_cot")
    p.add_argument("--address-mode", type=str, choices=["ben", "sen"], default="ben")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=str, choices=["v1", "v2", "v3", "v4"], default="v1",
                   help="Framing-text version. v1 = Stages 1-4 (6 cells incl. C); "
                        "v2 = revised framings (4 cells, GE/GK/BE/BK; longer and "
                        "more elaborated; C dropped); v3 = V2 framings with explicit "
                        "'Bu politik ortamda' connector + rewritten peace context "
                        "paragraph; v4 = preamble sentence + HABER 1/HABER 2 labels "
                        "structurally separating context and framing paragraphs "
                        "(connector dropped).")
    p.add_argument("--conditions", type=str, default="all",
                   help='"all" or comma-separated codes like "GE,BK"')
    p.add_argument("--items", type=str, default="all",
                   help='"all" or comma-separated item ids like "dv_legitimacy,dv_threat"')
    p.add_argument("--n-respondents", type=int, default=None,
                   help="If set, run only the first N respondents (sorted).")
    p.add_argument("--random-sample", type=int, default=None,
                   help="Pick N respondents at random (overrides --n-respondents).")
    p.add_argument("--sample-seed", type=int, default=42,
                   help="Seed for --random-sample (so the same N respondents reproduce).")
    p.add_argument("--concurrency", type=int, default=100,
                   help="Parallel API requests in flight. Tier 4 → 80-120, "
                        "Tier 5 → 200-400. Lower if you see 429 rate-limit errors.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--export-csvs", action="store_true",
                   help="Skip API calls; regenerate per-section CSVs from existing JSONL.")
    args = p.parse_args()

    active_frame_pool = FRAMES_BY_VERSION[args.frames]
    conditions = parse_conditions_arg(args.conditions, active=active_frame_pool)
    items = parse_items_arg(args.items)
    if not items:
        raise SystemExit("No items to run.")

    respondents = list_respondents(args.persona_config)
    if args.random_sample is not None:
        if args.random_sample > len(respondents):
            raise SystemExit(
                f"--random-sample {args.random_sample} > available "
                f"respondents ({len(respondents)})"
            )
        rng = random.Random(args.sample_seed)
        respondents = sorted(rng.sample(respondents, args.random_sample))
    elif args.n_respondents is not None:
        respondents = respondents[: args.n_respondents]

    out_path = EXPERIMENT_OUT / args.experiment_id / make_cell_filename(
        experiment_id=args.experiment_id,
        persona_config=args.persona_config,
        model=args.model,
        temperature=args.temperature,
        address_mode=args.address_mode,
        sampling=args.sampling,
        seed=args.seed,
    )

    n_calls = len(respondents) * len(conditions) * len(items)

    print("=== 12_run_experiment.py ===")
    print(f"  experiment_id : {args.experiment_id}")
    print(f"  frames        : {args.frames}  ({len(active_frame_pool)} cells available)")
    print(f"  persona       : {args.persona_config}  (n={len(respondents)})")
    print(f"  model         : {args.model}")
    print(f"  T             : {args.temperature}")
    print(f"  sampling      : {args.sampling}")
    print(f"  address       : {args.address_mode}")
    print(f"  base_seed     : {args.seed}")
    print(f"  conditions    : {[c.code for c in conditions]}  ({len(conditions)})")
    print(f"  items         : {len(items)}")
    for section in sorted({it.section for it in items}):
        sec_items = [it.item_id for it in items if it.section == section]
        print(f"    [{section:20s}]: {sec_items}")
    print(f"  output cell   : {out_path.relative_to(ROOT)}")
    print(f"  total API calls: {n_calls}")
    print(f"  concurrency   : {args.concurrency}")

    # --- Cost + time estimate (rough; assumes gpt-4o-mini pricing & VS-CoT token sizes) ---
    # Token assumptions: ~1000 prompt tokens (persona + vignette + question),
    # ~250 completion tokens (VS-CoT ANALİZ + DAĞILIM JSON).
    prompt_toks_per_call = 1000
    completion_toks_per_call = 250 if args.sampling == "vs_cot" else 10
    pricing = {
        "gpt-4o-mini":  (0.15 / 1_000_000, 0.60 / 1_000_000),     # ($/in_tok, $/out_tok)
        "gpt-5.4-mini": (0.25 / 1_000_000, 2.00 / 1_000_000),     # placeholder
        "gpt-5-mini":   (0.25 / 1_000_000, 2.00 / 1_000_000),     # placeholder
    }
    in_rate, out_rate = pricing.get(args.model, (None, None))
    if in_rate is not None:
        cost_in = n_calls * prompt_toks_per_call * in_rate
        cost_out = n_calls * completion_toks_per_call * out_rate
        print(f"  est cost      : ${cost_in + cost_out:.2f}  "
              f"(input ${cost_in:.2f} + output ${cost_out:.2f})")
    # Time estimate: assume ~2 sec/call end-to-end (network + model latency).
    sec_per_call = 2.0
    eff_calls_per_sec = args.concurrency / sec_per_call
    est_sec = n_calls / eff_calls_per_sec
    h, rem = divmod(int(est_sec), 3600)
    m, s = divmod(rem, 60)
    print(f"  est wall time : ~{h}h{m:02d}m  "
          f"(at {eff_calls_per_sec:.0f} calls/s with concurrency={args.concurrency})")

    bad_conds = [c.code for c in conditions if "TODO" in c.vignette]
    bad_items = [it.item_id for it in items if "TODO" in it.question]
    if bad_conds:
        print(f"\n⚠ vignette TODO for: {bad_conds}")
    if bad_items:
        print(f"⚠ item question TODO for: {bad_items}")

    if args.export_csvs:
        if not out_path.exists():
            raise SystemExit(f"No JSONL at {out_path} — nothing to export.")
        print(f"\nExporting CSVs from {out_path} ...")
        export_csvs(out_path, items)
        return

    if args.dry_run:
        print("\n--dry-run: not calling the API.")
        return

    if bad_conds or bad_items:
        raise SystemExit("Refusing to run: fix TODOs above first (or pass --dry-run).")

    if AsyncOpenAI is None:
        raise SystemExit("openai SDK not installed; run `pip install openai`.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        import getpass
        api_key = getpass.getpass("OpenAI API key: ").strip()
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY missing.")
    client = AsyncOpenAI(api_key=api_key)
    asyncio.run(run_cell(
        client=client, experiment_id=args.experiment_id,
        persona_config=args.persona_config, respondents=respondents,
        conditions=conditions, items=items, model=args.model,
        temperature=args.temperature, address_mode=args.address_mode,
        sampling=args.sampling, base_seed=args.seed,
        concurrency=args.concurrency, out_path=out_path,
    ))

    print(f"\n✓ Wrote {out_path}")
    print("Exporting per-section CSVs ...")
    export_csvs(out_path, items)


if __name__ == "__main__":
    main()
