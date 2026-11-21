"""Пороги чутливості та пресети номінацій.

Пороги застосовуються до вже обчислених метрик (див. analysis.py), тому
їх можна змінювати «наживо» без повторного аналізу фото.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("WLE_DATA_DIR", REPO_DIR / "wle_data"))
PRESETS_FILE = DATA_DIR / "presets.json"
YOLO_WEIGHTS = os.environ.get("WLE_YOLO_WEIGHTS", str(REPO_DIR / "yolov8n.pt"))

# Ширина мініатюри для аналізу. Пороги «таблички» залежать від розміру в пікселях,
# тому локальні фото теж зменшуються до цієї ширини — результати збігаються з Commons.
ANALYSIS_WIDTH = 800

# Версія шкали порогів. Збільшується, коли змінюється зміст метрик
# (v3: повернення до вихідних алгоритмів; пресети v2 ігноруються).
# Збережені пресети старішої версії ігноруються, щоб не застосувати старі числа до нових метрик.
THRESHOLDS_VERSION = 3

# YOLO викликається з налаштуваннями за замовчуванням (як у вихідному коді),
# тож детекції з упевненістю < 0.25 не зберігаються, і поріг не може бути нижчим.
PERSON_CONF_FLOOR = 0.25

HTTP_HEADERS = {"User-Agent": "WLE-Filter-Bot/3.0 (research; commons.wikimedia.org)"}


@dataclass
class Thresholds:
    """Значення за замовчуванням = константи вихідного main.py."""
    # BLUR_THRESHOLD: дисперсія Лапласіана; менше — розмитіше
    blur_enabled: bool = True
    blur_threshold: float = 200.0

    # PERSON_CONFIDENCE / PERSON_AREA_THRESHOLD
    person_enabled: bool = True
    person_confidence: float = 0.25
    person_area: float = 0.05

    # Таблички / написи
    sign_enabled: bool = True
    sign_method: str = "ocr"          # "ocr" — розпізнавання тексту; "legacy" — евристика main.py
    sign_text_area: float = 0.015     # ocr: мін. сумарна площа написів (багато табличок/вивісок)
    sign_object_area: float = 0.15    # ocr: мін. площа таблички разом з постаментом
    sign_coverage: float = 0.15       # legacy: SIGN_COVERAGE_MIN

    # Вотермарка: дрібний поодинокий напис біля краю кадру (лише OCR)
    watermark_enabled: bool = True

    # HASH_THRESHOLD / HASH_TYPE
    duplicate_enabled: bool = True
    duplicate_distance: int = 12
    hash_type: str = "dhash"          # "dhash" | "phash"
    duplicate_scope: str = "all"      # "all" — як у вихідному коді; "author" — у межах автора; "monument" — у межах пам'ятки (WLM)

    # Мінімальна роздільність оригіналу, мегапікселі (у вихідному коді не було; вимкнено)
    min_mpx_enabled: bool = False
    min_mpx: float = 2.0

    # ── Лише WLM (за замовчуванням вимкнено) ──
    # Чорний список категорій: підрядок (без урахування регістру), по одному в рядку
    blacklist_enabled: bool = False
    blacklist_categories: str = ""
    # Рік побудови пам'ятки новіший за межу — авторське право (немає свободи панорами)
    year_enabled: bool = False
    year_cutoff: int = 1954

    @classmethod
    def from_dict(cls, d: dict | None) -> "Thresholds":
        d = d or {}
        names = {f.name: f.type for f in fields(cls)}
        clean = {}
        defaults = cls()
        for k, v in d.items():
            if k not in names:
                continue
            ref = getattr(defaults, k)
            try:
                if isinstance(ref, bool):
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
                elif isinstance(ref, int):
                    v = int(float(v))
                elif isinstance(ref, float):
                    v = float(v)
                else:
                    v = str(v)
            except (TypeError, ValueError):
                continue
            clean[k] = v
        t = cls(**clean)
        if t.hash_type not in ("dhash", "phash"):
            t.hash_type = "dhash"
        if t.duplicate_scope not in ("all", "author", "monument"):
            t.duplicate_scope = "all"
        if t.sign_method not in ("ocr", "legacy"):
            t.sign_method = "ocr"
        t.person_confidence = max(t.person_confidence, PERSON_CONF_FLOOR)
        return t

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Пресети номінацій ───────────────────────────────────────────────────────
# Стартові значення; їх варто відкалібрувати на реальних даних через інтерфейс
# і зберегти кнопкою «Зберегти як пресет».
CAMPAIGNS = {
    "wle": {"label": "WLE", "title": "Wiki Loves Earth"},
    "wlm": {"label": "WLM", "title": "Wiki Loves Monuments"},
}
DEFAULT_CAMPAIGN = "wle"

WLM_BLACKLIST = "world war ii graves in ukraine\nbrotherhood graves in ukraine\nmass grave"

DEFAULT_PRESETS: dict[str, dict] = {
    "quality": {
        "campaign": "wle",
        "label": "Якісна номінація",
        "description": "Вихідні пороги main.py: blur 200, люди 0.25/5%, табличка 15%, dHash 12.",
        "category": "Wiki Loves Earth Ukraine 2026 - Quality",
        "thresholds": Thresholds().to_dict(),
    },
    "quantity": {
        "campaign": "wle",
        "label": "Кількісна номінація",
        "description": "Ті самі вихідні пороги — налаштуйте повзунками і збережіть у пресет.",
        "category": "Wiki Loves Earth Ukraine 2026 - Quantity",
        "thresholds": Thresholds().to_dict(),
    },
    # WLM: пороги зі скрипта WLM Ukraine 2021 (v3.4.8). Табличка — стара евристика з 45%,
    # вотермарка вимкнена (у тому скрипті її не було).
    "wlm": {
        "campaign": "wlm",
        "label": "Wiki Loves Monuments",
        "description": "Пороги зі скрипта WLM 2021: blur 60, люди 0,50 / 25%, табличка 45%, dHash 16; "
                       "чорний список категорій, рік > 1954, дублі в межах пам'ятки.",
        "category": "",
        "thresholds": Thresholds(
            blur_threshold=60.0, person_confidence=0.50, person_area=0.25,
            sign_method="legacy", sign_coverage=0.45, watermark_enabled=False,
            duplicate_distance=16, duplicate_scope="monument",
            blacklist_enabled=True, blacklist_categories=WLM_BLACKLIST,
            year_enabled=True, year_cutoff=1954).to_dict(),
    },
}


def load_presets() -> dict[str, dict]:
    presets = json.loads(json.dumps(DEFAULT_PRESETS))
    if PRESETS_FILE.exists():
        try:
            saved = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
            if saved.pop("_version", 1) < THRESHOLDS_VERSION:
                saved = {}
            for key, p in saved.items():
                base = presets.get(key, {"campaign": DEFAULT_CAMPAIGN, "label": key,
                                         "description": "", "category": ""})
                base.update({k: v for k, v in p.items() if k != "thresholds"})
                base["thresholds"] = Thresholds.from_dict(p.get("thresholds")).to_dict()
                presets[key] = base
        except (OSError, ValueError):
            pass
    return presets


def save_preset(key: str, thresholds: dict, label: str | None = None) -> dict:
    presets = load_presets()
    p = presets.get(key, {"campaign": DEFAULT_CAMPAIGN, "label": label or key,
                          "description": "", "category": ""})
    if label:
        p["label"] = label
    p["thresholds"] = Thresholds.from_dict(thresholds).to_dict()
    presets[key] = p
    _write_presets(presets)
    return presets


def _write_presets(presets: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_FILE.write_text(json.dumps({"_version": THRESHOLDS_VERSION, **presets},
                                       ensure_ascii=False, indent=2), encoding="utf-8")


def reset_preset(key: str) -> dict:
    presets = load_presets()
    if key in DEFAULT_PRESETS:
        presets[key] = json.loads(json.dumps(DEFAULT_PRESETS[key]))
    _write_presets(presets)
    return presets
