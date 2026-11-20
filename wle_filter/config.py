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

# Мінімальна впевненість, з якою YOLO-детекції зберігаються в базі.
# Поріг «впевненість людини» в інтерфейсі не може бути нижчим за це значення.
PERSON_CONF_FLOOR = 0.05

HTTP_HEADERS = {"User-Agent": "WLE-Filter-Bot/3.0 (research; commons.wikimedia.org)"}


@dataclass
class Thresholds:
    # Розмитість: дисперсія Лапласіана; менше — розмитіше
    blur_enabled: bool = True
    blur_threshold: float = 200.0

    # Люди: YOLO клас 0 з впевненістю ≥ person_confidence і площею ≥ person_area
    person_enabled: bool = True
    person_confidence: float = 0.25
    person_area: float = 0.05

    # Таблички / текст: частка площі, вкрита текстоподібними блоками
    sign_enabled: bool = True
    sign_coverage: float = 0.15

    # Дублікати: відстань Геммінга між перцептивними хешами
    duplicate_enabled: bool = True
    duplicate_distance: int = 12
    hash_type: str = "dhash"          # "dhash" | "phash"
    duplicate_scope: str = "all"      # "all" — серед усіх фото, "author" — лише в межах автора

    # Мінімальна роздільність оригіналу, мегапікселі
    min_mpx_enabled: bool = False
    min_mpx: float = 2.0

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
        if t.duplicate_scope not in ("all", "author"):
            t.duplicate_scope = "all"
        t.person_confidence = max(t.person_confidence, PERSON_CONF_FLOOR)
        return t

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Пресети номінацій ───────────────────────────────────────────────────────
# Стартові значення; їх варто відкалібрувати на реальних даних через інтерфейс
# і зберегти кнопкою «Зберегти як пресет».
DEFAULT_PRESETS: dict[str, dict] = {
    "quality": {
        "label": "Якісна номінація",
        "description": "Суворіше до різкості, людей і табличок у кадрі.",
        "category": "Wiki Loves Earth Ukraine 2026 - Quality",
        "thresholds": Thresholds().to_dict(),
    },
    "quantity": {
        "label": "Кількісна номінація",
        "description": "Головне — відсіяти повтори одного кадру одним автором; "
                       "до технічної якості м'якше.",
        "category": "",
        "thresholds": Thresholds(
            blur_threshold=100.0,
            person_area=0.15,
            sign_coverage=0.25,
            duplicate_distance=10,
            duplicate_scope="author",
        ).to_dict(),
    },
}


def load_presets() -> dict[str, dict]:
    presets = json.loads(json.dumps(DEFAULT_PRESETS))
    if PRESETS_FILE.exists():
        try:
            saved = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
            for key, p in saved.items():
                base = presets.get(key, {"label": key, "description": "", "category": ""})
                base.update({k: v for k, v in p.items() if k != "thresholds"})
                base["thresholds"] = Thresholds.from_dict(p.get("thresholds")).to_dict()
                presets[key] = base
        except (OSError, ValueError):
            pass
    return presets


def save_preset(key: str, thresholds: dict, label: str | None = None) -> dict:
    presets = load_presets()
    p = presets.get(key, {"label": label or key, "description": "", "category": ""})
    if label:
        p["label"] = label
    p["thresholds"] = Thresholds.from_dict(thresholds).to_dict()
    presets[key] = p
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_FILE.write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")
    return presets


def reset_preset(key: str) -> dict:
    presets = load_presets()
    if key in DEFAULT_PRESETS:
        presets[key] = json.loads(json.dumps(DEFAULT_PRESETS[key]))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_FILE.write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")
    return presets
