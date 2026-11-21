"""Застосування порогів чутливості до збережених метрик.

Правила — як у вихідному main.py: blur < BLUR_THRESHOLD; людина з conf ≥
PERSON_CONFIDENCE і площею ≥ PERSON_AREA_THRESHOLD; табличка coverage >
SIGN_COVERAGE_MIN; дублікат — перший збіг у реєстрі з відстанню ≤ HASH_THRESHOLD.

Функції тут чисті та швидкі: не читають файли й не викликають моделі,
тому їх можна викликати при кожному русі повзунка.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .config import Thresholds

REASON_LABELS = {
    "blur": "Розмите",
    "person": "Людина в кадрі",
    "sign": "Табличка / написи",
    "watermark": "Вотермарка",
    "duplicate": "Дублікат",
    "mpx": "Мала роздільність",
    "error": "Помилка",
}


def person_area(persons: list[dict] | None, min_conf: float) -> float:
    """Найбільша площа людини серед детекцій з впевненістю ≥ min_conf."""
    best = 0.0
    for p in persons or []:
        if p.get("conf", 0) >= min_conf:
            best = max(best, float(p.get("area", 0)))
    return best


def item_reasons(item: dict, t: Thresholds) -> list[dict]:
    """Причини відсіювання без урахування дублікатів."""
    reasons = []
    blur = item.get("blur")
    if t.blur_enabled and blur is not None and blur < t.blur_threshold:
        reasons.append({"code": "blur", "value": blur})

    if t.person_enabled:
        area = person_area(item.get("persons"), t.person_confidence)
        if area >= t.person_area and area > 0:
            reasons.append({"code": "person", "value": area})

    text_area = item.get("text_area")
    use_ocr = t.sign_method == "ocr" and text_area is not None
    if t.sign_enabled:
        if use_ocr:
            obj = item.get("sign_object") or 0
            if text_area >= t.sign_text_area and text_area > 0:
                reasons.append({"code": "sign", "value": text_area, "how": "ocr"})
            elif obj >= t.sign_object_area and text_area > 0:
                reasons.append({"code": "sign", "value": obj, "how": "object"})
        else:
            sign = item.get("sign")
            if sign is not None and sign > t.sign_coverage:   # coverage > SIGN_COVERAGE_MIN
                reasons.append({"code": "sign", "value": sign, "how": "legacy"})

    if t.watermark_enabled and item.get("watermark"):
        reasons.append({"code": "watermark", "value": item["watermark"]})

    if t.min_mpx_enabled:
        w, h = item.get("width") or 0, item.get("height") or 0
        if w and h:
            mpx = w * h / 1e6
            if mpx < t.min_mpx:
                reasons.append({"code": "mpx", "value": round(mpx, 2)})
    return reasons


def _hash_to_int(h: str | None) -> int | None:
    if not h:
        return None
    try:
        return int(h, 16)
    except ValueError:
        return None


def find_duplicates(items: list[dict], t: Thresholds) -> dict[int, tuple[int, int]]:
    """Знаходить дублікати.

    items мають бути впорядковані за часом (ранніші — «оригінали»).
    Як і в оригінальному скрипті, кожне фото порівнюється лише з уже
    зареєстрованими оригіналами; дублікат до реєстру не додається.
    Повертає {id_дубліката: (id_оригіналу, відстань)}.
    """
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for it in items:
        if it.get("status") != "done" or it.get("outdated"):
            continue
        hv = _hash_to_int(it.get(t.hash_type))
        if hv is None:
            continue
        key = (it.get("author") or "") if t.duplicate_scope == "author" else ""
        groups[key].append((it["id"], hv))

    result: dict[int, tuple[int, int]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        reg_ids = np.empty(len(members), dtype=np.int64)
        reg_hash = np.empty(len(members), dtype=np.uint64)
        n = 0
        for img_id, hv in members:
            if n:
                # як у вихідному check_duplicate: ПЕРШИЙ збіг у порядку реєстру
                dist = np.bitwise_count(reg_hash[:n] ^ np.uint64(hv))
                hits = np.nonzero(dist <= t.duplicate_distance)[0]
                if len(hits):
                    j = int(hits[0])
                    result[img_id] = (int(reg_ids[j]), int(dist[j]))
                    continue
            reg_ids[n] = img_id
            reg_hash[n] = hv
            n += 1
    return result


def evaluate(items: Iterable[dict], t: Thresholds, pairs=None) -> list[dict]:
    """Повертає для кожного фото: auto_verdict, verdict (з урахуванням ручного), reasons.

    items: словники з полями id, status, blur, persons, sign, dhash, phash,
    width, height, author, manual. Порядок = хронологічний.
    """
    items = list(items)
    dups = find_duplicates(items, t) if t.duplicate_enabled else {}
    out = []
    for it in items:
        if it.get("status") == "error":
            reasons = [{"code": "error", "value": it.get("error") or ""}]
            auto = "error"
        elif it.get("status") != "done" or it.get("outdated"):
            reasons, auto = [], "pending"
        else:
            reasons = item_reasons(it, t)
            if it["id"] in dups:
                orig, dist = dups[it["id"]]
                reasons.append({"code": "duplicate", "value": dist, "of": orig})
            auto = "flagged" if reasons else "ok"

        manual = it.get("manual")
        if manual == "accept":
            verdict = "ok"
        elif manual == "reject":
            verdict = "flagged"
        else:
            verdict = auto
        out.append({
            "id": it["id"],
            "auto_verdict": auto,
            "verdict": verdict,
            "manual": manual,
            "reasons": reasons,
            "person_area": round(person_area(it.get("persons"), t.person_confidence), 3),
        })
    return out


def reasons_text(reasons: list[dict]) -> str:
    """Текст причин у форматі старого progress.csv: 'blur:120.5 | person:12.0%'."""
    parts = []
    for r in reasons:
        c, v = r["code"], r.get("value")
        if c == "watermark":
            parts.append(f"watermark:{v}")
        elif c in ("person", "sign"):
            parts.append(f"{c}:{float(v):.1%}")
        elif c == "duplicate":
            parts.append(f"duplicate:similar(dist={v})")
        else:
            parts.append(f"{c}:{v}")
    return " | ".join(parts)
