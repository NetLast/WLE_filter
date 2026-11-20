"""Застосування порогів чутливості до збережених метрик.

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
    "sign": "Табличка / текст",
    "horizon": "Завалений горизонт",
    "shadows": "Глибокі тіні",
    "highlights": "Пересвіт",
    "saturation": "Перенасичені кольори",
    "duplicate": "Дублікат",
    "mpx": "Мала роздільність",
    "error": "Помилка",
}

# Практично ідентичні кадри (pHash) вважаються дублікатами завжди,
# навіть якщо на фото замало ключових точок для SIFT (туман, вода, небо)
EXACT_HASH_DISTANCE = 4


def person_area(persons: list[dict] | None, min_conf: float) -> float:
    """Найбільша площа людини серед детекцій з впевненістю ≥ min_conf."""
    best = 0.0
    for p in persons or []:
        if p.get("conf", 0) >= min_conf:
            best = max(best, float(p.get("area", 0)))
    return round(best, 4)


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

    sign = item.get("sign")
    if t.sign_enabled and sign is not None and sign >= t.sign_coverage:
        reasons.append({"code": "sign", "value": sign})

    if t.horizon_enabled and item.get("horizon_angle") is not None:
        ang, conf = float(item["horizon_angle"]), float(item.get("horizon_conf") or 0)
        if conf >= t.horizon_conf and abs(ang) >= t.horizon_angle:
            reasons.append({"code": "horizon", "value": round(ang, 1)})

    sh = item.get("shadows")
    if t.shadows_enabled and sh is not None and sh >= t.shadows_max:
        reasons.append({"code": "shadows", "value": sh})
    hi = item.get("highlights")
    if t.highlights_enabled and hi is not None and hi >= t.highlights_max:
        reasons.append({"code": "highlights", "value": hi})
    sc = item.get("sat_clip")
    if t.saturation_enabled and sc is not None and sc >= t.saturation_clip:
        reasons.append({"code": "saturation", "value": sc})

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


def find_duplicates(items: list[dict], t: Thresholds,
                    pairs: list[tuple[int, int, int]] | None = None) -> dict[int, tuple[int, int, str]]:
    """Знаходить дублікати. items — у хронологічному порядку (ранні — «оригінали»).

    method="features": пари зі спільними ключовими точками (SIFT + RANSAC),
        порахованими під час аналізу; дублікат — пізніше фото пари з
        кількістю точок ≥ duplicate_inliers. Ловить той самий кадр з іншим
        кадруванням, експозицією, поворотом чи орієнтацією.
    method="hash": як у v2 — відстань Геммінга між хешами ≤ duplicate_distance.
    В обох режимах майже ідентичні кадри (pHash ≤ 4) — дублікати завжди.

    Повертає {id_дубліката: (id_оригіналу, значення, метод)}.
    """
    done = {it["id"]: it for it in items if it.get("status") == "done" and not it.get("outdated")}
    order = {it["id"]: i for i, it in enumerate(items)}

    def same_scope(a: dict, b: dict) -> bool:
        return t.duplicate_scope != "author" or ((a.get("author") or "") == (b.get("author") or ""))

    result: dict[int, tuple[int, int, str]] = {}
    if t.duplicate_method == "features" and pairs:
        for a, b, n in pairs:
            if n < t.duplicate_inliers or a not in done or b not in done:
                continue
            if order[a] > order[b]:
                a, b = b, a
            if not same_scope(done[a], done[b]):
                continue
            if b not in result or result[b][1] < n:
                result[b] = (a, int(n), "points")

    # Хеші: режим «hash» або завжди — практично ідентичні кадри
    limit = t.duplicate_distance if t.duplicate_method == "hash" else EXACT_HASH_DISTANCE
    htype = t.hash_type if t.duplicate_method == "hash" else "phash"
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for it in items:
        if it["id"] not in done:
            continue
        hv = _hash_to_int(it.get(htype))
        if hv is None:
            continue
        key = (it.get("author") or "") if t.duplicate_scope == "author" else ""
        groups[key].append((it["id"], hv))
    for members in groups.values():
        if len(members) < 2:
            continue
        reg_ids = np.empty(len(members), dtype=np.int64)
        reg_hash = np.empty(len(members), dtype=np.uint64)
        n = 0
        for img_id, hv in members:
            if n:
                dist = np.bitwise_count(reg_hash[:n] ^ np.uint64(hv))
                j = int(np.argmin(dist))
                if int(dist[j]) <= limit:
                    if img_id not in result:
                        result[img_id] = (int(reg_ids[j]), int(dist[j]), "hash")
                    continue
            reg_ids[n] = img_id
            reg_hash[n] = hv
            n += 1
    return result


def evaluate(items: Iterable[dict], t: Thresholds,
             pairs: list[tuple[int, int, int]] | None = None) -> list[dict]:
    """Повертає для кожного фото: auto_verdict, verdict (з урахуванням ручного), reasons.

    items: словники з полями id, status, blur, persons, sign, dhash, phash,
    width, height, author, manual. Порядок = хронологічний.
    """
    items = list(items)
    dups = find_duplicates(items, t, pairs) if t.duplicate_enabled else {}
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
                orig, val, how = dups[it["id"]]
                reasons.append({"code": "duplicate", "value": val, "of": orig, "how": how})
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
            "person_area": person_area(it.get("persons"), t.person_confidence),
        })
    return out


def reasons_text(reasons: list[dict]) -> str:
    """Текст причин у форматі старого progress.csv: 'blur:120.5 | person:12.0%'."""
    parts = []
    for r in reasons:
        c, v = r["code"], r.get("value")
        if c in ("person", "sign", "shadows", "highlights", "saturation"):
            parts.append(f"{c}:{float(v):.1%}")
        elif c == "horizon":
            parts.append(f"horizon:{v}°")
        elif c == "duplicate":
            parts.append(f"duplicate:points={v}" if r.get("how") == "points" else f"duplicate:similar(dist={v})")
        else:
            parts.append(f"{c}:{v}")
    return " | ".join(parts)
