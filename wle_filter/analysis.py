"""Обчислення «сирих» метрик фото.

Усе, що тут рахується, НЕ залежить від порогів чутливості — пороги
застосовуються пізніше у verdict.py. Тому прогін аналізу робиться один раз,
а повзунки в інтерфейсі просто перераховують вердикти.

Версія метрик (METRICS_VERSION) зберігається для кожного фото: якщо вона
застаріла, фото автоматично переаналізовуються з кешу (без повторного
завантаження з Commons).
"""
from __future__ import annotations

import threading

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps

from .config import ANALYSIS_WIDTH, PERSON_CONF_FLOOR, YOLO_WEIGHTS

METRICS_VERSION = 2


# ─── Підготовка зображення ───────────────────────────────────────────────────
def load_for_analysis(path: str) -> Image.Image:
    """Відкриває фото, враховує EXIF-орієнтацію і зменшує до ANALYSIS_WIDTH."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    if img.width > ANALYSIS_WIDTH:
        h = round(img.height * ANALYSIS_WIDTH / img.width)
        img = img.resize((ANALYSIS_WIDTH, h), Image.LANCZOS)
    return img


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _resize_max(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= side:
        return img
    s = side / max(h, w)
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


# ─── Різкість ────────────────────────────────────────────────────────────────
def blur_scores(img_bgr: np.ndarray) -> tuple[float, float]:
    """Різкість: (найрізкіша частина кадру, весь кадр).

    Кадр ділиться на 6×6 частин; береться 90-й перцентиль дисперсії Лапласіана
    по частинах. Так макро з розмитим фоном (боке) не вважається розмитим,
    якщо сам об'єкт різкий. Друге значення — класична метрика по всьому кадру.
    """
    gray = cv2.cvtColor(_resize_max(img_bgr, 600), cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = gray.shape
    th, tw = max(h // 6, 1), max(w // 6, 1)
    tiles = [lap[y:y + th, x:x + tw].var()
             for y in range(0, h - th + 1, th) for x in range(0, w - tw + 1, tw)]
    return round(float(np.percentile(tiles, 90)), 1), round(float(lap.var()), 1)


def blur_score(img_bgr: np.ndarray) -> float:
    """Сумісність зі старим API: різкість найрізкішої частини кадру."""
    return blur_scores(img_bgr)[0]


# ─── Люди ────────────────────────────────────────────────────────────────────
def person_detections(img_bgr: np.ndarray, yolo_model) -> list[dict]:
    """Усі детекції людей (клас 0) з впевненістю ≥ PERSON_CONF_FLOOR.

    Повертає [{conf, area, box:[x1,y1,x2,y2]}] з координатами в частках 0..1,
    щоб інтерфейс міг намалювати рамки, а поріг впевненості змінювати наживо.
    """
    h, w = img_bgr.shape[:2]
    res = yolo_model(img_bgr, verbose=False, conf=PERSON_CONF_FLOOR, classes=[0])[0]
    out = []
    for box in res.boxes:
        if int(box.cls[0]) != 0:
            continue
        conf = float(box.conf[0])
        if conf < PERSON_CONF_FLOOR:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        out.append({
            "conf": round(conf, 3),
            "area": round((x2 - x1) * (y2 - y1) / (h * w), 4),
            "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
        })
    out.sort(key=lambda d: -d["conf"])
    return out


# ─── Таблички / текст ────────────────────────────────────────────────────────
def text_regions(img_bgr: np.ndarray, ocr) -> dict:
    """Розпізнає текст (RapidOCR) і оцінює площу таблички.

    sign  — частка кадру, яку займає прямокутник навколо всього впевнено
            розпізнаного тексту (≈ площа таблички з написом);
    text  — розпізнаний текст (для підказки в інтерфейсі);
    boxes — рамки тексту в частках 0..1.
    Текстура трави, гілок чи відблиски на воді тут не рахуються: потрібен
    саме розпізнаний текст з упевненістю ≥ 0.6 і довжиною ≥ 3 символи.
    """
    h, w = img_bgr.shape[:2]
    res, _ = ocr(img_bgr)
    found = []
    for pts, txt, score in res or []:
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        txt = (txt or "").strip()
        if score >= 0.6 and sum(ch.isalnum() for ch in txt) >= 3:
            found.append((np.array(pts, dtype=np.float32), txt))
    if not found:
        return {"sign": 0.0, "text": "", "text_boxes": []}
    allp = np.concatenate([p for p, _ in found])
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    boxes = []
    for p, _ in found:
        a0, b0 = p.min(0)
        a1, b1 = p.max(0)
        boxes.append([round(float(a0) / w, 4), round(float(b0) / h, 4),
                      round(float(a1) / w, 4), round(float(b1) / h, 4)])
    return {
        "sign": round(float((x1 - x0) * (y1 - y0) / (w * h)), 4),
        "text": " ".join(t for _, t in found)[:120],
        "text_boxes": boxes[:30],
    }


def sign_coverage_heuristic(img_bgr: np.ndarray) -> float:
    """Запасний варіант без OCR (стара евристика v2, дає хибні спрацювання)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    dilated = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3)), iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_area = 0
    for c in cnts:
        _, _, cw, ch = cv2.boundingRect(c)
        if 2 < cw / max(ch, 1.0) < 60 and 8 < ch < h * 0.2 and cw * ch > 200:
            text_area += cw * ch
    return round(min(text_area / (h * w), 1.0), 4)


sign_coverage = sign_coverage_heuristic  # сумісність


# ─── Горизонт ────────────────────────────────────────────────────────────────
def horizon_tilt(img_bgr: np.ndarray, max_angle: float = 10.0, step: float = 0.25) -> dict:
    """Шукає найвиразнішу пряму лінію через увесь кадр (горизонт, урізу води)
    і повертає її нахил.

    Метод: беруться лише «горизонтальні» перепади яскравості (Собель по Y),
    кадр повертається з кроком 0.25° в межах ±max_angle, і для кожного кута
    шукається рядок з найбільшою узгодженою сумою перепадів. Кут, за якого
    лінія найчіткіша, — це нахил горизонту.

    angle      — нахил у градусах (+ — праворуч вниз);
    confidence — наскільки лінія виділяється (пік/медіана × частка ширини,
                 де лінія присутня). <1.5 — горизонту, найімовірніше, немає
                 (макро, ліс), ≥2 — чіткий горизонт;
    row        — положення лінії по висоті кадру (0 — верх, 1 — низ).
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    s = 400 / g.shape[1]
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    e = np.where(np.abs(gy) > 1.5 * np.abs(gx), gy, 0).astype(np.float32)
    h, w = e.shape
    c = (w / 2, h / 2)
    ones = np.ones_like(e)
    best_val, best_a, best_row = 0.0, 0.0, 0
    peaks = []
    for a in np.arange(-max_angle, max_angle + step / 2, step):
        M = cv2.getRotationMatrix2D(c, float(a), 1.0)
        r = cv2.warpAffine(e, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpAffine(ones, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0).sum(axis=1)
        rows = np.abs(r.sum(axis=1)) / np.maximum(mask, 1)
        rows[mask < w * 0.5] = 0
        rows[:int(h * 0.06)] = 0
        rows[int(h * 0.94):] = 0
        i = int(rows.argmax())
        peaks.append(rows[i])
        if rows[i] > best_val:
            best_val, best_a, best_row = float(rows[i]), float(a), i
    if best_val <= 0:
        return {"horizon_angle": 0.0, "horizon_conf": 0.0, "horizon_row": 0.5}
    contrast = best_val / (float(np.median(peaks)) + 1e-6)
    M = cv2.getRotationMatrix2D(c, best_a, 1.0)
    r = cv2.warpAffine(e, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    band = r[max(best_row - 1, 0):best_row + 2]
    sign = np.sign(band.sum()) or 1.0
    thr = np.abs(band).sum(axis=0).mean() * 0.25
    coverage = float(np.mean([seg.mean() > thr for seg in np.array_split(band.sum(axis=0) * sign, 10)]))
    return {
        "horizon_angle": round(best_a, 2),         # кут лінії: + праворуч вниз, − праворуч вгору
        "horizon_conf": round(contrast * coverage, 2),
        "horizon_row": round(best_row / h, 3),
    }


# ─── Експозиція та кольори ───────────────────────────────────────────────────
def exposure_color(img_bgr: np.ndarray) -> dict:
    """shadows    — частка майже чорних пікселів (V < 12%): глибока тінь;
    highlights — частка пересвічених пікселів (V > 98%);
    sat_clip   — частка помітних (V ≥ 25%) пікселів з насиченістю «в стелю»
                 (S ≥ 94%) після легкого згладжування, щоб JPEG-шум не впливав:
                 ознака перекручених кольорів. Експериментально: яскраві квіти
                 й зелень теж можуть мати високе значення."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    hs = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (0, 0), 1.0), cv2.COLOR_BGR2HSV)
    return {
        "shadows": round(float(np.mean(v < 31)), 4),
        "highlights": round(float(np.mean(v > 250)), 4),
        "sat_clip": round(float(np.mean((hs[..., 1] >= 240) & (hs[..., 2] >= 64))), 4),
    }


# ─── Дублікати ───────────────────────────────────────────────────────────────
def image_hashes(pil_img: Image.Image) -> dict[str, str]:
    return {"dhash": str(imagehash.dhash(pil_img)), "phash": str(imagehash.phash(pil_img))}


_sift = None


def sift_features(img_bgr: np.ndarray, n: int = 1000):
    """Ключові точки SIFT для перевірки «той самий кадр/сцена» (RootSIFT, uint8)."""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(n)
    g = cv2.cvtColor(_resize_max(img_bgr, 800), cv2.COLOR_BGR2GRAY)
    kp, desc = _sift.detectAndCompute(g, None)
    if desc is None or len(kp) < 8:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    pts = np.float32([k.pt for k in kp])
    return pts, desc.astype(np.uint8)


_bf = None


def match_inliers(a: tuple, b: tuple) -> int:
    """Кількість геометрично узгоджених спільних точок двох фото (RANSAC).
    Для різних сцен — зазвичай < 15, для того самого кадру з іншим
    кадруванням — десятки й сотні."""
    global _bf
    pa, da = a
    pb, db = b
    if len(da) < 8 or len(db) < 8:
        return 0
    if _bf is None:
        _bf = cv2.BFMatcher(cv2.NORM_L2)
    m = _bf.knnMatch(da.astype(np.float32), db.astype(np.float32), k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < 8:
        return len(good)
    src = np.float32([pa[x.queryIdx] for x in good])
    dst = np.float32([pb[x.trainIdx] for x in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if mask is None:
        _, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 3.0)
    return int(mask.sum()) if mask is not None else 0


def embedding(img_bgr: np.ndarray, yolo_model) -> np.ndarray | None:
    """Глобальний опис кадру (256 чисел з YOLO) — для швидкого пошуку
    кандидатів у дублікати перед точною перевіркою SIFT."""
    try:
        v = yolo_model.embed(img_bgr, verbose=False)[0].cpu().numpy().astype(np.float32)
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float16) if n else None
    except Exception:  # noqa: BLE001 — старі версії ultralytics без embed()
        return None


# ─── Усе разом ───────────────────────────────────────────────────────────────
def analyze(pil_img: Image.Image, yolo_model, ocr=None) -> dict:
    """Повний набір метрик для одного фото.

    Ключі з підкресленням (_sift, _embed) — службові: їх зберігає worker
    окремо (файл кешу / BLOB), у вердиктах вони не беруть участі.
    """
    bgr = pil_to_bgr(pil_img)
    hashes = image_hashes(pil_img)
    sharp, sharp_global = blur_scores(bgr)
    if ocr is not None:
        text = text_regions(bgr, ocr)
    else:
        text = {"sign": sign_coverage_heuristic(bgr), "text": "", "text_boxes": []}
    m = {
        "blur": sharp,
        "blur_global": sharp_global,
        "persons": person_detections(bgr, yolo_model) if yolo_model is not None else [],
        **text,
        "dhash": hashes["dhash"],
        "phash": hashes["phash"],
        **horizon_tilt(bgr),
        **exposure_color(bgr),
        "_sift": sift_features(bgr),
        "_embed": embedding(bgr, yolo_model) if yolo_model is not None else None,
    }
    return m


# ─── Моделі: одна на процес ──────────────────────────────────────────────────
_model = None
_ocr = None
_model_lock = threading.Lock()
infer_lock = threading.Lock()


def get_yolo():
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            _model = YOLO(YOLO_WEIGHTS)
        return _model


def get_ocr():
    """RapidOCR (Apache-2.0, моделі вбудовані в пакет). None, якщо не встановлено."""
    global _ocr
    with _model_lock:
        if _ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                _ocr = RapidOCR()
            except ImportError:
                _ocr = False
        return _ocr or None
