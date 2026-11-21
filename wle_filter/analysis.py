"""Обчислення метрик фото — алгоритми ВИХІДНОГО скрипта (main.py з першого коміту).

Функції check_blur / check_person / check_sign_or_plaque / check_duplicate
відтворено без змін логіки; різниця лише в тому, що метрики зберігаються,
а пороги застосовуються окремо (verdict.py), щоб повзунки в інтерфейсі
працювали без повторного аналізу.
"""
from __future__ import annotations

import threading

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps

from .config import ANALYSIS_WIDTH, YOLO_WEIGHTS

# 3 — повернення до вихідних алгоритмів; 4 — додано OCR-написи та вотермарки
METRICS_VERSION = 4


# ─── Підготовка зображення ───────────────────────────────────────────────────
def load_for_analysis(path: str) -> Image.Image:
    """Як у вихідному скрипті: Image.open(...).convert("RGB").

    Мініатюри Commons уже мають ширину 800 px. Для локальних фото
    додатково враховується EXIF-орієнтація і зменшення до 800 px,
    щоб пороги працювали так само, як для Commons.
    """
    img = Image.open(path)
    if img.width > ANALYSIS_WIDTH or img.getexif().get(0x0112, 1) != 1:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        if img.width > ANALYSIS_WIDTH:
            h = round(img.height * ANALYSIS_WIDTH / img.width)
            img = img.resize((ANALYSIS_WIDTH, h), Image.LANCZOS)
        return img
    return img.convert("RGB")


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ─── Фільтри (вихідний код) ──────────────────────────────────────────────────
def blur_score(img_array: np.ndarray) -> float:
    """check_blur: дисперсія Лапласіана на зображенні ≤600 px."""
    h, w = img_array.shape[:2]
    if max(h, w) > 600:
        scale = 600 / max(h, w)
        img_array = cv2.resize(img_array, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return round(score, 1)


def person_detections(img_array: np.ndarray, yolo_model) -> list[dict]:
    """check_person: YOLO з налаштуваннями за замовчуванням (conf 0.25).

    Зберігаються всі детекції класу 0, щоб у перегляді фото малювати рамки;
    поріг впевненості й площі застосовується у verdict.py так само,
    як у вихідному коді: conf ≥ PERSON_CONFIDENCE і площа ≥ PERSON_AREA_THRESHOLD.
    """
    h, w = img_array.shape[:2]
    res = yolo_model(img_array, verbose=False)[0]
    out = []
    for box in res.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append({
                "conf": round(float(box.conf[0]), 3),
                "area": (x2 - x1) * (y2 - y1) / (h * w),
                "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
            })
    out.sort(key=lambda d: -d["conf"])
    return out


def sign_coverage(img_array: np.ndarray) -> float:
    """check_sign_or_plaque: частка кадру з текстоподібними блоками."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    dilated = cv2.dilate(bw, kernel, iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    text_area = 0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(ch, 1.0)
        area = cw * ch
        if 2 < aspect < 60 and 8 < ch < h * 0.2 and area > 200:
            text_area += area
    coverage = text_area / (h * w)
    return round(coverage, 3)


# ─── Таблички / написи через OCR (новий метод) ───────────────────────────────
# Рамки беруться з детектора тексту (стабільний), а розпізнавання — лише як м'яка
# перевірка «це текст, а не текстура»: модель не знає кирилиці, тож оцінки для
# українських написів низькі (0.3–0.8), а текстури дають порожній рядок з 0.
OCR_MIN_SCORE = 0.3
OCR_MIN_CHARS = 2
EDGE = 0.05               # смуга біля краю кадру (частка ширини/висоти)
WM_MAX_HEIGHT = 0.03      # вотермарка — дрібний рядок (висота < 3% кадру)
PAD = 0.15                # поля навколо групи рядків — оцінка площі таблички


def _valid_line(txt: str, score) -> bool:
    try:
        score = float(score)
    except (TypeError, ValueError):
        return False
    alnum = [c for c in (txt or "") if c.isalnum()]
    if score < OCR_MIN_SCORE or len(alnum) < OCR_MIN_CHARS:
        return False
    # «1111», «JUUU», «II» — візерунки (огорожа, вікна), а не текст
    top = max(alnum.count(c) for c in set(alnum))
    if len(alnum) <= 3:
        return top < len(alnum)
    return top / len(alnum) < 0.6


def _clusters(boxes: list[list[float]]) -> list[list[int]]:
    """Групує рядки, що стоять поруч (одна табличка/вивіска)."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            h = max(a[3] - a[1], b[3] - b[1])
            if (a[0] - h <= b[2] and b[0] - h <= a[2] and
                    a[1] - 1.5 * h <= b[3] and b[1] - 1.5 * h <= a[3]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def sign_object(img_array: np.ndarray, box: list[float]) -> tuple[float, list[float]]:
    """Оцінка площі предмета з написом (табличка + постамент/стенд/камінь).

    GrabCut: напис з найближчим оточенням — точно предмет; широка область
    навколо (особливо вниз, де зазвичай постамент) — можливо предмет; решта
    кадру — фон. Береться зв'язна область, що містить напис. Це наближена
    оцінка: може «прихопити» сусідній предмет схожого кольору (стовбур дерева,
    стіну) — тому в перегляді фото вона показана червоною рамкою.
    Рахується на копії ≤ 400 px — швидко й достатньо для оцінки площі.
    """
    small = img_array
    if max(img_array.shape[:2]) > 400:
        sc = 400 / max(img_array.shape[:2])
        small = cv2.resize(img_array, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    H, W = small.shape[:2]
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    cw, ch = max(x1 - x0, 2), max(y1 - y0, 2)
    rx0, rx1 = max(0, x0 - 1.5 * cw), min(W - 1, x1 + 1.5 * cw)
    ry0, ry1 = max(0, y0 - 2 * ch), min(H - 1, y1 + 6 * ch)
    fx0, fx1 = max(0, x0 - 0.15 * cw), min(W - 1, x1 + 0.15 * cw)
    fy0, fy1 = max(0, y0 - 0.5 * ch), min(H - 1, y1 + 0.5 * ch)
    fg_box = [round(fx0 / W, 4), round(fy0 / H, 4), round(fx1 / W, 4), round(fy1 / H, 4)]
    region = small[int(ry0):int(ry1) + 1, int(rx0):int(rx1) + 1]
    if region.size == 0 or float(region.std()) < 4:
        # однорідна ділянка — межу предмета не визначити; беремо саму табличку
        return (fx1 - fx0) * (fy1 - fy0) / (W * H), fg_box
    mask = np.full((H, W), cv2.GC_BGD, np.uint8)
    mask[int(ry0):int(ry1), int(rx0):int(rx1)] = cv2.GC_PR_BGD
    mask[int(fy0):int(fy1) + 1, int(fx0):int(fx1) + 1] = cv2.GC_FGD
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, None, bg, fg, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return (fx1 - fx0) * (fy1 - fy0) / (W * H), fg_box
    m = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    _, lab = cv2.connectedComponents(m)
    comp = lab == lab[int((y0 + y1) / 2), int((x0 + x1) / 2)]
    ys, xs = np.nonzero(comp)
    if not len(xs):
        return 0.0, fg_box
    return float(comp.mean()), [round(float(xs.min()) / W, 4), round(float(ys.min()) / H, 4),
                                round(float(xs.max()) / W, 4), round(float(ys.max()) / H, 4)]


def text_signs(img_array: np.ndarray, ocr) -> dict:
    """Розпізнає написи й розділяє їх на таблички/вивіски та вотермарки.

    text_area  — сумарна площа груп написів (з полями) — оцінка площі
                 табличок і вивісок у кадрі; кілька вивісок додаються;
    watermark  — текст вотермарки: дрібний поодинокий рядок біля краю кадру;
    text_boxes — [x1, y1, x2, y2, "sign"|"watermark"] у частках 0..1 —
                 показуються рамками на мініатюрі.
    """
    h, w = img_array.shape[:2]
    try:
        res, _ = ocr(img_array, text_score=0.0)   # власний фільтр OCR вимкнено — фільтруємо самі
    except TypeError:
        res, _ = ocr(img_array)
    lines = []
    for pts, txt, score in res or []:
        if not _valid_line(txt, score):
            continue
        p = np.array(pts, dtype=np.float32)
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        lines.append(([float(x0) / w, float(y0) / h, float(x1) / w, float(y1) / h], txt.strip()))
    boxes = [b for b, _ in lines]
    area, texts, wm_texts, out = 0.0, [], [], []
    obj_area, obj_box = 0.0, None
    for group in sorted(_clusters(boxes), key=len, reverse=True):
        gb = [boxes[i] for i in group]
        x0, y0 = min(b[0] for b in gb), min(b[1] for b in gb)
        x1, y1 = max(b[2] for b in gb), max(b[3] for b in gb)
        near_edge = x0 < EDGE or x1 > 1 - EDGE or y0 < EDGE or y1 > 1 - EDGE
        if len(group) == 1 and near_edge and (y1 - y0) < WM_MAX_HEIGHT:
            wm_texts.append(lines[group[0]][1])
            out.append([round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4), "watermark"])
            continue
        if len(out) < 5:   # предмет з написом — для найбільших груп
            a, ob = sign_object(img_array, [x0, y0, x1, y1])
            if a > obj_area:
                obj_area, obj_box = a, ob
        pw, ph = (x1 - x0) * PAD, (y1 - y0) * PAD
        x0, y0, x1, y1 = max(x0 - pw, 0), max(y0 - ph, 0), min(x1 + pw, 1), min(y1 + ph, 1)
        area += (x1 - x0) * (y1 - y0)
        texts.extend(lines[i][1] for i in group)
        out.append([round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4), "sign"])
    return {
        "text_area": round(min(area, 1.0), 4),
        "text": " ".join(texts)[:120],
        "watermark": " ".join(wm_texts)[:60],
        "sign_object": round(obj_area, 4),
        "sign_object_box": obj_box,
        "text_boxes": out[:30],
    }


def image_hashes(pil_img: Image.Image) -> dict[str, str]:
    """check_duplicate: хеші для порівняння (dhash за замовчуванням)."""
    return {"dhash": str(imagehash.dhash(pil_img)), "phash": str(imagehash.phash(pil_img))}


def analyze(pil_img: Image.Image, yolo_model, ocr=None) -> dict:
    """Повний набір метрик для одного фото.

    sign — вихідна евристика (як у main.py); text_area/watermark — OCR,
    якщо встановлено rapidocr-onnxruntime (інакше None)."""
    np_img = pil_to_bgr(pil_img)
    hashes = image_hashes(pil_img)
    m = {
        "blur": blur_score(np_img),
        "persons": person_detections(np_img, yolo_model) if yolo_model is not None else [],
        "sign": sign_coverage(np_img),
        "dhash": hashes["dhash"],
        "phash": hashes["phash"],
        "text_area": None, "text": "", "watermark": "", "text_boxes": [],
        "sign_object": None, "sign_object_box": None,
    }
    if ocr is not None:
        m.update(text_signs(np_img, ocr))
    return m


# ─── YOLO: одна модель на процес ─────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()
infer_lock = threading.Lock()


def get_yolo():
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            _model = YOLO(YOLO_WEIGHTS)
        return _model


_ocr = None


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
