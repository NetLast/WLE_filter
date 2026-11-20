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

# 3 — повернення до вихідних алгоритмів (прогони v3.1 переаналізовуються автоматично)
METRICS_VERSION = 3


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


def image_hashes(pil_img: Image.Image) -> dict[str, str]:
    """check_duplicate: хеші для порівняння (dhash за замовчуванням)."""
    return {"dhash": str(imagehash.dhash(pil_img)), "phash": str(imagehash.phash(pil_img))}


def analyze(pil_img: Image.Image, yolo_model, ocr=None) -> dict:
    """Повний набір метрик для одного фото (ocr не використовується)."""
    np_img = pil_to_bgr(pil_img)
    hashes = image_hashes(pil_img)
    return {
        "blur": blur_score(np_img),
        "persons": person_detections(np_img, yolo_model) if yolo_model is not None else [],
        "sign": sign_coverage(np_img),
        "dhash": hashes["dhash"],
        "phash": hashes["phash"],
    }


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


def get_ocr():
    """Сумісність: OCR у вихідних фільтрах не використовується."""
    return None
