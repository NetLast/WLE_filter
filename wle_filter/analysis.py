"""Обчислення «сирих» метрик фото.

Усе, що тут рахується, НЕ залежить від порогів чутливості — пороги
застосовуються пізніше у verdict.py. Тому прогін аналізу робиться один раз,
а повзунки в інтерфейсі просто перераховують вердикти.
"""
from __future__ import annotations

import threading

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps

from .config import ANALYSIS_WIDTH, PERSON_CONF_FLOOR, YOLO_WEIGHTS


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


# ─── Метрики ─────────────────────────────────────────────────────────────────
def blur_score(img_bgr: np.ndarray) -> float:
    """Дисперсія Лапласіана на зображенні ≤600 px. Менше значення — розмитіше."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > 600:
        scale = 600 / max(h, w)
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)


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


def sign_coverage(img_bgr: np.ndarray) -> float:
    """Частка кадру, вкрита текстоподібними горизонтальними блоками."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    dilated = cv2.dilate(bw, kernel, iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_area = 0
    for c in cnts:
        _, _, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(ch, 1.0)
        area = cw * ch
        if 2 < aspect < 60 and 8 < ch < h * 0.2 and area > 200:
            text_area += area
    return round(min(text_area / (h * w), 1.0), 4)


def image_hashes(pil_img: Image.Image) -> dict[str, str]:
    return {"dhash": str(imagehash.dhash(pil_img)), "phash": str(imagehash.phash(pil_img))}


def analyze(pil_img: Image.Image, yolo_model) -> dict:
    """Повний набір метрик для одного фото."""
    bgr = pil_to_bgr(pil_img)
    hashes = image_hashes(pil_img)
    return {
        "blur": blur_score(bgr),
        "persons": person_detections(bgr, yolo_model) if yolo_model is not None else [],
        "sign": sign_coverage(bgr),
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
