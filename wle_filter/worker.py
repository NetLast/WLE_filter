"""Фонова обробка прогону: список фото → завантаження → метрики.

Обробка відновлювана: уже оброблені фото пропускаються, тож прогін можна
поставити на паузу, закрити сервер і продовжити пізніше.
"""
from __future__ import annotations

import threading
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import requests

from . import analysis, config, sources
from .store import Store

PAIR_MIN_INLIERS = 12     # пари з меншою кількістю спільних точок не зберігаються
CANDIDATES_ALL = 15       # найсхожіших за загальним описом кадру серед усіх фото
CANDIDATES_AUTHOR = 40    # … і серед фото того самого автора


class Worker:
    def __init__(self, store: Store, yolo_loader=analysis.get_yolo, ocr_loader=analysis.get_ocr):
        self.store = store
        self.yolo_loader = yolo_loader
        self.ocr_loader = ocr_loader
        self._sift_cache: OrderedDict[int, tuple] = OrderedDict()
        self._threads: dict[int, threading.Thread] = {}
        self._stop: dict[int, threading.Event] = {}
        self._lock = threading.Lock()

    def is_running(self, run_id: int) -> bool:
        t = self._threads.get(run_id)
        return bool(t and t.is_alive())

    def start(self, run_id: int, limit: int | None = None) -> bool:
        with self._lock:
            if self.is_running(run_id):
                return False
            ev = threading.Event()
            self._stop[run_id] = ev
            t = threading.Thread(target=self._run, args=(run_id, ev, limit), daemon=True,
                                 name=f"wle-run-{run_id}")
            self._threads[run_id] = t
            t.start()
            return True

    def pause(self, run_id: int):
        ev = self._stop.get(run_id)
        if ev:
            ev.set()

    def join(self, run_id: int, timeout: float | None = None):
        t = self._threads.get(run_id)
        if t:
            t.join(timeout)

    # ─── Основний цикл ───────────────────────────────────────────────────────
    def _run(self, run_id: int, stop: threading.Event, limit: int | None):
        store = self.store
        run = store.get_run(run_id)
        try:
            if run["total"] == 0:
                store.set_run_status(run_id, "listing", "Отримую список фото…")
                if run["source_type"] == "commons":
                    cands = sources.list_commons_category(
                        run["source"], limit=limit,
                        progress=lambda m: store.set_run_status(run_id, "listing", m))
                else:
                    cands = sources.list_local_folder(run["source"], limit=limit)
                if not cands:
                    store.set_run_status(run_id, "failed", "Фото не знайдено")
                    return
                store.add_candidates(run_id, cands)

            store.set_run_status(run_id, "running", "Завантажую моделі…")
            model = self.yolo_loader()
            ocr = self.ocr_loader() if self.ocr_loader else None
            session = requests.Session()
            todo = store.pending_images(run_id)
            # Прогін, проаналізований старою версією: пороги старої шкали скидаємо до пресету
            if any(i["status"] == "done" for i in todo) and "horizon_enabled" not in run["thresholds"]:
                preset = config.load_presets().get(run["nomination"])
                if preset:
                    store.set_run_thresholds(run_id, preset["thresholds"])
            store.set_run_status(run_id, "running", None)

            for img in todo:
                if stop.is_set():
                    store.set_run_status(run_id, "paused", "Призупинено")
                    return
                self._process_one(img, model, session, ocr)

            store.set_run_status(run_id, "done", None)
        except Exception as e:  # noqa: BLE001 — показуємо помилку в інтерфейсі
            traceback.print_exc()
            store.set_run_status(run_id, "failed", f"{type(e).__name__}: {e}"[:300])

    def _process_one(self, img: dict, model, session, ocr=None):
        store = self.store
        try:
            if img.get("mime") and not img["mime"].startswith("image/"):
                raise RuntimeError(f"unsupported: {img['mime']}")
            cache = self.cache_path(img)
            if not cache.exists():
                if img.get("local_path"):
                    src = img["local_path"]
                    pil = analysis.load_for_analysis(src)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    pil.save(cache, "JPEG", quality=88)
                else:
                    if not img.get("thumb_url"):
                        raise RuntimeError("download_failed: немає URL")
                    sources.download(img["thumb_url"], cache, session)
            pil = analysis.load_for_analysis(str(cache))
            with analysis.infer_lock:
                metrics = analysis.analyze(pil, model, ocr)
            pts, desc = metrics["_sift"]
            np.savez_compressed(self.sift_path(img), pts=pts, desc=desc)
            self._sift_cache[img["id"]] = (pts, desc)
            store.save_metrics(img["id"], metrics)
            self._find_pairs(img, metrics)
        except Exception as e:  # noqa: BLE001
            msg = str(e) if str(e).startswith(("download_failed", "unsupported")) else \
                f"error: {type(e).__name__}: {e}"
            store.save_error(img["id"], msg)

    # ─── Пошук дублікатів ────────────────────────────────────────────────────
    def _sift(self, image_id: int, run_id: int):
        if image_id in self._sift_cache:
            self._sift_cache.move_to_end(image_id)
            return self._sift_cache[image_id]
        p = self.store.cache_dir / str(run_id) / f"{image_id}.sift.npz"
        if not p.exists():
            return None
        with np.load(p) as z:
            v = (z["pts"], z["desc"])
        self._sift_cache[image_id] = v
        while len(self._sift_cache) > 400:
            self._sift_cache.popitem(last=False)
        return v

    def _find_pairs(self, img: dict, metrics: dict):
        """Порівнює нове фото з уже проаналізованими: спершу швидкий відбір
        кандидатів за загальним описом кадру, потім точна перевірка SIFT."""
        emb = metrics.get("_embed")
        others = [(i, a, e) for i, a, e in self.store.embeddings(img["run_id"]) if i != img["id"]]
        if not others:
            return
        ids = [o[0] for o in others]
        sims = np.zeros(len(others), np.float32)
        if emb is not None:
            e32 = emb.astype(np.float32)
            for j, o in enumerate(others):
                if o[2] is not None and len(o[2]) == e32.size * 2:
                    sims[j] = float(np.frombuffer(o[2], dtype=np.float16).astype(np.float32) @ e32)
        order = np.argsort(-sims)
        cand = set(int(ids[j]) for j in order[:CANDIDATES_ALL])
        same = [j for j in order if others[j][1] and others[j][1] == img.get("author")]
        cand |= set(int(ids[j]) for j in same[:CANDIDATES_AUTHOR])
        mine = metrics["_sift"]
        seq = {r["id"]: r["seq"] for r in self.store.images(img["run_id"]) if r["id"] in cand}
        found = []
        self.store.clear_pairs_for(img["id"])
        for oid in cand:
            other = self._sift(oid, img["run_id"])
            if other is None:
                continue
            n = analysis.match_inliers(mine, other)
            if n >= PAIR_MIN_INLIERS:
                a, b = (oid, img["id"]) if seq.get(oid, 0) < img["seq"] else (img["id"], oid)
                found.append((a, b, n))
        if found:
            self.store.save_pairs(img["run_id"], found)

    def sift_path(self, img: dict) -> Path:
        return self.store.cache_dir / str(img["run_id"]) / f"{img['id']}.sift.npz"

    def cache_path(self, img: dict) -> Path:
        return self.store.cache_dir / str(img["run_id"]) / f"{img['id']}.jpg"
