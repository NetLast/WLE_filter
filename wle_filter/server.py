"""FastAPI-сервер: REST API + веб-інтерфейс для перевірки фото.

Запуск:  python -m wle_filter.server   (або:  python app.py)
Потім відкрийте http://127.0.0.1:8000
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Literal, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from . import __version__, config
from .config import Thresholds
from .store import Store
from .verdict import REASON_LABELS, evaluate, reasons_text
from .worker import Worker

STATIC_DIR = Path(__file__).parent / "static"


# ─── Моделі запитів ──────────────────────────────────────────────────────────
class RunCreate(BaseModel):
    name: str = ""
    campaign: Literal["wle", "wlm"] = "wle"
    nomination: str = "quality"
    source_type: Literal["commons", "folder", "csv"]
    source: str = ""                       # категорія / папка / ім'я CSV-файлу (для відображення)
    csv_text: Optional[str] = None         # вміст CSV (завантажується з браузера)
    csv_path: Optional[str] = None         # або шлях до CSV на комп'ютері сервера
    oblast: str = ""                       # WLM: префікс monument_id, напр. «63»
    limit: Optional[int] = Field(None, ge=1)
    start: bool = True


class YearsUpload(BaseModel):
    data: dict                             # {monument_id: рік}
    replace: bool = False


class ResultsQuery(BaseModel):
    thresholds: Optional[dict] = None      # якщо None — збережені пороги прогону
    view: str = "flagged"                  # flagged|ok|error|pending|manual|all
    reason: Optional[str] = None           # blur|person|sign|duplicate|mpx
    sort: str = "seq"                      # seq|blur|person|sign|title
    search: str = ""
    offset: int = 0
    limit: int = 60


class Decision(BaseModel):
    decision: Optional[Literal["accept", "reject"]] = None
    note: Optional[str] = None


class BulkDecision(BaseModel):
    ids: list[int]
    decision: Optional[Literal["accept", "reject"]] = None


class PresetSave(BaseModel):
    thresholds: dict
    label: Optional[str] = None


# ─── Допоміжне ───────────────────────────────────────────────────────────────
def _hist(values: list[float], lo: float, hi: float, bins: int = 40, log: bool = False) -> dict:
    counts = [0] * bins
    if log:
        lo, hi = math.log10(max(lo, 1)), math.log10(max(hi, 10))
    span = (hi - lo) or 1
    for v in values:
        x = math.log10(max(v, 1)) if log else v
        i = int((x - lo) / span * bins)
        counts[min(max(i, 0), bins - 1)] += 1
    return {"lo": lo, "hi": hi, "log": log, "counts": counts}


def _sort_key(sort: str):
    return {
        "blur": lambda r: (r["blur"] if r["blur"] is not None else 1e12),
        "person": lambda r: -r["person_area"],
        "sign": lambda r: -(r["text_area"] if r.get("text_area") is not None else (r["sign"] or 0)),
        "title": lambda r: r["title"].lower(),
    }.get(sort, lambda r: r["seq"])


def _merge(images: list[dict], evals: list[dict]) -> list[dict]:
    by_id = {e["id"]: e for e in evals}
    titles = {i["id"]: i["title"] for i in images}
    out = []
    for img in images:
        e = by_id[img["id"]]
        for r in e["reasons"]:
            if r["code"] == "duplicate":
                r["of_title"] = titles.get(r["of"], "")
        out.append({**img, **e})
    return out


def _public(item: dict) -> dict:
    keys = ("id", "seq", "title", "author", "page_id", "page_url", "timestamp", "width", "height",
            "status", "error", "blur", "blur_global", "sign", "text_area", "text", "watermark", "text_boxes",
            "sign_object", "sign_object_box", "monument_id", "categories", "year",
            "persons", "person_area",
            "horizon_angle", "horizon_conf", "horizon_row", "shadows", "highlights", "sat_clip",
            "outdated", "reasons", "auto_verdict", "verdict", "manual", "note")
    d = {k: item.get(k) for k in keys}
    d["thumb"] = f"/api/images/{item['id']}/thumb"
    return d


# ─── Застосунок ──────────────────────────────────────────────────────────────
def create_app(store: Store | None = None, worker: Worker | None = None) -> FastAPI:
    store = store or Store()
    worker = worker or Worker(store)

    # Прогони, що «висіли» після аварійного завершення, позначаємо як призупинені
    for r in store.list_runs():
        if r["status"] in ("running", "listing"):
            store.set_run_status(r["id"], "paused", "Сервер було перезапущено — натисніть «Продовжити»")

    app = FastAPI(title="WLE Filter", version=__version__,
                  description="Попередня автоматична перевірка фото Wiki Loves Earth")
    app.state.store = store
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.worker = worker

    def run_or_404(run_id: int) -> dict:
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(404, "Прогін не знайдено")
        run["active"] = worker.is_running(run_id)
        return run

    # ─── Сторінка ────────────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/info")
    def info():
        from .analysis import get_ocr
        return {"version": __version__, "reasons": REASON_LABELS, "ocr": get_ocr() is not None,
                "campaigns": config.CAMPAIGNS, "years_loaded": len(store.years()),
                "person_conf_floor": config.PERSON_CONF_FLOOR,
                "data_dir": str(store.path.parent)}

    # ─── Пресети ─────────────────────────────────────────────────────────────
    @app.get("/api/presets")
    def get_presets():
        return config.load_presets()

    @app.put("/api/presets/{key}")
    def put_preset(key: str, body: PresetSave):
        return config.save_preset(key, body.thresholds, body.label)

    @app.post("/api/presets/{key}/reset")
    def post_reset_preset(key: str):
        return config.reset_preset(key)

    # ─── Роки побудови (WLM) ─────────────────────────────────────────────────
    @app.get("/api/years")
    def get_years():
        ys = store.years()
        return {"total": len(ys), "min": min(ys.values()) if ys else None,
                "max": max(ys.values()) if ys else None}

    @app.post("/api/years")
    def post_years(body: YearsUpload):
        """Підвантажує {monument_id: рік}; вердикти всіх прогонів оновлюються одразу."""
        if not isinstance(body.data, dict) or not body.data:
            raise HTTPException(400, "Очікується JSON-об'єкт {\"63-101-0001\": 1890, …}")
        return store.load_years(body.data, replace=body.replace)

    @app.delete("/api/years", status_code=204)
    def delete_years():
        store.clear_years()
        return Response(status_code=204)

    # ─── Прогони ─────────────────────────────────────────────────────────────
    @app.get("/api/runs")
    def list_runs():
        runs = store.list_runs()
        for r in runs:
            r["active"] = worker.is_running(r["id"])
        return runs

    @app.post("/api/runs", status_code=201)
    def create_run(body: RunCreate):
        presets = config.load_presets()
        preset = presets.get(body.nomination)
        if not preset:
            raise HTTPException(400, f"Невідома номінація: {body.nomination}")
        if preset.get("campaign", config.DEFAULT_CAMPAIGN) != body.campaign:
            raise HTTPException(400, "Пресет не належить до цього режиму")
        source, options, rows = body.source.strip(), {}, None
        if body.campaign == "wlm" and body.source_type == "commons":
            raise HTTPException(400, "У режимі WLM джерело — CSV-файл або локальна папка")
        if body.source_type == "csv" and body.campaign != "wlm":
            raise HTTPException(400, "CSV-джерело доступне лише в режимі WLM")
        text = None
        if body.csv_text:
            text = body.csv_text
        elif body.csv_path:
            p = Path(body.csv_path).expanduser()
            if not p.is_file():
                raise HTTPException(400, f"Файл не знайдено: {body.csv_path}")
            text = p.read_text(encoding="utf-8-sig")
            source = source or p.name
        if body.source_type == "csv":
            if text is None:
                raise HTTPException(400, "Оберіть CSV-файл")
            source = source or "wlm.csv"
        else:
            if not source:
                raise HTTPException(400, "Вкажіть категорію або папку")
            if body.source_type == "folder" and not Path(source).expanduser().is_dir():
                raise HTTPException(400, f"Папку не знайдено: {source}")
        if text is not None and body.campaign == "wlm":
            from .sources import parse_wlm_csv
            try:
                rows = parse_wlm_csv(text, body.oblast)
            except ValueError as e:
                raise HTTPException(400, str(e))
            if not rows:
                raise HTTPException(400, "У CSV немає рядків" + (f" з monument_id «{body.oblast}-…»" if body.oblast.strip() else ""))
        scope = f"{source}" + (f" · область {body.oblast.strip()}" if rows is not None and body.oblast.strip() else "")
        name = body.name.strip() or f"{preset['label']}: {scope}"
        run_id = store.create_run(name, body.nomination, body.source_type, source,
                                  preset["thresholds"], campaign=body.campaign, options=options)
        if rows is not None:
            # зберігаємо вже відфільтрований список, щоб прогін не залежав від вихідного файлу
            up = store.path.parent / "uploads"
            up.mkdir(parents=True, exist_ok=True)
            dest = up / f"run_{run_id}.csv"
            import csv as _csv
            with dest.open("w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=["title", "monument_id", "categories"])
                w.writeheader()
                w.writerows(rows)
            store.set_run_options(run_id, {"csv_path": str(dest), "oblast": body.oblast.strip(),
                                           "rows": len(rows)})
        if body.start:
            worker.start(run_id, limit=body.limit)
        return run_or_404(run_id)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int):
        return run_or_404(run_id)

    @app.patch("/api/runs/{run_id}")
    def patch_run(run_id: int, name: str = Body(..., embed=True)):
        run_or_404(run_id)
        store.rename_run(run_id, name.strip() or "Без назви")
        return run_or_404(run_id)

    @app.delete("/api/runs/{run_id}", status_code=204)
    def delete_run(run_id: int):
        run_or_404(run_id)
        worker.pause(run_id)
        worker.join(run_id, timeout=30)
        store.delete_run(run_id)
        return Response(status_code=204)

    @app.post("/api/runs/{run_id}/start")
    def start_run(run_id: int):
        run_or_404(run_id)
        worker.start(run_id)
        return run_or_404(run_id)

    @app.post("/api/runs/{run_id}/pause")
    def pause_run(run_id: int):
        run_or_404(run_id)
        worker.pause(run_id)
        return run_or_404(run_id)

    @app.post("/api/runs/{run_id}/retry-errors")
    def retry_errors(run_id: int):
        run_or_404(run_id)
        n = store.retry_errors(run_id)
        if n:
            worker.start(run_id)
        return {"requeued": n, "run": run_or_404(run_id)}

    @app.post("/api/runs/{run_id}/reanalyze")
    def reanalyze(run_id: int):
        """Повторний аналіз усіх фото (з кешу мініатюр, без нового завантаження)."""
        run_or_404(run_id)
        worker.pause(run_id)
        worker.join(run_id, timeout=30)
        n = store.reanalyze_all(run_id)
        worker.start(run_id)
        return {"requeued": n, "run": run_or_404(run_id)}

    @app.put("/api/runs/{run_id}/thresholds")
    @app.post("/api/runs/{run_id}/thresholds", include_in_schema=False)  # для navigator.sendBeacon
    def put_thresholds(run_id: int, thresholds: dict = Body(...)):
        run_or_404(run_id)
        t = Thresholds.from_dict(thresholds).to_dict()
        store.set_run_thresholds(run_id, t)
        return t

    # ─── Результати з «живими» порогами ──────────────────────────────────────
    @app.post("/api/runs/{run_id}/results")
    def results(run_id: int, q: ResultsQuery):
        run = run_or_404(run_id)
        t = Thresholds.from_dict(q.thresholds if q.thresholds is not None else run["thresholds"])
        images = store.images(run_id)
        items = _merge(images, evaluate(images, t, store.pairs(run_id)))

        summary = {"total": len(items), "ok": 0, "flagged": 0, "error": 0, "pending": 0,
                   "manual": 0, "auto_flagged": 0, "overridden": 0,
                   "reasons": {k: 0 for k in REASON_LABELS if k != "error"}}
        for it in items:
            summary[it["verdict"]] = summary.get(it["verdict"], 0) + 1
            if it["auto_verdict"] == "flagged":
                summary["auto_flagged"] += 1
            if it["manual"]:
                summary["manual"] += 1
                if it["verdict"] != it["auto_verdict"]:
                    summary["overridden"] += 1
            for r in it["reasons"]:
                if r["code"] in summary["reasons"]:
                    summary["reasons"][r["code"]] += 1

        done = [it for it in items if it["status"] == "done" and not it.get("outdated")]
        blurs = [it["blur"] for it in done if it["blur"] is not None]
        histograms = {
            "blur": _hist(blurs, 1, max(blurs + [1000]), log=True),
            "person": _hist([it["person_area"] for it in done if it["person_area"] > 0], 0, 1),
            "person_zero": sum(1 for it in done if it["person_area"] == 0),
            "sign": _hist(sign_vals := [
                (it["text_area"] if t.sign_method == "ocr" and it.get("text_area") is not None else it["sign"]) or 0
                for it in done], 0, max(sign_vals + [0.05 if t.sign_method == "ocr" else 0.5])),
        }

        view = q.view
        if view == "manual":
            sel = [it for it in items if it["manual"]]
        elif view == "all":
            sel = items
        else:
            sel = [it for it in items if it["verdict"] == view]
        if q.reason:
            sel = [it for it in sel if any(r["code"] == q.reason for r in it["reasons"])]
        if q.search.strip():
            s = q.search.strip().lower()
            sel = [it for it in sel if s in it["title"].lower() or s in (it["author"] or "").lower()]
        sel.sort(key=_sort_key(q.sort))

        page = sel[q.offset:q.offset + max(1, min(q.limit, 500))]
        return {"run": run, "thresholds": t.to_dict(), "summary": summary,
                "histograms": histograms, "matched": len(sel),
                "ids": [it["id"] for it in sel],
                "items": [_public(it) for it in page]}

    # ─── Фото та рішення ─────────────────────────────────────────────────────
    @app.get("/api/images/{image_id}/thumb", include_in_schema=False)
    def thumb(image_id: int):
        img = store.image(image_id)
        if not img:
            raise HTTPException(404)
        cache = worker.cache_path(img)
        if cache.exists():
            return FileResponse(cache, media_type="image/jpeg",
                                headers={"Cache-Control": "max-age=86400"})
        if img.get("thumb_url"):
            return RedirectResponse(img["thumb_url"])
        if img.get("local_path") and os.path.exists(img["local_path"]):
            return FileResponse(img["local_path"])
        raise HTTPException(404)

    @app.get("/api/images/{image_id}/original", include_in_schema=False)
    def original(image_id: int):
        img = store.image(image_id)
        if not img:
            raise HTTPException(404)
        if img.get("local_path") and os.path.exists(img["local_path"]):
            return FileResponse(img["local_path"])
        if img.get("page_url"):
            return RedirectResponse(img["page_url"])
        raise HTTPException(404)

    @app.post("/api/images/{image_id}/decision")
    def decision(image_id: int, body: Decision):
        if not store.image(image_id):
            raise HTTPException(404)
        store.set_manual(image_id, body.decision, body.note)
        img = store.image(image_id)
        return {"id": image_id, "manual": img["manual"], "note": img["note"]}

    @app.post("/api/runs/{run_id}/decisions")
    def bulk_decision(run_id: int, body: BulkDecision):
        run_or_404(run_id)
        valid = {i["id"] for i in store.images(run_id)}
        ids = [i for i in body.ids if i in valid]
        store.set_manual_bulk(ids, body.decision)
        return {"updated": len(ids)}

    # ─── Експорт ─────────────────────────────────────────────────────────────
    @app.get("/api/runs/{run_id}/export")
    def export(run_id: int, format: Literal["csv", "json"] = "csv",
               only: Literal["all", "flagged", "ok"] = "all"):
        run = run_or_404(run_id)
        t = Thresholds.from_dict(run["thresholds"])
        images = store.images(run_id)
        items = _merge(images, evaluate(images, t, store.pairs(run_id)))
        if only != "all":
            items = [it for it in items if it["verdict"] == only]
        safe = "".join(ch if ch.isalnum() else "_" for ch in run["name"])[:60]

        def disposition(ext: str) -> dict:
            from urllib.parse import quote
            return {"Content-Disposition": f'attachment; filename="wle_run_{run_id}.{ext}"; '
                                           f"filename*=UTF-8''{quote(safe + '.' + ext)}"}

        if format == "json":
            payload = {
                "run": {k: run[k] for k in ("id", "name", "campaign", "nomination", "source_type", "source")},
                "thresholds": t.to_dict(),
                "images": [{
                    "page_id": it["page_id"], "title": it["title"], "author": it["author"],
                    "monument_id": it.get("monument_id"), "year": it.get("year"),
                    "verdict": it["verdict"], "auto_verdict": it["auto_verdict"],
                    "manual": it["manual"], "note": it["note"],
                    "reasons": [{k: v for k, v in r.items() if k != "of"} for r in it["reasons"]],
                    "metrics": {"blur": it["blur"], "sign": it["sign"],
                                "text_area": it.get("text_area"), "text": it.get("text") or None,
                                "watermark": it.get("watermark") or None,
                                "person_area": it["person_area"],
                                "width": it["width"], "height": it["height"]},
                } for it in items],
            }
            return Response(json.dumps(payload, ensure_ascii=False, indent=1),
                            media_type="application/json",
                            headers=disposition("json"))

        buf = io.StringIO()
        cols = ["file", "monument_id", "build_year", "page_id", "author", "blur_score", "person_area", "sign_coverage",
                "text_area", "text", "watermark",
                "duplicate_of", "width", "height", "verdict", "auto_verdict", "manual",
                "reasons", "note", "page_url", "local_path"]
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        for it in items:
            dup = next((r.get("of_title") for r in it["reasons"] if r["code"] == "duplicate"), "")
            w.writerow({"file": it["title"], "monument_id": it.get("monument_id") or "",
                        "build_year": it.get("year") or "", "page_id": it["page_id"] or "", "author": it["author"] or "",
                        "blur_score": it["blur"] if it["blur"] is not None else "",
                        "person_area": it["person_area"], "sign_coverage": it["sign"] if it["sign"] is not None else "",
                        "text_area": it["text_area"] if it.get("text_area") is not None else "",
                        "text": it.get("text") or "", "watermark": it.get("watermark") or "",
                        "duplicate_of": dup, "width": it["width"] or "", "height": it["height"] or "",
                        "verdict": it["verdict"], "auto_verdict": it["auto_verdict"],
                        "manual": it["manual"] or "",
                        "reasons": it["error"] if it["status"] == "error" else reasons_text(it["reasons"]),
                        "note": it["note"] or "", "page_url": it["page_url"] or "",
                        "local_path": it["local_path"] or ""})
        return Response("﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers=disposition("csv"))

    # ─── Вибір локальної папки ───────────────────────────────────────────────
    @app.get("/api/fs")
    def browse(path: str = Query("~")):
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            raise HTTPException(400, "Не папка")
        dirs, n_images = [], 0
        try:
            for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                if c.name.startswith("."):
                    continue
                if c.is_dir():
                    dirs.append(c.name)
                elif c.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"):
                    n_images += 1
        except PermissionError:
            raise HTTPException(403, "Немає доступу")
        return {"path": str(p), "parent": str(p.parent) if p.parent != p else None,
                "dirs": dirs, "images": n_images}

    return app


def main():
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(description="WLE Filter — веб-інтерфейс")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"WLE Filter {__version__}: http://{args.host}:{args.port}")
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
