"""Джерела фото: категорія Wikimedia Commons і локальна папка.

Кожне джерело повертає список словників-«кандидатів» з полями:
    key, title, page_id, author, timestamp, width, height, mime,
    thumb_url, page_url, local_path
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

from .config import ANALYSIS_WIDTH, HTTP_HEADERS

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
SLEEP_S = 0.7


def commons_page_url(title: str) -> str:
    name = title.removeprefix("File:").replace(" ", "_")
    return "https://commons.wikimedia.org/wiki/File:" + quote(name)


# ─── Commons ─────────────────────────────────────────────────────────────────
def _api_get(session: requests.Session, params: dict, retries: int = 5) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    for attempt in range(retries):
        try:
            r = session.get(COMMONS_API, params=params, headers=HTTP_HEADERS, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep((2 ** attempt) * 2)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError("Commons API недоступний")


def list_commons_category(category: str, limit: int | None = None,
                          session: requests.Session | None = None,
                          progress=None) -> list[dict]:
    """Файли категорії з метаданими (pageid, автор, розмір, thumb 800 px)."""
    session = session or requests.Session()
    category = category.strip().removeprefix("Category:")
    members: list[dict] = []
    params = {"action": "query", "list": "categorymembers",
              "cmtitle": "Category:" + category, "cmtype": "file",
              "cmlimit": "500", "cmprop": "ids|title|timestamp"}
    while True:
        data = _api_get(session, params)
        members.extend(data.get("query", {}).get("categorymembers", []))
        if progress:
            progress(f"Отримано список: {len(members)} файлів")
        if limit and len(members) >= limit:
            members = members[:limit]
            break
        if "continue" not in data:
            break
        params.update(data["continue"])
        time.sleep(0.3)

    by_id: dict[int, dict] = {}
    for i in range(0, len(members), 50):
        batch = members[i:i + 50]
        data = _api_get(session, {
            "action": "query", "pageids": "|".join(str(m["pageid"]) for m in batch),
            "prop": "imageinfo", "iiprop": "url|size|user|timestamp|mime",
            "iiurlwidth": str(ANALYSIS_WIDTH)})
        for p in data.get("query", {}).get("pages", []):
            ii = (p.get("imageinfo") or [{}])[0]
            by_id[p["pageid"]] = ii
        if progress:
            progress(f"Метадані: {min(i + 50, len(members))}/{len(members)}")
        time.sleep(0.3)

    out = []
    for m in members:
        ii = by_id.get(m["pageid"], {})
        out.append({
            "key": m["title"],
            "title": m["title"],
            "page_id": m["pageid"],
            "author": ii.get("user"),
            "timestamp": ii.get("timestamp") or m.get("timestamp"),
            "width": ii.get("width"),
            "height": ii.get("height"),
            "mime": ii.get("mime"),
            "thumb_url": ii.get("thumburl") or ii.get("url"),
            "page_url": ii.get("descriptionurl") or commons_page_url(m["title"]),
            "local_path": None,
        })
    return out


def download(url: str, dest: Path, session: requests.Session | None = None) -> None:
    """Завантажує мініатюру з повторами при 429 (як у старому main.py)."""
    session = session or requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = "download_failed"
    for attempt in range(8):
        try:
            r = session.get(url, headers=HTTP_HEADERS, timeout=25)
            if r.status_code == 200:
                if not r.headers.get("Content-Type", "").startswith("image/"):
                    raise RuntimeError("download_failed: не зображення")
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(r.content)
                os.replace(tmp, dest)
                time.sleep(SLEEP_S)
                return
            if r.status_code == 429:
                time.sleep((2 ** attempt) * 3)
                last = "download_failed: 429"
            elif 400 <= r.status_code < 500:
                raise RuntimeError(f"download_failed: HTTP {r.status_code}")
            else:
                time.sleep(3)
                last = f"download_failed: HTTP {r.status_code}"
        except requests.RequestException as e:
            last = f"download_failed: {type(e).__name__}"
            time.sleep(3)
    raise RuntimeError(last)


# ─── Локальна папка ──────────────────────────────────────────────────────────
def list_local_folder(folder: str, limit: int | None = None) -> list[dict]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Папку не знайдено: {root}")
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                   and not p.name.startswith("."))
    if limit:
        files = files[:limit]
    out = []
    for p in files:
        w = h = None
        try:
            with Image.open(p) as im:
                w, h = im.size
                # EXIF-орієнтація 5–8 міняє сторони місцями
                if im.getexif().get(0x0112) in (5, 6, 7, 8):
                    w, h = h, w
        except Exception:
            pass
        rel = p.relative_to(root).as_posix()
        # Автор — назва підпапки першого рівня (якщо фото розкладені по авторах)
        author = rel.split("/")[0] if "/" in rel else None
        out.append({
            "key": rel,
            "title": p.name,
            "page_id": None,
            "author": author,
            "timestamp": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                                  .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "width": w,
            "height": h,
            "mime": None,
            "thumb_url": None,
            "page_url": None,
            "local_path": str(p),
        })
    return out
