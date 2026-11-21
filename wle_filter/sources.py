"""Джерела фото: категорія Wikimedia Commons, CSV зі списком файлів (WLM) і локальна папка.

Кожне джерело повертає список словників-«кандидатів» з полями:
    key, title, page_id, author, timestamp, width, height, mime,
    thumb_url, page_url, local_path
"""
from __future__ import annotations

import csv
import io
import json
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


def fetch_file_categories(session: requests.Session, page_ids: list[int], progress=None) -> dict[int, str]:
    """Категорії файлів Commons: {pageid: 'Кат1|Кат2'} (без префікса «Category:», видимі й приховані)."""
    out: dict[int, list[str]] = {}
    for i in range(0, len(page_ids), 50):
        batch = page_ids[i:i + 50]
        params = {"action": "query", "pageids": "|".join(str(x) for x in batch),
                  "prop": "categories", "cllimit": "max"}
        while True:
            data = _api_get(session, params)
            for p in data.get("query", {}).get("pages", []):
                out.setdefault(p["pageid"], []).extend(
                    c["title"].removeprefix("Category:") for c in p.get("categories", []))
            if "continue" not in data:
                break
            params.update(data["continue"])
        if progress:
            progress(f"Категорії файлів: {min(i + 50, len(page_ids))}/{len(page_ids)}")
        time.sleep(0.3)
    return {k: "|".join(v) for k, v in out.items()}


def attach_csv(cands: list[dict], csv_path: str) -> int:
    """Додає monument_id і категорії з CSV до кандидатів за назвою файлу. Повертає, скільком знайшлося."""
    rows = {_name_key(r["title"]): r for r in parse_wlm_csv(Path(csv_path).read_text(encoding="utf-8-sig"))}
    found = 0
    for c in cands:
        r = rows.get(_name_key(c["title"]))
        if not r:
            continue
        found += 1
        c["monument_id"] = r["monument_id"]
        extra = [x for x in r["categories"].split("|") if x and x not in (c.get("categories") or "")]
        c["categories"] = "|".join(filter(None, [c.get("categories")] + extra))
    return found


def list_commons_category(category: str, limit: int | None = None,
                          session: requests.Session | None = None,
                          progress=None, with_categories: bool = False) -> list[dict]:
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

    cats = fetch_file_categories(session, [m["pageid"] for m in members], progress) if with_categories else {}
    out = []
    for m in members:
        ii = by_id.get(m["pageid"], {})
        out.append({
            "categories": cats.get(m["pageid"]),
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


# ─── CSV зі списком файлів (WLM) ─────────────────────────────────────────────
CSV_REQUIRED = ("title", "monument_id")


def _json_rows(text: str) -> list[dict]:
    """Рядки {title, monument_id, categories} з JSON. Підтримуються:
    список об'єктів [{"title":…, "monument_id":…, "categories":…}] (також ключі file/name і id),
    словник {"назва файлу": "monument_id"} або {"назва": {"monument_id":…, "categories":…}},
    і експорт цього застосунку {"images": [...]}."""
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("images"), list):
        data = data["images"]
    items: list[tuple[str, object]] = []
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        for o in data:
            if isinstance(o, dict):
                title = next((o[k] for k in ("title", "file", "name") if o.get(k)), "")
                items.append((str(title), o))
    else:
        raise ValueError("JSON має бути списком об'єктів або словником {назва файлу: monument_id}")
    if not items:
        raise ValueError("У JSON немає записів зі списком фото")
    rows = []
    for title, v in items:
        if isinstance(v, dict):
            mid = next((v[k] for k in ("monument_id", "id", "monument") if v.get(k)), "")
            cats = v.get("categories", "")
        else:
            mid, cats = v, ""
        if isinstance(cats, list):
            cats = "|".join(str(c) for c in cats)
        rows.append({"title": str(title), "monument_id": str(mid or ""), "categories": str(cats or "")})
    if rows and not any(r["title"] for r in rows):
        raise ValueError("У JSON немає назв файлів (ключі title / file / name)")
    return rows


def parse_wlm_csv(text: str, oblast: str = "") -> list[dict]:
    """Рядки списку (CSV або JSON) → {title, monument_id, categories}.

    CSV: колонки title, monument_id, categories. JSON: див. _json_rows.
    oblast — код області: лишаються рядки, де monument_id починається на «<код>-».
    Файли, що повторюються, зводяться до першого входження; categories різних рядків
    одного файлу об'єднуються.
    """
    text = text.lstrip("\ufeff")
    if text.lstrip()[:1] in ("[", "{"):
        try:
            raw_rows = _json_rows(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Некоректний JSON: {e}") from None
    else:
        head = text.split("\n", 1)[0]
        delim = max(",;\t", key=head.count) if head.strip() else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delim)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        missing = [c for c in CSV_REQUIRED if c not in cols]
        if missing:
            raise ValueError(f"У CSV немає колонок: {', '.join(missing)}. Є: {', '.join(cols) or '—'}")
        raw_rows = [{k: (v or "").strip() for k, v in
                     ((k.strip() if k else "", v) for k, v in raw.items())}
                    for raw in reader]
    prefix = oblast.strip().rstrip("-") + "-" if oblast.strip() else ""
    rows: dict[str, dict] = {}
    for row in raw_rows:
        title, mid = row.get("title", "").strip(), row.get("monument_id", "").strip()
        if not title or (prefix and not mid.startswith(prefix)):
            continue
        title = title.replace("_", " ")
        if not title.lower().startswith("file:"):
            title = "File:" + title
        cur = rows.get(title)
        cats = row.get("categories", "").strip()
        if cur is None:
            rows[title] = {"title": title, "monument_id": mid, "categories": cats}
        elif cats and cats not in cur["categories"]:
            cur["categories"] += "|" + cats
    return list(rows.values())


def list_wlm_csv(csv_path: str, oblast: str = "", limit: int | None = None,
                 session: requests.Session | None = None, progress=None) -> list[dict]:
    """Кандидати з CSV; метадані та thumb 800 px беруться з Commons пачками по 50."""
    session = session or requests.Session()
    text = Path(csv_path).read_text(encoding="utf-8-sig")
    rows = parse_wlm_csv(text, oblast)
    if limit:
        rows = rows[:limit]
    info: dict[str, dict] = {}
    for i in range(0, len(rows), 50):
        batch = rows[i:i + 50]
        data = _api_get(session, {
            "action": "query", "titles": "|".join(r["title"] for r in batch),
            "prop": "imageinfo", "iiprop": "url|size|user|timestamp|mime",
            "iiurlwidth": str(ANALYSIS_WIDTH)})
        q = data.get("query", {})
        norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
        pages = {p["title"]: p for p in q.get("pages", [])}
        for r in batch:
            page = pages.get(norm.get(r["title"], r["title"]))
            if page and not page.get("missing"):
                info[r["title"]] = {**((page.get("imageinfo") or [{}])[0]), "pageid": page.get("pageid")}
        if progress:
            progress(f"Метадані: {min(i + 50, len(rows))}/{len(rows)}")
        time.sleep(0.3)

    out = []
    for r in rows:
        ii = info.get(r["title"], {})
        out.append({
            "key": r["title"], "title": r["title"], "page_id": ii.get("pageid"),
            "author": ii.get("user"), "timestamp": ii.get("timestamp"),
            "width": ii.get("width"), "height": ii.get("height"), "mime": ii.get("mime"),
            "thumb_url": ii.get("thumburl") or ii.get("url"),
            "page_url": ii.get("descriptionurl") or commons_page_url(r["title"]),
            "local_path": None,
            "monument_id": r["monument_id"], "categories": r["categories"],
        })
    return out


def _name_key(name: str) -> str:
    """Ключ порівняння назви файлу: без «File:»/«File_», підкреслення = пробіли, без регістру."""
    n = name.replace("_", " ").strip().lower()
    for pre in ("file:", "file "):
        if n.startswith(pre):
            n = n[len(pre):].strip()
    return n


def list_wlm_folder(folder: str, csv_path: str | None = None, oblast: str = "",
                    limit: int | None = None) -> tuple[list[dict], int]:
    """Локальна папка для WLM. Якщо є CSV — файли зіставляються з рядками за назвою
    (monument_id, категорії), а в прогін потрапляють лише знайдені; повертає (кандидати, не знайдено).
    Без CSV беруться всі фото папки."""
    local = list_local_folder(folder)
    if not csv_path:
        return (local[:limit] if limit else local), 0
    rows = parse_wlm_csv(Path(csv_path).read_text(encoding="utf-8-sig"), oblast)
    by_name = {}
    for c in local:
        by_name.setdefault(_name_key(c["title"]), c)
    out, missing = [], 0
    for r in rows:
        c = by_name.get(_name_key(r["title"]))
        if not c:
            missing += 1
            continue
        out.append({**c, "key": r["title"], "title": r["title"], "page_url": commons_page_url(r["title"]),
                    "monument_id": r["monument_id"], "categories": r["categories"]})
    return (out[:limit] if limit else out), missing


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
