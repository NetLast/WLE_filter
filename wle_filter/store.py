"""SQLite-сховище: прогони, метрики фото, ручні рішення."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .analysis import METRICS_VERSION
from .config import DATA_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    nomination  TEXT NOT NULL,
    source_type TEXT NOT NULL,          -- 'commons' | 'folder'
    source      TEXT NOT NULL,          -- назва категорії або шлях до папки
    status      TEXT NOT NULL DEFAULT 'new',  -- new|listing|running|paused|done|failed
    message     TEXT,
    thresholds  TEXT NOT NULL,          -- JSON поточних порогів цього прогону
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,       -- хронологічний порядок (для дублікатів)
    key         TEXT NOT NULL,
    title       TEXT NOT NULL,
    page_id     INTEGER,
    author      TEXT,
    timestamp   TEXT,
    width       INTEGER,
    height      INTEGER,
    mime        TEXT,
    thumb_url   TEXT,
    page_url    TEXT,
    local_path  TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending|done|error
    error       TEXT,
    blur        REAL,
    sign        REAL,
    dhash       TEXT,
    phash       TEXT,
    persons     TEXT,                   -- JSON [{conf, area, box}]
    manual      TEXT,                   -- NULL | 'accept' | 'reject'
    note        TEXT,
    reviewed_at TEXT,
    UNIQUE(run_id, key)
);
CREATE INDEX IF NOT EXISTS idx_images_run ON images(run_id, seq);
CREATE TABLE IF NOT EXISTS pairs (
    run_id  INTEGER NOT NULL,
    a       INTEGER NOT NULL,           -- раніше завантажене фото
    b       INTEGER NOT NULL,           -- пізніше завантажене фото
    inliers INTEGER NOT NULL,           -- спільні ключові точки (SIFT + RANSAC)
    PRIMARY KEY (a, b)
);
CREATE INDEX IF NOT EXISTS idx_pairs_run ON pairs(run_id);
"""

# Колонки, додані у v3.1 (міграція наявних баз)
NEW_COLUMNS = {
    "blur_global": "REAL", "text": "TEXT", "text_boxes": "TEXT",
    "horizon_angle": "REAL", "horizon_conf": "REAL", "horizon_row": "REAL",
    "shadows": "REAL", "highlights": "REAL", "sat_clip": "REAL",
    "embed": "BLOB", "mver": "INTEGER NOT NULL DEFAULT 0",
}

RUN_FIELDS = ("id", "name", "nomination", "source_type", "source", "status",
              "message", "thresholds", "created_at", "updated_at")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DATA_DIR / "wle_filter.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        have = {r[1] for r in self._conn.execute("PRAGMA table_info(images)")}
        for col, typ in NEW_COLUMNS.items():
            if col not in have:
                self._conn.execute(f"ALTER TABLE images ADD COLUMN {col} {typ}")
        self._conn.commit()

    @property
    def cache_dir(self) -> Path:
        return self.path.parent / "cache"

    def close(self):
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, args=()):
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur

    def _all(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    # ─── Прогони ─────────────────────────────────────────────────────────────
    def create_run(self, name, nomination, source_type, source, thresholds: dict) -> int:
        t = now()
        cur = self._exec(
            "INSERT INTO runs(name, nomination, source_type, source, thresholds, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (name, nomination, source_type, source, json.dumps(thresholds), t, t))
        return cur.lastrowid

    def get_run(self, run_id: int) -> dict | None:
        rows = self._all("SELECT * FROM runs WHERE id=?", (run_id,))
        if not rows:
            return None
        run = rows[0]
        run["thresholds"] = json.loads(run["thresholds"])
        run.update(self.run_counts(run_id))
        return run

    def list_runs(self) -> list[dict]:
        runs = self._all("SELECT * FROM runs ORDER BY id DESC")
        for r in runs:
            r["thresholds"] = json.loads(r["thresholds"])
            r.update(self.run_counts(r["id"]))
        return runs

    def run_counts(self, run_id: int) -> dict:
        rows = self._all(
            "SELECT CASE WHEN status='done' AND mver<? THEN 'outdated' ELSE status END st, COUNT(*) n"
            " FROM images WHERE run_id=? GROUP BY st", (METRICS_VERSION, run_id))
        c = {r["st"]: r["n"] for r in rows}
        total = sum(c.values())
        return {"total": total, "processed": c.get("done", 0) + c.get("error", 0),
                "errors": c.get("error", 0), "outdated": c.get("outdated", 0)}

    def set_run_status(self, run_id: int, status: str, message: str | None = None):
        self._exec("UPDATE runs SET status=?, message=?, updated_at=? WHERE id=?",
                   (status, message, now(), run_id))

    def set_run_thresholds(self, run_id: int, thresholds: dict):
        self._exec("UPDATE runs SET thresholds=?, updated_at=? WHERE id=?",
                   (json.dumps(thresholds), now(), run_id))

    def rename_run(self, run_id: int, name: str):
        self._exec("UPDATE runs SET name=?, updated_at=? WHERE id=?", (name, now(), run_id))

    def delete_run(self, run_id: int):
        self._exec("DELETE FROM pairs WHERE run_id=?", (run_id,))
        self._exec("DELETE FROM images WHERE run_id=?", (run_id,))
        self._exec("DELETE FROM runs WHERE id=?", (run_id,))

    # ─── Фото ────────────────────────────────────────────────────────────────
    def add_candidates(self, run_id: int, candidates: list[dict]) -> int:
        """Додає нові фото (наявні за key пропускає). Повертає кількість доданих."""
        candidates = sorted(candidates, key=lambda c: (c.get("timestamp") or "", c["key"]))
        with self._lock:
            start = self._conn.execute("SELECT COALESCE(MAX(seq), -1) FROM images WHERE run_id=?",
                                       (run_id,)).fetchone()[0] + 1
            added = 0
            for i, c in enumerate(candidates):
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO images(run_id, seq, key, title, page_id, author, timestamp,"
                    " width, height, mime, thumb_url, page_url, local_path)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, start + i, c["key"], c["title"], c.get("page_id"), c.get("author"),
                     c.get("timestamp"), c.get("width"), c.get("height"), c.get("mime"),
                     c.get("thumb_url"), c.get("page_url"), c.get("local_path")))
                added += cur.rowcount
            self._conn.commit()
        return added

    def pending_images(self, run_id: int) -> list[dict]:
        """Фото, які треба (пере)аналізувати: нові та з застарілою версією метрик."""
        return self._all("SELECT * FROM images WHERE run_id=? AND"
                         " (status='pending' OR (status='done' AND mver<?)) ORDER BY seq",
                         (run_id, METRICS_VERSION))

    def save_metrics(self, image_id: int, m: dict):
        emb = m.get("_embed")
        self._exec(
            "UPDATE images SET status='done', error=NULL, blur=?, blur_global=?, sign=?, text=?,"
            " text_boxes=?, dhash=?, phash=?, persons=?, horizon_angle=?, horizon_conf=?, horizon_row=?,"
            " shadows=?, highlights=?, sat_clip=?, embed=?, mver=? WHERE id=?",
            (m["blur"], m.get("blur_global"), m["sign"], m.get("text") or None,
             json.dumps(m.get("text_boxes") or []), m["dhash"], m["phash"], json.dumps(m["persons"]),
             m.get("horizon_angle"), m.get("horizon_conf"), m.get("horizon_row"),
             m.get("shadows"), m.get("highlights"), m.get("sat_clip"),
             emb.tobytes() if emb is not None else None, METRICS_VERSION, image_id))

    def embeddings(self, run_id: int) -> list[tuple[int, str | None, bytes]]:
        """(id, author, embed|None) уже проаналізованих фото поточної версії, за порядком seq."""
        with self._lock:
            return [tuple(r) for r in self._conn.execute(
                "SELECT id, author, embed FROM images WHERE run_id=? AND status='done' AND mver=?"
                " ORDER BY seq", (run_id, METRICS_VERSION))]

    def save_pairs(self, run_id: int, pairs: list[tuple[int, int, int]]):
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO pairs(run_id, a, b, inliers) VALUES (?,?,?,?)",
                                   [(run_id, a, b, n) for a, b, n in pairs])
            self._conn.commit()

    def clear_pairs_for(self, image_id: int):
        self._exec("DELETE FROM pairs WHERE a=? OR b=?", (image_id, image_id))

    def pairs(self, run_id: int) -> list[tuple[int, int, int]]:
        with self._lock:
            return [tuple(r) for r in self._conn.execute(
                "SELECT a, b, inliers FROM pairs WHERE run_id=?", (run_id,))]

    def reanalyze_all(self, run_id: int) -> int:
        """Позначає всі фото прогону для повторного аналізу (кеш мініатюр зберігається)."""
        self._exec("DELETE FROM pairs WHERE run_id=?", (run_id,))
        return self._exec("UPDATE images SET mver=0 WHERE run_id=? AND status='done'", (run_id,)).rowcount

    def save_error(self, image_id: int, error: str):
        self._exec("UPDATE images SET status='error', error=? WHERE id=?", (error[:200], image_id))

    def retry_errors(self, run_id: int) -> int:
        return self._exec("UPDATE images SET status='pending', error=NULL WHERE run_id=? AND status='error'",
                          (run_id,)).rowcount

    def images(self, run_id: int) -> list[dict]:
        rows = self._all("SELECT * FROM images WHERE run_id=? ORDER BY seq", (run_id,))
        for r in rows:
            self._decode(r)
        return rows

    @staticmethod
    def _decode(r: dict) -> dict:
        r["persons"] = json.loads(r["persons"]) if r.get("persons") else []
        r["text_boxes"] = json.loads(r["text_boxes"]) if r.get("text_boxes") else []
        r.pop("embed", None)
        if r.get("status") == "done" and (r.get("mver") or 0) < METRICS_VERSION:
            r["outdated"] = True
        return r

    def image(self, image_id: int) -> dict | None:
        rows = self._all("SELECT * FROM images WHERE id=?", (image_id,))
        if not rows:
            return None
        return self._decode(rows[0])

    def set_manual(self, image_id: int, decision: str | None, note: str | None = None):
        self._exec("UPDATE images SET manual=?, note=COALESCE(?, note), reviewed_at=? WHERE id=?",
                   (decision, note, now() if decision else None, image_id))

    def set_manual_bulk(self, ids: list[int], decision: str | None):
        t = now() if decision else None
        with self._lock:
            self._conn.executemany("UPDATE images SET manual=?, reviewed_at=? WHERE id=?",
                                   [(decision, t, i) for i in ids])
            self._conn.commit()
