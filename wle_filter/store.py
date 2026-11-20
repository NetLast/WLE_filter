"""SQLite-сховище: прогони, метрики фото, ручні рішення."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

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
"""

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
        rows = self._all("SELECT status, COUNT(*) n FROM images WHERE run_id=? GROUP BY status", (run_id,))
        c = {r["status"]: r["n"] for r in rows}
        total = sum(c.values())
        return {"total": total, "processed": c.get("done", 0) + c.get("error", 0),
                "errors": c.get("error", 0)}

    def set_run_status(self, run_id: int, status: str, message: str | None = None):
        self._exec("UPDATE runs SET status=?, message=?, updated_at=? WHERE id=?",
                   (status, message, now(), run_id))

    def set_run_thresholds(self, run_id: int, thresholds: dict):
        self._exec("UPDATE runs SET thresholds=?, updated_at=? WHERE id=?",
                   (json.dumps(thresholds), now(), run_id))

    def rename_run(self, run_id: int, name: str):
        self._exec("UPDATE runs SET name=?, updated_at=? WHERE id=?", (name, now(), run_id))

    def delete_run(self, run_id: int):
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
        return self._all("SELECT * FROM images WHERE run_id=? AND status='pending' ORDER BY seq", (run_id,))

    def save_metrics(self, image_id: int, m: dict):
        self._exec("UPDATE images SET status='done', error=NULL, blur=?, sign=?, dhash=?, phash=?,"
                   " persons=? WHERE id=?",
                   (m["blur"], m["sign"], m["dhash"], m["phash"], json.dumps(m["persons"]), image_id))

    def save_error(self, image_id: int, error: str):
        self._exec("UPDATE images SET status='error', error=? WHERE id=?", (error[:200], image_id))

    def retry_errors(self, run_id: int) -> int:
        return self._exec("UPDATE images SET status='pending', error=NULL WHERE run_id=? AND status='error'",
                          (run_id,)).rowcount

    def images(self, run_id: int) -> list[dict]:
        rows = self._all("SELECT * FROM images WHERE run_id=? ORDER BY seq", (run_id,))
        for r in rows:
            r["persons"] = json.loads(r["persons"]) if r["persons"] else []
        return rows

    def image(self, image_id: int) -> dict | None:
        rows = self._all("SELECT * FROM images WHERE id=?", (image_id,))
        if not rows:
            return None
        r = rows[0]
        r["persons"] = json.loads(r["persons"]) if r["persons"] else []
        return r

    def set_manual(self, image_id: int, decision: str | None, note: str | None = None):
        self._exec("UPDATE images SET manual=?, note=COALESCE(?, note), reviewed_at=? WHERE id=?",
                   (decision, note, now() if decision else None, image_id))

    def set_manual_bulk(self, ids: list[int], decision: str | None):
        t = now() if decision else None
        with self._lock:
            self._conn.executemany("UPDATE images SET manual=?, reviewed_at=? WHERE id=?",
                                   [(decision, t, i) for i in ids])
            self._conn.commit()
