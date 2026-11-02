import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS entries(
    id INTEGER PRIMARY KEY,
    ns TEXT,
    model TEXT,
    doc_ver TEXT,
    tag TEXT,
    cat TEXT,
    key TEXT,
    query TEXT,
    answer TEXT,
    emb BLOB,
    ts REAL,
    ttl REAL,
    hits INT DEFAULT 0,
    last REAL
);
CREATE INDEX IF NOT EXISTS ix_key ON entries(key);
CREATE INDEX IF NOT EXISTS ix_ns_tag ON entries(ns, tag);
CREATE INDEX IF NOT EXISTS ix_last ON entries(last);

CREATE TABLE IF NOT EXISTS log(
    id INTEGER PRIMARY KEY,
    ts REAL,
    ns TEXT,
    query TEXT,
    level TEXT,
    score REAL,
    ms REAL,
    tokens INT,
    entry_id INT,
    cat TEXT
);

CREATE TABLE IF NOT EXISTS settings(
    k TEXT PRIMARY KEY,
    v TEXT
);
"""


class Database:
    """Thread-safe SQLite database manager for entries, logging, and dynamic settings."""

    def __init__(self, db_path: str = "cache.db") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.init_schema()

    def init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def get_thresholds(self, default_thresholds: Dict[str, float]) -> Dict[str, float]:
        with self._lock:
            row = self._conn.execute("SELECT v FROM settings WHERE k='thr'").fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except Exception:
                    pass
            return dict(default_thresholds)

    def save_thresholds(self, thresholds: Dict[str, float]) -> None:
        with self._lock:
            self._conn.execute("REPLACE INTO settings VALUES('thr', ?)", (json.dumps(thresholds),))
            self._conn.commit()

    def get_entry_by_key(self, key: str) -> Optional[Tuple[int, str, float, float]]:
        with self._lock:
            row = self._conn.execute("SELECT id, answer, ts, ttl FROM entries WHERE key=?", (key,)).fetchone()
            return row

    def get_entry_by_id(self, entry_id: int) -> Optional[Tuple[int, str, str, float, float]]:
        with self._lock:
            row = self._conn.execute("SELECT id, query, answer, ts, ttl FROM entries WHERE id=?", (entry_id,)).fetchone()
            return row

    def touch_entry(self, entry_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE entries SET hits=hits+1, last=? WHERE id=?", (time.time(), entry_id))
            self._conn.commit()

    def insert_entry(
        self,
        ns: str,
        model: str,
        doc_ver: str,
        tag: str,
        cat: str,
        key: str,
        query: str,
        answer: str,
        ttl: float,
    ) -> Tuple[int, List[int]]:
        """Inserts a new entry, replacing any prior entry with the same key. Returns (new_id, old_ids)."""
        now = time.time()
        with self._lock:
            old_ids = [r[0] for r in self._conn.execute("SELECT id FROM entries WHERE key=?", (key,)).fetchall()]
            cur = self._conn.execute(
                """
                INSERT INTO entries(ns, model, doc_ver, tag, cat, key, query, answer, emb, ts, ttl, last)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ns, model, doc_ver, tag, cat, key, query, answer, b"", now, ttl, now),
            )
            new_id = cur.lastrowid
            assert new_id is not None
            if old_ids:
                q = ",".join("?" * len(old_ids))
                self._conn.execute(f"DELETE FROM entries WHERE id IN ({q})", old_ids)
            self._conn.commit()
            return new_id, old_ids

    def count_entries(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])

    def get_lru_excess_ids(self, max_entries: int) -> List[int]:
        with self._lock:
            total = self.count_entries()
            if total <= max_entries:
                return []
            excess = total - max_entries
            return [r[0] for r in self._conn.execute("SELECT id FROM entries ORDER BY last ASC LIMIT ?", (excess,)).fetchall()]

    def delete_by_ids(self, ids: List[int]) -> List[str]:
        """Deletes entries by id and returns associated cache keys."""
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        with self._lock:
            keys = [r[0] for r in self._conn.execute(f"SELECT key FROM entries WHERE id IN ({q})", ids).fetchall()]
            self._conn.execute(f"DELETE FROM entries WHERE id IN ({q})", ids)
            self._conn.commit()
            return keys

    def delete_by_tag_and_ns(self, tag: str, ns: str) -> List[int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM entries WHERE (?='' OR tag=?) AND (?='' OR ns=?)",
                (tag, tag, ns, ns),
            ).fetchall()
            return [r[0] for r in rows]

    def clear_all(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM entries")
            self._conn.commit()

    def log_request(
        self,
        ns: str,
        query: str,
        level: str,
        score: float,
        ms: float,
        tokens: int,
        entry_id: Optional[int],
        cat: str,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO log(ts, ns, query, level, score, ms, tokens, entry_id, cat)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (time.time(), ns, query, level, score, ms, tokens, entry_id, cat),
            )
            self._conn.commit()
            assert cur.lastrowid is not None
            return cur.lastrowid

    def get_log_record(self, log_id: int) -> Optional[Tuple[str, str]]:
        with self._lock:
            return self._conn.execute("SELECT cat, level FROM log WHERE id=?", (log_id,)).fetchone()

    def get_stats(self, ns: str = "") -> Dict[str, Any]:
        with self._lock:
            w, p = ("WHERE ns=?", (ns,)) if ns else ("", ())
            cur = self._conn.execute(
                f"SELECT level, COUNT(*), AVG(ms), SUM(tokens) FROM log {w} GROUP BY level",
                p,
            )
            levels = {
                r[0]: {"n": r[1], "avg_ms": round(r[2], 1) if r[2] else 0.0, "tokens": r[3] or 0}
                for r in cur.fetchall()
            }
            recent = [
                {"query": r[0], "level": r[1], "score": round(r[2], 4), "ms": round(r[3], 1)}
                for r in self._conn.execute(
                    f"SELECT query, level, score, ms FROM log {w} ORDER BY id DESC LIMIT 12",
                    p,
                ).fetchall()
            ]
            top = [
                {"query": r[0], "hits": r[1]}
                for r in self._conn.execute(
                    "SELECT query, hits FROM entries ORDER BY hits DESC LIMIT 5"
                ).fetchall()
            ]
            return {"levels": levels, "recent": recent, "top": top}

    def get_entries_list(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, ns, tag, cat, query, answer, hits, ts, ttl
                FROM entries ORDER BY last DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "ns": r[1],
                    "tag": r[2],
                    "cat": r[3],
                    "query": r[4],
                    "answer": r[5],
                    "hits": r[6],
                    "ts": r[7],
                    "ttl": r[8],
                }
                for r in rows
            ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

