"""Thread-safe SQLite history storage; image binaries remain on disk."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_filename TEXT,
    original_path TEXT,
    preview_path TEXT,
    result_path TEXT,
    mask_path TEXT,
    report_path TEXT,
    file_hash TEXT UNIQUE,
    ocr_text TEXT,
    detected_entities TEXT,
    redaction_mode TEXT,
    model_name TEXT,
    model_source TEXT,
    processing_time TEXT,
    warnings TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""


class DatabaseService:
    def __init__(self, path: str | Path = "data/app.db", retries: int = 3) -> None:
        self.path = Path(path); self.retries = retries; self._lock = threading.RLock(); self.last_error = ""; self.available = self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def initialize(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(SCHEMA); connection.execute("PRAGMA journal_mode=WAL")
            self.last_error = ""; return True
        except (OSError, sqlite3.Error) as exc:
            self.last_error = f"数据库初始化失败：{exc}"; return False

    @staticmethod
    def file_hash(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
        return digest.hexdigest()

    def _execute_write(self, sql: str, parameters: tuple[Any, ...]) -> int | None:
        if not self.available: return None
        with self._lock:
            for attempt in range(self.retries):
                try:
                    with self._connect() as connection:
                        cursor = connection.execute(sql, parameters); self.last_error = ""; return int(cursor.lastrowid)
                except sqlite3.IntegrityError:
                    raise
                except sqlite3.OperationalError as exc:
                    if "locked" in str(exc).lower() and attempt + 1 < self.retries:
                        time.sleep(0.1 * (attempt + 1)); continue
                    self.last_error = f"数据库写入失败：{exc}"; return None
                except sqlite3.Error as exc:
                    self.last_error = f"数据库写入失败：{exc}"; return None
        return None

    def insert_document(self, record: dict[str, Any]) -> int | None:
        file_hash = record.get("file_hash")
        if not file_hash and record.get("original_path") and Path(record["original_path"]).is_file():
            file_hash = self.file_hash(record["original_path"])
        if not file_hash:
            self.last_error = "数据库写入失败：缺少文件 SHA-256"; return None
        fields = ("original_filename", "original_path", "preview_path", "result_path", "mask_path", "report_path", "file_hash", "ocr_text", "detected_entities", "redaction_mode", "model_name", "model_source", "processing_time", "warnings")
        values = []
        for field in fields:
            value = file_hash if field == "file_hash" else record.get(field)
            if field in {"detected_entities", "processing_time", "warnings"}: value = json.dumps(value if value is not None else ([] if field != "processing_time" else {}), ensure_ascii=False)
            values.append(value)
        try:
            return self._execute_write(f"INSERT INTO documents ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", tuple(values))
        except sqlite3.IntegrityError:
            existing = self.get_by_hash(file_hash); return int(existing["id"]) if existing else None

    def get_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM documents WHERE file_hash = ?", (file_hash,)); return rows[0] if rows else None

    def query_documents(self, page: int = 1, page_size: int = 20, filename: str = "", entity_type: str = "", descending: bool = True) -> list[dict[str, Any]]:
        clauses = []; values: list[Any] = []
        if filename: clauses.append("original_filename LIKE ?"); values.append(f"%{filename}%")
        if entity_type: clauses.append("detected_entities LIKE ?"); values.append(f'%"type": "{entity_type}"%')
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        order = "DESC" if descending else "ASC"; values.extend([max(1, page_size), max(0, page - 1) * max(1, page_size)])
        return self._query(f"SELECT * FROM documents{where} ORDER BY created_at {order}, id {order} LIMIT ? OFFSET ?", tuple(values))

    def delete_document(self, document_id: int, delete_files: bool = False) -> bool:
        record = self._query("SELECT * FROM documents WHERE id = ?", (document_id,))
        result = self._execute_write("DELETE FROM documents WHERE id = ?", (document_id,))
        if result is None: return False
        if delete_files and record:
            for field in ("preview_path", "result_path", "mask_path", "report_path"):
                path = record[0].get(field)
                if path:
                    try: Path(path).unlink(missing_ok=True)
                    except OSError: pass
        return True

    def _query(self, sql: str, parameters: tuple[Any, ...]) -> list[dict[str, Any]]:
        if not self.available: return []
        try:
            with self._lock, self._connect() as connection:
                rows = [dict(row) for row in connection.execute(sql, parameters).fetchall()]
            for row in rows:
                for field in ("detected_entities", "processing_time", "warnings"):
                    try: row[field] = json.loads(row[field] or ("{}" if field == "processing_time" else "[]"))
                    except json.JSONDecodeError: pass
            self.last_error = ""; return rows
        except sqlite3.Error as exc:
            self.last_error = f"数据库查询失败：{exc}"; return []
