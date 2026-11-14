"""Tiny SQLite cache for LLM explanations (no user data is stored)."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .config import settings

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    settings.web_cache_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.web_cache_db))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_explanations (
            cache_key TEXT PRIMARY KEY,
            lang TEXT NOT NULL,
            model TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def get(cache_key: str) -> Optional[Dict[str, Any]]:
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT payload FROM llm_explanations WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        finally:
            conn.close()
    return json.loads(row[0]) if row else None


def put(cache_key: str, lang: str, model: str, payload: Dict[str, Any]) -> None:
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO llm_explanations VALUES (?, ?, ?, ?, ?)",
                (
                    cache_key,
                    lang,
                    model,
                    json.dumps(payload, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
