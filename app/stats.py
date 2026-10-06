"""Anonymous diagnosis counts. Label, latency, and timestamp only."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

logger = logging.getLogger("fieldsight")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS diagnoses (
    id BIGSERIAL PRIMARY KEY,
    label TEXT,
    latency_s DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS diagnoses_created_at_idx ON diagnoses (created_at)
"""

COUNT_TODAY_SQL = """
SELECT count(*) FROM diagnoses
WHERE created_at >= (date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')
"""

INSERT_SQL = """
INSERT INTO diagnoses (label, latency_s) VALUES (%s, %s)
"""


def today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _utc_midnight() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _clean_label(label: str | None) -> str | None:
    if label is None:
        return None
    text = str(label).strip()
    return text[:200] or None


def _clean_latency(latency_s: float | None) -> float | None:
    if latency_s is None:
        return None
    try:
        value = float(latency_s)
    except (TypeError, ValueError):
        return None
    if value < 0 or value > 600:
        return None
    return value


@dataclass
class DiagnosisRecord:
    label: str | None
    latency_s: float | None
    created_at: datetime


class Store(Protocol):
    kind: str
    fallback: bool

    def record(self, label: str | None, latency_s: float | None) -> None: ...

    def count_today(self) -> int: ...

    def close(self) -> None: ...


@dataclass
class MemoryStore:
    kind: str = "memory"
    fallback: bool = False
    rows: list[DiagnosisRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, label: str | None, latency_s: float | None) -> None:
        row = DiagnosisRecord(
            label=_clean_label(label),
            latency_s=_clean_latency(latency_s),
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self.rows.append(row)
            if len(self.rows) > 5000:
                self.rows = self.rows[-4000:]

    def count_today(self) -> int:
        start = _utc_midnight()
        with self._lock:
            return sum(1 for row in self.rows if row.created_at >= start)

    def close(self) -> None:
        return None


class PostgresStore:
    kind = "postgres"
    fallback = False

    def __init__(self, conn: Any):
        self._conn = conn
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                cur.execute(CREATE_INDEX_SQL)
            self._conn.commit()

    def _run(self, sql: str, params: tuple | None = None, fetch: bool = False) -> Any:
        with self._lock:
            try:
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone() if fetch else None
                self._conn.commit()
                return row
            except Exception:
                self._conn.rollback()
                raise

    def record(self, label: str | None, latency_s: float | None) -> None:
        self._run(INSERT_SQL, (_clean_label(label), _clean_latency(latency_s)))

    def count_today(self) -> int:
        row = self._run(COUNT_TODAY_SQL, fetch=True)
        return int(row[0]) if row else 0

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            logger.exception("failed to close postgres connection")


def _connect_psycopg(url: str) -> Any:
    import psycopg

    return psycopg.connect(url, connect_timeout=5)


def open_store(url: str | None, connector: Callable[[str], Any] | None = None) -> Store:
    if not url:
        return MemoryStore()
    connect = connector or _connect_psycopg
    conn = None
    try:
        conn = connect(url)
        return PostgresStore(conn)
    except Exception:
        logger.warning("DATABASE_URL is set but Postgres is unavailable; counts stay in memory")
        if conn is not None:
            try:
                conn.close()
            except Exception:
                logger.exception("failed to close postgres connection after setup error")
        store = MemoryStore()
        store.fallback = True
        return store
