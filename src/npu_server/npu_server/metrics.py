import logging
import time
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          REAL    NOT NULL,
    request_id         TEXT    NOT NULL,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    ttft_ms            REAL,
    tokens_per_sec     REAL,
    context_usage      INTEGER,
    model              TEXT,
    error              TEXT
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON requests (timestamp);
"""


@dataclass
class RequestRecord:
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    tokens_per_sec: float
    context_usage: int
    model: str
    error: str | None = None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = max(0, int(len(s) * 0.95) - 1)
    return s[idx]


class MetricsDB:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def setup(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        logger.info("Metrics DB ready at %s", self._db_path)

    async def record(self, rec: RequestRecord) -> None:
        if self._db is None:
            return
        try:
            await self._db.execute(
                """
                INSERT INTO requests
                    (timestamp, request_id, prompt_tokens, completion_tokens,
                     ttft_ms, tokens_per_sec, context_usage, model, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    rec.request_id,
                    rec.prompt_tokens,
                    rec.completion_tokens,
                    rec.ttft_ms,
                    rec.tokens_per_sec,
                    rec.context_usage,
                    rec.model,
                    rec.error,
                ),
            )
            await self._db.commit()
        except Exception:
            logger.warning("Failed to record metrics", exc_info=True)

    async def stats(self, window_hours: float | None) -> dict:
        if self._db is None:
            return {}

        since = (time.time() - window_hours * 3600) if window_hours is not None else 0.0
        window_label = f"{int(window_hours)}h" if window_hours is not None else "all"

        async with self._db.execute(
            "SELECT ttft_ms, tokens_per_sec, completion_tokens, error FROM requests WHERE timestamp >= ?",
            (since,),
        ) as cursor:
            rows = await cursor.fetchall()

        total = len(rows)
        errors = sum(1 for r in rows if r[3] is not None)
        ttfts = [r[0] for r in rows if r[0] is not None and r[3] is None]
        toks_per_sec = [r[1] for r in rows if r[1] is not None and r[3] is None]
        comp_tokens = [r[2] for r in rows if r[2] is not None]

        return {
            "window": window_label,
            "request_count": total,
            "error_count": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
            "p95_ttft_ms": round(_p95(ttfts), 1) if ttfts else None,
            "avg_tokens_per_sec": round(sum(toks_per_sec) / len(toks_per_sec), 2) if toks_per_sec else None,
            "avg_completion_tokens": round(sum(comp_tokens) / len(comp_tokens), 1) if comp_tokens else None,
            "total_completion_tokens": sum(comp_tokens),
        }

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
