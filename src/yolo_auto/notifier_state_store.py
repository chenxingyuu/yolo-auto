from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from yolo_auto.models import JobStatus


@dataclass(frozen=True)
class NotifierState:
    job_id: str
    feishu_message_id: str | None
    last_reported_epoch: int
    last_notified_state: JobStatus | None
    updated_at: int


class NotifierStateStore:
    """Minimal local state for notification dedup.

    This store intentionally does NOT persist training metrics/paths. It only keeps
    information needed to update Feishu cards and avoid duplicate milestone pushes.
    """

    def __init__(self, state_file: str) -> None:
        self._path = Path(state_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def get(self, job_id: str) -> NotifierState | None:
        row = self._conn.execute(
            """
            SELECT job_id, feishu_message_id, last_reported_epoch, last_notified_state, updated_at
            FROM notify_state
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        last_state = str(row["last_notified_state"] or "").strip()
        return NotifierState(
            job_id=str(row["job_id"]),
            feishu_message_id=str(row["feishu_message_id"]) if row["feishu_message_id"] else None,
            last_reported_epoch=int(row["last_reported_epoch"] or 0),
            last_notified_state=JobStatus(last_state) if last_state else None,
            updated_at=int(row["updated_at"] or 0),
        )

    def upsert(
        self,
        *,
        job_id: str,
        now_ts: int,
        feishu_message_id: str | None = None,
        last_reported_epoch: int | None = None,
        last_notified_state: JobStatus | None = None,
    ) -> NotifierState:
        prev = self.get(job_id)
        effective_message_id = feishu_message_id if feishu_message_id is not None else (prev.feishu_message_id if prev else None)
        effective_epoch = int(last_reported_epoch) if last_reported_epoch is not None else (prev.last_reported_epoch if prev else 0)
        effective_state = (
            last_notified_state if last_notified_state is not None else (prev.last_notified_state if prev else None)
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO notify_state(job_id, feishu_message_id, last_reported_epoch, last_notified_state, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                  feishu_message_id = excluded.feishu_message_id,
                  last_reported_epoch = excluded.last_reported_epoch,
                  last_notified_state = excluded.last_notified_state,
                  updated_at = excluded.updated_at
                """,
                (
                    job_id,
                    effective_message_id,
                    int(effective_epoch),
                    effective_state.value if effective_state else None,
                    int(now_ts),
                ),
            )
        return NotifierState(
            job_id=job_id,
            feishu_message_id=effective_message_id,
            last_reported_epoch=int(effective_epoch),
            last_notified_state=effective_state,
            updated_at=int(now_ts),
        )

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notify_state (
                  job_id TEXT PRIMARY KEY,
                  feishu_message_id TEXT,
                  last_reported_epoch INTEGER NOT NULL DEFAULT 0,
                  last_notified_state TEXT,
                  updated_at INTEGER NOT NULL
                )
                """
            )

