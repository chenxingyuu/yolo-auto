from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from yolo_auto.models import JobRecord, JobStatus, can_transition


class JobStateStore:
    def __init__(self, state_file: str) -> None:
        self._state_file = Path(state_file)
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = self._resolve_db_path(self._state_file)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._maybe_migrate_json()

    def get(self, job_id: str) -> JobRecord | None:
        row = self._conn.execute(
            "SELECT payload_json FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        return JobRecord.from_dict(json.loads(str(row["payload_json"])))

    def upsert(self, record: JobRecord) -> None:
        payload = json.dumps(record.to_dict(), ensure_ascii=True, separators=(",", ":"))
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO jobs(job_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                  payload_json = excluded.payload_json,
                  updated_at = excluded.updated_at
                """,
                (record.job_id, payload, int(record.updated_at)),
            )

    def update_status(self, job_id: str, target_status: JobStatus, now_ts: int) -> JobRecord:
        record = self.get(job_id)
        if not record:
            raise ValueError(f"job not found: {job_id}")
        if not can_transition(record.status, target_status):
            raise ValueError(f"invalid transition: {record.status.value} -> {target_status.value}")
        updated = JobRecord(
            job_id=record.job_id,
            run_id=record.run_id,
            status=target_status,
            pid=record.pid,
            paths=record.paths,
            created_at=record.created_at,
            updated_at=now_ts,
            env_id=record.env_id,
            last_notified_state=record.last_notified_state,
            feishu_message_id=record.feishu_message_id,
            last_metrics_at=record.last_metrics_at,
            train_epochs=record.train_epochs,
            last_reported_epoch=record.last_reported_epoch,
        )
        self.upsert(updated)
        return updated

    def mark_notified(self, job_id: str, notified_state: JobStatus, now_ts: int) -> JobRecord:
        record = self.get(job_id)
        if not record:
            raise ValueError(f"job not found: {job_id}")
        updated = JobRecord(
            job_id=record.job_id,
            run_id=record.run_id,
            status=record.status,
            pid=record.pid,
            paths=record.paths,
            created_at=record.created_at,
            updated_at=now_ts,
            env_id=record.env_id,
            last_notified_state=notified_state,
            feishu_message_id=record.feishu_message_id,
            last_metrics_at=record.last_metrics_at,
            train_epochs=record.train_epochs,
            last_reported_epoch=record.last_reported_epoch,
        )
        self.upsert(updated)
        return updated

    def mark_metrics(self, job_id: str, now_ts: int) -> JobRecord:
        record = self.get(job_id)
        if not record:
            raise ValueError(f"job not found: {job_id}")
        updated = JobRecord(
            job_id=record.job_id,
            run_id=record.run_id,
            status=record.status,
            pid=record.pid,
            paths=record.paths,
            created_at=record.created_at,
            updated_at=now_ts,
            env_id=record.env_id,
            last_notified_state=record.last_notified_state,
            feishu_message_id=record.feishu_message_id,
            last_metrics_at=now_ts,
            train_epochs=record.train_epochs,
            last_reported_epoch=record.last_reported_epoch,
        )
        self.upsert(updated)
        return updated

    def mark_milestone_epoch(self, job_id: str, epoch: int, now_ts: int) -> JobRecord:
        record = self.get(job_id)
        if not record:
            raise ValueError(f"job not found: {job_id}")
        updated = JobRecord(
            job_id=record.job_id,
            run_id=record.run_id,
            status=record.status,
            pid=record.pid,
            paths=record.paths,
            created_at=record.created_at,
            updated_at=now_ts,
            env_id=record.env_id,
            last_notified_state=record.last_notified_state,
            feishu_message_id=record.feishu_message_id,
            last_metrics_at=record.last_metrics_at,
            train_epochs=record.train_epochs,
            last_reported_epoch=epoch,
        )
        self.upsert(updated)
        return updated

    def mark_feishu_message(
        self,
        job_id: str,
        message_id: str | None,
        now_ts: int,
    ) -> JobRecord:
        record = self.get(job_id)
        if not record:
            raise ValueError(f"job not found: {job_id}")
        updated = JobRecord(
            job_id=record.job_id,
            run_id=record.run_id,
            status=record.status,
            pid=record.pid,
            paths=record.paths,
            created_at=record.created_at,
            updated_at=now_ts,
            env_id=record.env_id,
            last_notified_state=record.last_notified_state,
            feishu_message_id=message_id,
            last_metrics_at=record.last_metrics_at,
            train_epochs=record.train_epochs,
            last_reported_epoch=record.last_reported_epoch,
        )
        self.upsert(updated)
        return updated

    def list_all(self) -> list[JobRecord]:
        rows = self._conn.execute(
            "SELECT payload_json FROM jobs ORDER BY updated_at DESC"
        ).fetchall()
        return [JobRecord.from_dict(json.loads(str(row["payload_json"]))) for row in rows]

    def delete(self, job_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        return cur.rowcount > 0

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                  job_id TEXT PRIMARY KEY,
                  payload_json TEXT NOT NULL,
                  updated_at INTEGER NOT NULL
                )
                """
            )

    @staticmethod
    def _resolve_db_path(state_path: Path) -> Path:
        suffix = state_path.suffix.lower()
        if suffix in {".db", ".sqlite", ".sqlite3"}:
            return state_path
        return state_path.with_suffix(".db")

    def _maybe_migrate_json(self) -> None:
        json_path = self._state_file
        if not json_path.exists():
            return
        if json_path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            return

        raw = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"invalid state json format: {json_path}")

        if raw:
            with self._conn:
                for job_id, payload in raw.items():
                    record = JobRecord.from_dict(dict(payload))
                    payload_json = json.dumps(
                        record.to_dict(),
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    self._conn.execute(
                        """
                        INSERT INTO jobs(job_id, payload_json, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(job_id) DO UPDATE SET
                          payload_json = excluded.payload_json,
                          updated_at = excluded.updated_at
                        """,
                        (job_id, payload_json, int(record.updated_at)),
                    )
        backup_path = json_path.with_suffix(f"{json_path.suffix}.bak")
        json_path.rename(backup_path)

