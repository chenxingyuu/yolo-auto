from __future__ import annotations

import json
from pathlib import Path

from yolo_auto.models import JobRecord, JobStatus, can_transition


class JobStateStore:
    def __init__(self, state_file: str) -> None:
        self._state_file = Path(state_file)
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._state_file.exists():
            self._write_raw({})

    def get(self, job_id: str) -> JobRecord | None:
        raw = self._read_raw()
        value = raw.get(job_id)
        if not value:
            return None
        return JobRecord.from_dict(value)

    def upsert(self, record: JobRecord) -> None:
        raw = self._read_raw()
        raw[record.job_id] = record.to_dict()
        self._write_raw(raw)

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
            last_notified_state=record.last_notified_state,
            last_metrics_at=record.last_metrics_at,
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
            last_notified_state=notified_state,
            last_metrics_at=record.last_metrics_at,
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
            last_notified_state=record.last_notified_state,
            last_metrics_at=now_ts,
        )
        self.upsert(updated)
        return updated

    def _read_raw(self) -> dict[str, dict]:
        if not self._state_file.exists():
            return {}
        return json.loads(self._state_file.read_text(encoding="utf-8"))

    def _write_raw(self, data: dict[str, dict]) -> None:
        self._state_file.write_text(
            json.dumps(data, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

