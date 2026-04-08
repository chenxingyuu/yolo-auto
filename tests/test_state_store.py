from __future__ import annotations

import json
from pathlib import Path

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.state_store import JobStateStore


def _record(job_id: str, status: JobStatus = JobStatus.RUNNING, updated_at: int = 1) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        run_id=f"run-{job_id}",
        status=status,
        pid="123",
        paths={},
        created_at=1,
        updated_at=updated_at,
    )


def test_state_store_crud_and_ordering(tmp_path: Path) -> None:
    store = JobStateStore(str(tmp_path / "jobs.db"))
    store.upsert(_record("job-1", updated_at=10))
    store.upsert(_record("job-2", updated_at=20))

    assert store.get("job-1") is not None
    all_jobs = store.list_all()
    assert [item.job_id for item in all_jobs] == ["job-2", "job-1"]

    assert store.find_by_run_id("run-job-1") is not None
    assert store.find_by_run_id("run-job-1").job_id == "job-1"
    assert store.find_by_run_id("missing") is None

    assert store.delete("job-1") is True
    assert store.delete("job-1") is False
    assert store.get("job-1") is None


def test_state_store_migrate_from_json_and_backup(tmp_path: Path) -> None:
    json_path = tmp_path / "jobs.json"
    payload = _record("job-json", status=JobStatus.COMPLETED, updated_at=99).to_dict()
    json_path.write_text(json.dumps({"job-json": payload}), encoding="utf-8")

    store = JobStateStore(str(json_path))

    migrated = store.get("job-json")
    assert migrated is not None
    assert migrated.status == JobStatus.COMPLETED

    assert (tmp_path / "jobs.db").exists()
    assert (tmp_path / "jobs.json.bak").exists()
    assert not json_path.exists()


def test_state_store_preserves_train_params(tmp_path: Path) -> None:
    store = JobStateStore(str(tmp_path / "jobs.db"))
    record = JobRecord(
        job_id="job-trainparams",
        run_id="run-job-trainparams",
        status=JobStatus.RUNNING,
        pid="123",
        paths={},
        created_at=1,
        updated_at=1,
        train_params={"lr0": 0.01, "optimizer": "SGD"},
        train_epochs=2,
        last_reported_epoch=0,
    )
    store.upsert(record)

    updated1 = store.update_status(
        "job-trainparams", JobStatus.COMPLETED, now_ts=2
    )
    assert updated1.train_params == record.train_params

    updated2 = store.mark_notified(
        "job-trainparams", JobStatus.STOPPED, now_ts=3
    )
    assert updated2.train_params == record.train_params

    updated3 = store.mark_metrics("job-trainparams", now_ts=4)
    assert updated3.train_params == record.train_params

    updated4 = store.mark_milestone_epoch(
        "job-trainparams", epoch=5, now_ts=5
    )
    assert updated4.train_params == record.train_params

    updated5 = store.mark_feishu_message(
        "job-trainparams", message_id="om_x", now_ts=6
    )
    assert updated5.train_params == record.train_params
