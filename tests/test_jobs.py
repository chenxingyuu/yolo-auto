from __future__ import annotations

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.tools.jobs import delete_job


def _upsert(state_store, job_id: str, status: JobStatus) -> None:
    state_store.upsert(
        JobRecord(
            job_id=job_id,
            run_id="run-1",
            status=status,
            pid="1",
            paths={},
            created_at=1,
            updated_at=1,
        )
    )


def test_delete_job_success(state_store) -> None:
    _upsert(state_store, "job-done", JobStatus.COMPLETED)
    result = delete_job("job-done", state_store)
    assert result["ok"] is True
    assert result["deleted"] is True
    assert state_store.get("job-done") is None


def test_delete_job_reject_running(state_store) -> None:
    _upsert(state_store, "job-running", JobStatus.RUNNING)
    result = delete_job("job-running", state_store)
    assert result["ok"] is False
    assert result["errorCode"] == "JOB_DELETE_FORBIDDEN"


def test_delete_job_not_found(state_store) -> None:
    result = delete_job("missing-job", state_store)
    assert result["ok"] is False
    assert result["errorCode"] == "JOB_NOT_FOUND"
