from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.tools.validate import run_validation


def _upsert_job(state_store, *, status: JobStatus, job_id: str = "job-val") -> None:
    state_store.upsert(
        JobRecord(
            job_id=job_id,
            run_id=f"run-{job_id}",
            status=status,
            pid="11",
            paths={
                "jobDir": f"/workspace/jobs/{job_id}",
                "bestPath": f"/workspace/jobs/{job_id}/weights/best.pt",
                "dataConfigPath": "/workspace/datasets/a/data.yaml",
            },
            created_at=1,
            updated_at=1,
        )
    )


def test_run_validation_http_success(state_store) -> None:
    _upsert_job(state_store, status=JobStatus.COMPLETED, job_id="job-val-ok")
    control_client = MagicMock()
    control_client.run_validation.return_value = {
        "jobId": "job-val-ok",
        "metrics": {"map50": 0.42, "map5095": 0.28},
    }

    result = run_validation(
        "job-val-ok",
        state_store,
        None,
        jobs_dir="/workspace/jobs",
        work_dir="/workspace",
        control_client=control_client,
    )

    assert result["ok"] is True
    assert result["jobId"] == "job-val-ok"
    control_client.run_validation.assert_called_once()


def test_run_validation_rejects_non_completed_job(state_store) -> None:
    _upsert_job(state_store, status=JobStatus.RUNNING, job_id="job-val-running")

    result = run_validation(
        "job-val-running",
        state_store,
        None,
        jobs_dir="/workspace/jobs",
        work_dir="/workspace",
        control_client=MagicMock(),
    )

    assert result["ok"] is False
    assert result["errorCode"] == "JOB_NOT_COMPLETED"
