from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.tools.export import run_export


def _make_completed_record(job_id: str) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        run_id="run-1",
        status=JobStatus.COMPLETED,
        pid="0",
        paths={
            "jobDir": f"/jobs/{job_id}",
            "logPath": f"/jobs/{job_id}/train.log",
            "metricsPath": f"/jobs/{job_id}/results.csv",
            "bestPath": f"/jobs/{job_id}/weights/best.pt",
            "lastPath": f"/jobs/{job_id}/weights/last.pt",
        },
        env_id="default",
        created_at=1,
        updated_at=1,
        train_epochs=2,
        last_reported_epoch=0,
        last_notified_state=None,
        last_metrics_at=None,
    )


def test_export_job_not_found(mock_ssh: MagicMock, state_store) -> None:
    result = run_export(
        job_id="missing",
        state_store=state_store,
        ssh_client=mock_ssh,
        jobs_dir="/jobs",
        work_dir="/workspace",
        formats=["onnx"],
    )
    assert result["ok"] is False
    assert result["errorCode"] == "JOB_NOT_FOUND"
    mock_ssh.file_exists.assert_not_called()


def test_export_job_not_completed(mock_ssh: MagicMock, state_store) -> None:
    job_id = "job-1"
    pending = JobRecord(
        job_id=job_id,
        run_id="run-1",
        status=JobStatus.RUNNING,
        pid="123",
        paths={"bestPath": f"/jobs/{job_id}/weights/best.pt"},
        env_id="default",
        created_at=1,
        updated_at=1,
        train_epochs=2,
        last_reported_epoch=0,
    )
    state_store.upsert(pending)

    result = run_export(
        job_id=job_id,
        state_store=state_store,
        ssh_client=mock_ssh,
        jobs_dir="/jobs",
        work_dir="/workspace",
        formats=["onnx"],
    )
    assert result["ok"] is False
    assert result["errorCode"] == "JOB_NOT_COMPLETED"
    mock_ssh.file_exists.assert_not_called()


def test_export_success_onnx(mock_ssh: MagicMock, state_store) -> None:
    job_id = "job-1"
    state_store.upsert(_make_completed_record(job_id))

    mock_ssh.file_exists.return_value = True
    # 1) yolo export command (onnx) => success
    # 2) find artifacts in job_dir => returns one file
    mock_ssh.execute.side_effect = [
        ("export ok", "", 0),
        ("./test.onnx\n", "", 0),
    ]

    result = run_export(
        job_id=job_id,
        state_store=state_store,
        ssh_client=mock_ssh,
        jobs_dir="/jobs",
        work_dir="/workspace",
        formats=["onnx"],
    )

    assert result["ok"] is True
    artifacts = result["artifacts"]
    assert any(a["format"] == "onnx" for a in artifacts)
    assert any(a["path"].endswith("/test.onnx") for a in artifacts)

