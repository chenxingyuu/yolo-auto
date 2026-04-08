from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.tools.validate import run_validation


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
            "dataConfigPath": "/data/dataset.yaml",
        },
        created_at=1,
        updated_at=1,
        train_epochs=2,
        last_reported_epoch=0,
        last_notified_state=None,
        last_metrics_at=None,
    )


def test_validate_success(
    mock_ssh: MagicMock,
    state_store,
) -> None:
    job_id = "job-1"
    state_store.upsert(_make_completed_record(job_id))

    mock_ssh.file_exists.return_value = True
    stdout = """
                   Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                     all        128        929      0.649      0.591      0.634      0.451
    """.strip()
    mock_ssh.execute.return_value = (stdout, "", 0)

    result = run_validation(
        job_id=job_id,
        state_store=state_store,
        ssh_client=mock_ssh,
        jobs_dir="/jobs",
        work_dir="/workspace",
        data_config_path=None,
    )

    assert result["ok"] is True
    assert result["metrics"]["precision"] == 0.649
    assert result["metrics"]["recall"] == 0.591
    assert result["metrics"]["map50"] == 0.634
    assert result["metrics"]["map5095"] == 0.451
    assert result["modelPath"].endswith("best.pt")


def test_validate_job_not_found(
    mock_ssh: MagicMock,
    state_store,
) -> None:
    result = run_validation(
        job_id="missing",
        state_store=state_store,
        ssh_client=mock_ssh,
        jobs_dir="/jobs",
        work_dir="/workspace",
    )
    assert result["ok"] is False
    assert result["errorCode"] == "JOB_NOT_FOUND"
    mock_ssh.file_exists.assert_not_called()


def test_validate_job_not_completed(
    mock_ssh: MagicMock,
    state_store,
) -> None:
    job_id = "job-1"
    pending = JobRecord(
        job_id=job_id,
        run_id="run-1",
        status=JobStatus.RUNNING,
        pid="123",
        paths={"bestPath": "/jobs/job-1/weights/best.pt"},
        created_at=1,
        updated_at=1,
        train_epochs=2,
        last_reported_epoch=0,
        last_notified_state=None,
        last_metrics_at=None,
    )
    state_store.upsert(pending)

    result = run_validation(
        job_id=job_id,
        state_store=state_store,
        ssh_client=mock_ssh,
        jobs_dir="/jobs",
        work_dir="/workspace",
    )

    assert result["ok"] is False
    assert result["errorCode"] == "JOB_NOT_COMPLETED"
    mock_ssh.file_exists.assert_not_called()

