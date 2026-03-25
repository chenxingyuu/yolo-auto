from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.tools.training import TrainRequest, start_training


def _make_req(job_id: str) -> TrainRequest:
    return TrainRequest(
        job_id=job_id,
        model="yolov8n.pt",
        data_config_path="/data/dataset.yaml",
        epochs=2,
        img_size=640,
        batch=16,
        learning_rate=0.01,
        work_dir="/workspace",
        jobs_dir="/jobs",
        extra_args={"val": True},
    )


def test_start_training_success(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    mock_tracker: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute_background.return_value = ("12345", 0)

    req = _make_req("job-1")
    result = start_training(req, mock_ssh, mock_notifier, mock_tracker, state_store)

    assert result["ok"] is True
    assert result["status"] == JobStatus.RUNNING.value
    assert state_store.get("job-1") is not None
    mock_ssh.execute_background.assert_called_once()
    mock_notifier.send_training_update.assert_called()


def test_start_training_duplicate(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    mock_tracker: MagicMock,
    state_store,
) -> None:
    req = _make_req("job-1")
    existing = JobRecord(
        job_id=req.job_id,
        run_id="existing-run",
        status=JobStatus.RUNNING,
        pid="999",
        paths={},
        created_at=1,
        updated_at=1,
        train_epochs=req.epochs,
        last_reported_epoch=0,
        last_notified_state=None,
        last_metrics_at=None,
    )
    state_store.upsert(existing)

    mock_ssh.execute_background.reset_mock()
    mock_tracker.start_run.reset_mock()
    mock_notifier.send_training_update.reset_mock()

    result = start_training(req, mock_ssh, mock_notifier, mock_tracker, state_store)
    assert result["ok"] is True
    assert result["jobId"] == req.job_id
    assert result["status"] == JobStatus.RUNNING.value
    mock_ssh.execute_background.assert_not_called()
    mock_tracker.start_run.assert_not_called()
    mock_notifier.send_training_update.assert_not_called()


def test_start_training_ssh_fail(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    mock_tracker: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute_background.side_effect = RuntimeError("ssh fail")

    req = _make_req("job-1")
    result = start_training(req, mock_ssh, mock_notifier, mock_tracker, state_store)

    assert result["ok"] is False
    assert result["errorCode"] == "START_FAILED"
    assert result["retryable"] is True

