from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

from yolo_auto.tools.training import TrainRequest, start_training


def _make_req(job_id: str) -> TrainRequest:
    return TrainRequest(
        job_id=job_id,
        model="yolov8n.pt",
        data_config_path="/workspace/datasets/dataset.yaml",
        epochs=2,
        img_size=640,
        batch=16,
        learning_rate=0.01,
        work_dir="/workspace",
        jobs_dir="/workspace/jobs",
        extra_args={"device": 0},
    )


def test_start_training_blocked_when_remote_has_active_job(
    mock_notifier: MagicMock,
    state_store,
) -> None:
    control_client = MagicMock()
    control_client.list_jobs.return_value = {
        "jobs": [{"jobId": "running-job", "executionId": "4321"}],
        "count": 1,
    }
    control_client.get_training_status.return_value = {"jobId": "running-job", "status": "running"}

    result = start_training(
        _make_req("job-new"),
        None,
        mock_notifier,
        state_store,
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="exp",
        control_client=control_client,
    )

    assert result["ok"] is False
    assert result["errorCode"] == "ACTIVE_TRAINING_EXISTS"
    assert result["activeJobs"] == [
        {"jobId": "running-job", "executionId": "4321", "status": "running"}
    ]
    control_client.start_training.assert_not_called()


def test_start_training_allows_continue_when_confirmed(
    mock_notifier: MagicMock,
    state_store,
) -> None:
    control_client = MagicMock()
    control_client.list_jobs.return_value = {
        "jobs": [{"jobId": "running-job", "executionId": "4321"}],
        "count": 1,
    }
    control_client.get_training_status.return_value = {"jobId": "running-job", "status": "running"}
    control_client.start_training.return_value = {"executionId": "9876", "status": "running"}
    req = replace(_make_req("job-confirmed"), confirm_continue_if_busy=True)

    result = start_training(
        req,
        None,
        mock_notifier,
        state_store,
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="exp",
        control_client=control_client,
    )

    assert result["ok"] is True
    assert result["jobId"] == "job-confirmed"
    control_client.start_training.assert_called_once()
