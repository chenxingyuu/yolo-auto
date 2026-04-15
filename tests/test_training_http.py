from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.models import JobStatus
from yolo_auto.tools.training import TrainRequest, start_training


def _request(*, job_id: str = "job-1") -> TrainRequest:
    return TrainRequest(
        job_id=job_id,
        model="/workspace/models/yolo11n.pt",
        data_config_path="/workspace/datasets/a/data.yaml",
        epochs=10,
        img_size=640,
        batch=16,
        learning_rate=0.01,
        work_dir="/workspace",
        jobs_dir="/workspace/jobs",
        extra_args={"optimizer": "SGD"},
    )


def test_start_training_requires_control_client(
    mock_notifier: MagicMock, state_store
) -> None:
    result = start_training(
        _request(),
        None,
        mock_notifier,
        state_store,
        mlflow_tracking_uri="sqlite:////workspace/mlflow.db",
        mlflow_experiment_name="yolo-auto",
        control_client=None,
    )
    assert result["ok"] is False
    assert result["errorCode"] == "REMOTE_CLIENT_MISSING"


def test_start_training_http_success_persists_running_record(
    mock_notifier: MagicMock, state_store
) -> None:
    control_client = MagicMock()
    control_client.list_jobs.return_value = {"jobs": []}
    control_client.start_training.return_value = {"pid": "12345"}
    mock_notifier.send_schema_card_with_message_id.return_value = "om_msg_1"

    result = start_training(
        _request(job_id="job-http"),
        None,
        mock_notifier,
        state_store,
        mlflow_tracking_uri="sqlite:////workspace/mlflow.db",
        mlflow_experiment_name="yolo-auto",
        control_client=control_client,
    )

    assert result["ok"] is True
    assert result["jobId"] == "job-http"
    assert result["status"] == JobStatus.RUNNING.value
    control_client.start_training.assert_called_once()

    stored = state_store.get("job-http")
    assert stored is not None
    assert stored.status == JobStatus.RUNNING
    assert stored.pid == "12345"
