from __future__ import annotations

from dataclasses import replace
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


def test_start_training_omits_lr0_without_learning_rate(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute.return_value = ("", "", 0)
    mock_ssh.execute_background.return_value = ("12345", 0)

    req = replace(_make_req("job-no-lr"), learning_rate=None)
    result = start_training(req, mock_ssh, mock_notifier, state_store)

    assert result["ok"] is True
    cmd = mock_ssh.execute_background.call_args[0][0]
    assert "lr0=" not in cmd


def test_start_training_optimizer_auto_hints(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute.return_value = ("", "", 0)
    mock_ssh.execute_background.return_value = ("12345", 0)

    req = replace(
        _make_req("job-auto"),
        extra_args={"optimizer": "auto", "val": True},
    )
    result = start_training(req, mock_ssh, mock_notifier, state_store)

    assert result["ok"] is True
    assert "trainingHints" in result
    assert any("optimizer=auto" in h for h in result["trainingHints"])
    card = mock_notifier.send_schema_card_with_message_id.call_args.kwargs["card"]
    body = " ".join(str(e) for e in card["body"]["elements"])
    assert "optimizer=auto" in body


def test_start_training_success(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute.return_value = ("", "", 0)
    mock_ssh.execute_background.return_value = ("12345", 0)

    req = _make_req("job-1")
    result = start_training(req, mock_ssh, mock_notifier, state_store)

    assert result["ok"] is True
    assert result["status"] == JobStatus.RUNNING.value
    stored = state_store.get("job-1")
    assert stored is not None
    assert stored.paths.get("modelPath") == "/workspace/yolov8n.pt"
    assert stored.paths.get("dataConfigPath") == "/data/dataset.yaml"
    assert result["runId"] == "job-1"
    mock_ssh.execute_background.assert_called_once()
    command = mock_ssh.execute_background.call_args[0][0]
    assert "MLFLOW_TRACKING_URI=sqlite:////workspace/mlflow.db" in command
    assert "lr0=0.01" in command
    mock_notifier.send_schema_card_with_message_id.assert_called_once()
    kwargs = mock_notifier.send_schema_card_with_message_id.call_args.kwargs
    card = kwargs["card"]
    assert card["schema"] == "2.0"
    body = " ".join(str(e) for e in card["body"]["elements"])
    assert "job=" not in body and "runId" not in body
    assert "640" in body and "16" in body and "Epochs" in body


def test_start_training_card_contains_mlflow_button_when_url_provided(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute.return_value = ("", "", 0)
    mock_ssh.execute_background.return_value = ("12345", 0)

    req = _make_req("job-mlflow-btn")
    result = start_training(
        req,
        mock_ssh,
        mock_notifier,
        state_store,
        mlflow_url="http://mlflow.local/#/experiments/1",
    )

    assert result["ok"] is True
    card = mock_notifier.send_schema_card_with_message_id.call_args.kwargs["card"]
    body_elements = card["body"]["elements"]
    assert any(
        element.get("tag") == "column_set"
        and any(
            col.get("tag") == "column"
            and any(btn.get("tag") == "button" for btn in col.get("elements", []))
            for col in element.get("columns", [])
        )
        for element in body_elements
        if isinstance(element, dict)
    )


def test_start_training_dataset_provenance_tags(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute.return_value = ("", "", 0)
    mock_ssh.execute_background.return_value = ("12345", 0)
    req = replace(
        _make_req("job-prov"),
        minio_export_zip="task1-export.zip",
        dataset_slug="myds",
        dataset_version_note="v2 labels",
    )
    result = start_training(req, mock_ssh, mock_notifier, state_store)
    assert result["ok"] is True
    stored = state_store.get("job-prov")
    assert stored and stored.dataset_provenance == {
        "minioExportZip": "task1-export.zip",
        "datasetSlug": "myds",
        "datasetVersionNote": "v2 labels",
    }


def test_start_training_duplicate(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
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
    mock_notifier.send_schema_card_with_message_id.reset_mock()

    result = start_training(req, mock_ssh, mock_notifier, state_store)
    assert result["ok"] is True
    assert result["jobId"] == req.job_id
    assert result["status"] == JobStatus.RUNNING.value
    mock_ssh.execute_background.assert_not_called()
    mock_notifier.send_schema_card_with_message_id.assert_not_called()


def test_start_training_ssh_fail(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute.return_value = ("", "", 0)
    mock_ssh.execute_background.side_effect = RuntimeError("ssh fail")

    req = _make_req("job-1")
    result = start_training(req, mock_ssh, mock_notifier, state_store)

    assert result["ok"] is False
    assert result["errorCode"] == "START_FAILED"
    assert result["retryable"] is True


def test_start_training_model_not_found(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute.return_value = ("", "", 1)

    req = _make_req("job-missing-model")
    result = start_training(req, mock_ssh, mock_notifier, state_store)

    assert result["ok"] is False
    assert result["errorCode"] == "MODEL_NOT_FOUND"
    mock_ssh.execute_background.assert_not_called()

