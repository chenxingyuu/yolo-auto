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
    mock_tracker: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute.return_value = ("", "", 0)
    mock_ssh.execute_background.return_value = ("12345", 0)

    req = replace(_make_req("job-no-lr"), learning_rate=None)
    result = start_training(req, mock_ssh, mock_notifier, mock_tracker, state_store)

    assert result["ok"] is True
    cmd = mock_ssh.execute_background.call_args[0][0]
    assert "lr0=" not in cmd
    cfg = mock_tracker.start_run.call_args.kwargs["config"]
    assert "learning_rate" not in cfg


def test_start_training_optimizer_auto_hints(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    mock_tracker: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute.return_value = ("", "", 0)
    mock_ssh.execute_background.return_value = ("12345", 0)

    req = replace(
        _make_req("job-auto"),
        extra_args={"optimizer": "auto", "val": True},
    )
    result = start_training(req, mock_ssh, mock_notifier, mock_tracker, state_store)

    assert result["ok"] is True
    assert "trainingHints" in result
    assert any("optimizer=auto" in h for h in result["trainingHints"])
    card = mock_notifier.send_schema_card_with_message_id.call_args.kwargs["card"]
    body = " ".join(str(e) for e in card["body"]["elements"])
    assert "optimizer=auto" in body


def test_start_training_success(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    mock_tracker: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute.return_value = ("", "", 0)
    mock_ssh.execute_background.return_value = ("12345", 0)

    req = _make_req("job-1")
    result = start_training(req, mock_ssh, mock_notifier, mock_tracker, state_store)

    assert result["ok"] is True
    assert result["status"] == JobStatus.RUNNING.value
    stored = state_store.get("job-1")
    assert stored is not None
    assert stored.paths.get("modelPath") == "/workspace/yolov8n.pt"
    assert stored.paths.get("dataConfigPath") == "/data/dataset.yaml"
    mock_tracker.start_run.assert_called_once()
    call_kw = mock_tracker.start_run.call_args.kwargs
    assert call_kw["tags"]["yolo_job_id"] == "job-1"
    # dataset.yaml 在 /data 下时 dataset_scope_key 取父目录名 data
    assert call_kw["tags"]["yolo_data_stem"] == "data"
    assert call_kw["tags"]["yolo_model_stem"] == "yolov8n"
    assert call_kw["tags"]["yolo_source"] == "yolo_auto.training"
    assert call_kw["config"]["env_id"] == "default"
    mock_ssh.execute_background.assert_called_once()
    assert "lr0=0.01" in mock_ssh.execute_background.call_args[0][0]
    mock_notifier.send_schema_card_with_message_id.assert_called_once()
    kwargs = mock_notifier.send_schema_card_with_message_id.call_args.kwargs
    card = kwargs["card"]
    assert card["schema"] == "2.0"
    body = " ".join(str(e) for e in card["body"]["elements"])
    assert "job=" not in body and "runId" not in body
    assert "640" in body and "16" in body and "Epochs" in body


def test_start_training_dataset_provenance_tags(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    mock_tracker: MagicMock,
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
    result = start_training(req, mock_ssh, mock_notifier, mock_tracker, state_store)
    assert result["ok"] is True
    tags = mock_tracker.start_run.call_args.kwargs["tags"]
    assert tags["yolo_minio_export_zip"] == "task1-export.zip"
    assert tags["yolo_dataset_slug"] == "myds"
    assert tags["yolo_dataset_version_note"] == "v2 labels"
    stored = state_store.get("job-prov")
    assert stored and stored.dataset_provenance == {
        "minioExportZip": "task1-export.zip",
        "datasetSlug": "myds",
        "datasetVersionNote": "v2 labels",
    }


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
    mock_notifier.send_schema_card_with_message_id.reset_mock()

    result = start_training(req, mock_ssh, mock_notifier, mock_tracker, state_store)
    assert result["ok"] is True
    assert result["jobId"] == req.job_id
    assert result["status"] == JobStatus.RUNNING.value
    mock_ssh.execute_background.assert_not_called()
    mock_tracker.start_run.assert_not_called()
    mock_notifier.send_schema_card_with_message_id.assert_not_called()


def test_start_training_ssh_fail(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    mock_tracker: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute.return_value = ("", "", 0)
    mock_ssh.execute_background.side_effect = RuntimeError("ssh fail")

    req = _make_req("job-1")
    result = start_training(req, mock_ssh, mock_notifier, mock_tracker, state_store)

    assert result["ok"] is False
    assert result["errorCode"] == "START_FAILED"
    assert result["retryable"] is True


def test_start_training_model_not_found(
    mock_ssh: MagicMock,
    mock_notifier: MagicMock,
    mock_tracker: MagicMock,
    state_store,
) -> None:
    mock_ssh.execute.return_value = ("", "", 1)

    req = _make_req("job-missing-model")
    result = start_training(req, mock_ssh, mock_notifier, mock_tracker, state_store)

    assert result["ok"] is False
    assert result["errorCode"] == "MODEL_NOT_FOUND"
    mock_tracker.start_run.assert_not_called()
    mock_ssh.execute_background.assert_not_called()

