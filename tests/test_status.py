from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.tools.status import _training_rows_to_chart_spec, get_status

CSV_CONTENT = (
    "epoch,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(B),metrics/recall(B)\n"
    "0,0.1,0.2,0.3,0.4,0.5\n"
    "1,0.12,0.34,0.56,0.78,0.90\n"
)

CSV_CONTENT_EPOCH5 = (
    "epoch,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(B),metrics/recall(B)\n"
    "0,0.2,0.2,0.2,0.4,0.5\n"
    "5,0.1,0.4,0.6,0.8,0.9\n"
)


def _make_record(job_id: str, *, status: JobStatus, pid: str, train_epochs: int) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        run_id="run-1",
        status=status,
        pid=pid,
        paths={
            "jobDir": f"/jobs/{job_id}",
            "logPath": f"/jobs/{job_id}/train.log",
            "metricsPath": f"/jobs/{job_id}/results.csv",
            "bestPath": f"/jobs/{job_id}/weights/best.pt",
            "lastPath": f"/jobs/{job_id}/weights/last.pt",
        },
        created_at=1,
        updated_at=1,
        train_epochs=train_epochs,
        last_reported_epoch=0,
        last_notified_state=None,
        last_metrics_at=None,
    )


def test_training_rows_to_chart_spec_none_when_only_epoch_zero() -> None:
    rows = [
        {
            "epoch": "0",
            "train/box_loss": "0.1",
            "metrics/mAP50(B)": "0.2",
            "metrics/mAP50-95(B)": "0.3",
            "metrics/recall(B)": "0.4",
        }
    ]
    assert _training_rows_to_chart_spec(rows) is None


def test_training_rows_to_chart_spec_builds_line_spec() -> None:
    rows = [
        {
            "epoch": "1",
            "train/box_loss": "0.1",
            "metrics/mAP50(B)": "0.34",
            "metrics/mAP50-95(B)": "0.56",
            "metrics/recall(B)": "0.78",
        }
    ]
    spec = _training_rows_to_chart_spec(rows)
    assert spec is not None
    assert spec["type"] == "line"
    assert spec["seriesField"] == "series"
    vals = spec["data"]["values"]
    assert len(vals) == 3
    assert {v["series"] for v in vals} == {"mAP50", "mAP50-95", "Recall"}


def test_get_status_job_not_found(
    mock_ssh: MagicMock,
    mock_tracker: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    result = get_status(
        job_id="job-1",
        run_id="run-1",
        state_store=state_store,
        ssh_client=mock_ssh,
        tracker=mock_tracker,
        notifier=mock_notifier,
    )
    assert result["ok"] is False
    assert result["errorCode"] == "JOB_NOT_FOUND"


def test_get_status_csv_not_ready_process_alive(
    mock_ssh: MagicMock,
    mock_tracker: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    record = _make_record("job-1", status=JobStatus.RUNNING, pid="123", train_epochs=5)
    state_store.upsert(record)

    mock_ssh.execute.return_value = ("", "results.csv not ready", 1)
    mock_ssh.process_alive.return_value = True
    mock_ssh.tail_file.return_value = ("", "", 0)

    result = get_status(
        job_id="job-1",
        run_id="run-1",
        state_store=state_store,
        ssh_client=mock_ssh,
        tracker=mock_tracker,
        notifier=mock_notifier,
    )

    assert result["ok"] is True
    assert result["status"] == JobStatus.RUNNING.value
    assert result["progress"] == 0.0
    assert result["error"] == "results.csv not ready"
    mock_ssh.tail_file.assert_not_called()


def test_get_status_running_with_metrics(
    mock_ssh: MagicMock,
    mock_tracker: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    record = _make_record("job-1", status=JobStatus.RUNNING, pid="123", train_epochs=2)
    state_store.upsert(record)

    mock_ssh.execute.return_value = (CSV_CONTENT, "", 0)
    mock_ssh.process_alive.return_value = True
    mock_notifier.send_schema_card_with_message_id.return_value = "om_1"

    result = get_status(
        job_id="job-1",
        run_id="run-1",
        state_store=state_store,
        ssh_client=mock_ssh,
        tracker=mock_tracker,
        notifier=mock_notifier,
    )

    assert result["ok"] is True
    assert result["status"] == JobStatus.RUNNING.value
    assert result["progress"] == 0.5
    metrics = result["metrics"]
    assert metrics["epoch"] == 1
    assert metrics["loss"] == 0.12
    assert metrics["map50"] == 0.34
    assert metrics["map5095"] == 0.56
    assert metrics["precision"] == 0.78
    assert metrics["recall"] == 0.9
    assert metrics["primaryMetric"] == 0.56
    latest = state_store.get("job-1")
    assert latest is not None
    assert latest.feishu_message_id == "om_1"


def test_get_status_completed(
    mock_ssh: MagicMock,
    mock_tracker: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    record = _make_record("job-1", status=JobStatus.RUNNING, pid="123", train_epochs=1)
    state_store.upsert(record)

    mock_ssh.execute.return_value = (CSV_CONTENT, "", 0)
    mock_ssh.process_alive.return_value = False

    result = get_status(
        job_id="job-1",
        run_id="run-1",
        state_store=state_store,
        ssh_client=mock_ssh,
        tracker=mock_tracker,
        notifier=mock_notifier,
    )

    assert result["ok"] is True
    assert result["status"] == JobStatus.COMPLETED.value
    assert result["progress"] == 1.0
    mock_tracker.finish_run.assert_called()


def test_get_status_milestone_message_contains_eta_and_delta(
    mock_ssh: MagicMock,
    mock_tracker: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    record = _make_record("job-2", status=JobStatus.RUNNING, pid="123", train_epochs=10)
    state_store.upsert(record)

    mock_ssh.execute.return_value = (CSV_CONTENT_EPOCH5, "", 0)
    mock_ssh.process_alive.return_value = True
    mock_notifier.send_schema_card_with_message_id.return_value = "om_new"

    result = get_status(
        job_id="job-2",
        run_id="run-1",
        state_store=state_store,
        ssh_client=mock_ssh,
        tracker=mock_tracker,
        notifier=mock_notifier,
        feishu_report_enable=True,
        feishu_report_every_n_epochs=5,
    )

    assert result["ok"] is True
    assert result["status"] == JobStatus.RUNNING.value
    mock_notifier.send_schema_card_with_message_id.assert_called_once()
    kwargs = mock_notifier.send_schema_card_with_message_id.call_args.kwargs
    card = kwargs["card"]
    assert card["schema"] == "2.0"
    assert card["header"]["title"]["content"] == "YOLO模型训练里程碑"
    assert card["body"]["elements"][0]["tag"] == "chart"
    assert card["body"]["elements"][0]["chart_spec"]["type"] == "line"
    assert card["body"]["elements"][1]["tag"] == "column_set"
    assert len(card["body"]["elements"][1]["columns"]) == 2
    assert any(
        "更新时间：" in str(element.get("content", ""))
        for element in card["body"]["elements"]
        if isinstance(element, dict)
    )


def test_get_status_card_supports_configurable_top_image(
    mock_ssh: MagicMock,
    mock_tracker: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    record = _make_record("job-img", status=JobStatus.RUNNING, pid="123", train_epochs=10)
    state_store.upsert(record)

    mock_ssh.execute.return_value = (CSV_CONTENT_EPOCH5, "", 0)
    mock_ssh.process_alive.return_value = True
    mock_notifier.send_schema_card_with_message_id.return_value = "om_img"

    result = get_status(
        job_id="job-img",
        run_id="run-1",
        state_store=state_store,
        ssh_client=mock_ssh,
        tracker=mock_tracker,
        notifier=mock_notifier,
        feishu_card_img_key="img_v3_demo",
        feishu_card_fallback_img_key="img_v3_fb",
    )

    assert result["ok"] is True
    kwargs = mock_notifier.send_schema_card_with_message_id.call_args.kwargs
    card = kwargs["card"]
    assert card["body"]["elements"][0]["tag"] == "img"
    assert card["body"]["elements"][0]["img_key"] == "img_v3_demo"
    assert card["body"]["elements"][0]["fallback_img_key"] == "img_v3_fb"
    assert card["body"]["elements"][1]["tag"] == "chart"
    assert card["body"]["elements"][2]["tag"] == "column_set"
    assert len(card["body"]["elements"][2]["columns"]) == 2
    assert any(
        "更新时间：" in str(element.get("content", ""))
        for element in card["body"]["elements"]
        if isinstance(element, dict)
    )


def test_get_status_updates_existing_card_then_no_resend(
    mock_ssh: MagicMock,
    mock_tracker: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    record = _make_record("job-3", status=JobStatus.RUNNING, pid="123", train_epochs=10)
    record = replace(record, feishu_message_id="om_exist")
    state_store.upsert(record)

    mock_ssh.execute.return_value = (CSV_CONTENT_EPOCH5, "", 0)
    mock_ssh.process_alive.return_value = True
    mock_notifier.update_schema_card.return_value = True

    result = get_status(
        job_id="job-3",
        run_id="run-1",
        state_store=state_store,
        ssh_client=mock_ssh,
        tracker=mock_tracker,
        notifier=mock_notifier,
        feishu_report_enable=True,
    )

    assert result["ok"] is True
    mock_notifier.update_schema_card.assert_called_once()
    mock_notifier.send_schema_card_with_message_id.assert_not_called()


def test_get_status_fallback_resend_when_update_fails(
    mock_ssh: MagicMock,
    mock_tracker: MagicMock,
    mock_notifier: MagicMock,
    state_store,
) -> None:
    record = _make_record("job-4", status=JobStatus.RUNNING, pid="123", train_epochs=10)
    record = replace(record, feishu_message_id="om_old")
    state_store.upsert(record)

    mock_ssh.execute.return_value = (CSV_CONTENT_EPOCH5, "", 0)
    mock_ssh.process_alive.return_value = True
    mock_notifier.update_schema_card.return_value = False
    mock_notifier.send_schema_card_with_message_id.return_value = "om_new"

    result = get_status(
        job_id="job-4",
        run_id="run-1",
        state_store=state_store,
        ssh_client=mock_ssh,
        tracker=mock_tracker,
        notifier=mock_notifier,
        feishu_report_enable=True,
    )

    assert result["ok"] is True
    latest = state_store.get("job-4")
    assert latest is not None
    assert latest.feishu_message_id == "om_new"

