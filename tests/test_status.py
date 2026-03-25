from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.tools.status import get_status

CSV_CONTENT = (
    "epoch,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(B),metrics/recall(B)\n"
    "0,0.1,0.2,0.3,0.4,0.5\n"
    "1,0.12,0.34,0.56,0.78,0.90\n"
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

