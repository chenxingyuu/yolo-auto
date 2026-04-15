from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.tools.status import get_status


def _upsert_running_job(state_store, job_id: str = "job-1") -> None:
    state_store.upsert(
        JobRecord(
            job_id=job_id,
            run_id=f"run-{job_id}",
            status=JobStatus.RUNNING,
            pid="321",
            paths={
                "jobDir": f"/workspace/jobs/{job_id}",
                "logPath": f"/workspace/jobs/{job_id}/train.log",
                "metricsPath": f"/workspace/jobs/{job_id}/results.csv",
                "bestPath": f"/workspace/jobs/{job_id}/weights/best.pt",
                "lastPath": f"/workspace/jobs/{job_id}/weights/last.pt",
            },
            created_at=1,
            updated_at=1,
            train_epochs=10,
            last_reported_epoch=0,
        )
    )


def test_get_status_job_not_found_http(mock_notifier: MagicMock, state_store) -> None:
    result = get_status(
        "missing",
        "run-missing",
        state_store,
        None,
        mock_notifier,
        control_client=MagicMock(),
    )
    assert result["ok"] is False
    assert result["errorCode"] == "JOB_NOT_FOUND"


def test_get_status_running_http_returns_metrics_and_progress_summary(
    mock_notifier: MagicMock, state_store
) -> None:
    _upsert_running_job(state_store, "job-http-status")
    control_client = MagicMock()
    control_client.get_training_status.return_value = {
        "status": "running",
        "progress": 0.5,
        "elapsedSeconds": 120,
        "metrics": {
            "epoch": 5,
            "map50": 0.4,
            "map5095": 0.31,
            "precision": 0.6,
            "recall": 0.5,
        },
        "trainingRows": [
            {
                "epoch": "1",
                "train/box_loss": "0.2",
                "metrics/mAP50(B)": "0.3",
                "metrics/mAP50-95(B)": "0.2",
                "metrics/recall(B)": "0.4",
            },
            {
                "epoch": "5",
                "train/box_loss": "0.1",
                "metrics/mAP50(B)": "0.4",
                "metrics/mAP50-95(B)": "0.31",
                "metrics/recall(B)": "0.5",
            },
        ],
    }

    result = get_status(
        "job-http-status",
        "run-job-http-status",
        state_store,
        None,
        mock_notifier,
        control_client=control_client,
        feishu_report_enable=True,
        feishu_report_every_n_epochs=5,
    )

    assert result["ok"] is True
    assert result["status"] == "running"
    assert result["metrics"]["epoch"] == 5
    assert "5/10 epochs" in result["progressSummary"]
    # epoch=5 reaches milestone threshold from last_reported_epoch=0
    mock_notifier.send_schema_card_with_message_id.assert_called_once()
