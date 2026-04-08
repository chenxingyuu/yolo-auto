from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.tools.tuner import TuneRequest, auto_tune


def test_auto_tune_returns_mlflow_comparison_fields(state_store, mock_ssh, mock_notifier) -> None:
    base_job = "job-tune"
    trial_job = f"{base_job}-t1"
    state_store.upsert(
        JobRecord(
            job_id=trial_job,
            run_id=trial_job,
            status=JobStatus.COMPLETED,
            pid="1",
            paths={
                "jobDir": f"/jobs/{trial_job}",
                "logPath": f"/jobs/{trial_job}/train.log",
                "metricsPath": f"/jobs/{trial_job}/results.csv",
                "bestPath": f"/jobs/{trial_job}/weights/best.pt",
                "lastPath": f"/jobs/{trial_job}/weights/last.pt",
            },
            created_at=1,
            updated_at=1,
            train_epochs=1,
            last_reported_epoch=0,
            last_notified_state=None,
            last_metrics_at=None,
        )
    )
    mock_ssh.execute.return_value = (
        "epoch,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(B),metrics/recall(B)\n"
        "1,0.1,0.2,0.3,0.4,0.5\n",
        "",
        0,
    )
    mock_ssh.process_alive.return_value = False

    mock_tracker = MagicMock()
    mock_tracker.summarize_top_runs.return_value = [
        {"runId": "r1", "metricKey": "map5095", "metric": 0.3}
    ]
    mock_tracker.get_experiment_url.return_value = "http://localhost:5001/#/experiments/1"

    req = TuneRequest(
        env_id="default",
        base_job_id=base_job,
        model="yolov8n.pt",
        data_config_path="/data/d.yaml",
        epochs=1,
        search_space={"learningRate": [0.01], "batch": [16], "imgSize": [640]},
        max_trials=1,
        work_dir="/workspace",
        jobs_dir="/jobs",
    )
    result = auto_tune(req, mock_ssh, mock_notifier, mock_tracker, state_store)
    assert result["ok"] is True
    assert "bestFromMlflow" in result
    assert "mlflowTopRuns" in result
    assert "disagreement" in result
