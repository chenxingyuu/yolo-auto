from __future__ import annotations

import json
from unittest.mock import MagicMock

from mcp.server.fastmcp import FastMCP

from yolo_auto.config import Settings, SSHEnv
from yolo_auto.resources import register_resources


def _settings() -> Settings:
    return Settings(
        yolo_ssh_host="127.0.0.1",
        yolo_ssh_port=22,
        yolo_ssh_user="u",
        yolo_ssh_key_path="k",
        yolo_ssh_envs={"default": SSHEnv(host="127.0.0.1", port=22, user="u", key_path="k")},
        feishu_webhook_url="https://example.com",
        feishu_app_id=None,
        feishu_app_secret=None,
        feishu_chat_id=None,
        feishu_report_enable=True,
        feishu_report_every_n_epochs=5,
        feishu_card_img_key=None,
        feishu_card_fallback_img_key=None,
        primary_metric_key="map5095",
        mlflow_tracking_uri="sqlite:////data/mlflow/mlflow.db",
        mlflow_experiment_name="yolo-auto",
        mlflow_external_url="http://localhost:5001",
        yolo_work_dir="/workspace/yolo-auto",
        yolo_datasets_dir="/workspace/datasets",
        yolo_jobs_dir="/workspace/jobs",
        yolo_models_dir="/workspace/models",
        yolo_state_file=".state/jobs.db",
        watch_poll_interval_seconds=30,
        watch_lock_file=".state/watch.lock",
    )


def test_register_resources_exposes_mlflow_readonly(state_store) -> None:
    mcp = FastMCP("test")
    mock_ssh = MagicMock()
    mock_tracker = MagicMock()
    mock_tracker.summarize_top_runs.return_value = [
        {"runId": "r1", "metricKey": "map5095", "metric": 0.5}
    ]
    mock_tracker.list_registered_models.return_value = [
        {"name": "m1", "aliases": {"approved": "1"}, "description": "", "latestVersion": "1"}
    ]
    mock_tracker.list_model_versions.return_value = [{"name": "m1", "version": "1", "runId": "r1"}]
    mock_tracker.list_experiments.return_value = [
        {
            "experimentId": "1",
            "name": "yolo-auto",
            "lifecycleStage": "active",
            "artifactLocation": "file:///tmp/mlruns",
        }
    ]

    register_resources(
        mcp,
        _settings(),
        {"default": mock_ssh},
        state_store,
        mock_tracker,
    )
    resources = mcp._resource_manager.list_resources()
    uris = {str(r.uri) for r in resources}
    assert "yolo://mlflow/experiments" in uris
    assert "yolo://mlflow/leaderboard" in uris
    assert "yolo://models/registry" in uris

    experiments_fn = next(
        r.fn for r in resources if str(r.uri) == "yolo://mlflow/experiments"
    )
    experiments_payload = json.loads(experiments_fn())
    assert experiments_payload["count"] == 1
    assert experiments_payload["experiments"][0]["name"] == "yolo-auto"

    leaderboard_fn = next(
        r.fn for r in resources if str(r.uri) == "yolo://mlflow/leaderboard"
    )
    payload = json.loads(leaderboard_fn())
    assert payload["metricKey"] == "map5095"
    assert payload["count"] == 1
