from __future__ import annotations

from unittest.mock import MagicMock, patch

from yolo_auto.config import Settings
from yolo_auto.tools.check_config import check_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    defaults = dict(
        yolo_control_base_url="http://127.0.0.1:18080",
        yolo_control_bearer_token=None,
        yolo_control_timeout_seconds=30,
        feishu_webhook_url="https://example.com/hook",
        feishu_app_id=None,
        feishu_app_secret=None,
        feishu_chat_id=None,
        feishu_report_enable=True,
        feishu_report_every_n_epochs=5,
        feishu_card_img_key=None,
        feishu_card_fallback_img_key=None,
        primary_metric_key="map5095",
        mlflow_tracking_uri="sqlite:////workspace/mlflow.db",
        mlflow_experiment_name="yolo-auto",
        mlflow_external_url=None,
        yolo_work_dir="/workspace/yolo-auto",
        yolo_datasets_dir="/workspace/datasets",
        yolo_jobs_dir="/workspace/jobs",
        yolo_models_dir="/workspace/models",
        yolo_state_file=".state/jobs.db",
        yolo_notify_state_file=".state/notify.db",
        watch_poll_interval_seconds=30,
        watch_lock_file=".state/watch.lock",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _ok_client() -> MagicMock:
    client = MagicMock()
    client.health_check.return_value = (True, "http://127.0.0.1:18080 可达")
    return client


def _fail_client(error_code: str = "REMOTE_UNREACHABLE") -> MagicMock:
    client = MagicMock()
    client.health_check.return_value = (False, "无法连接训练控制面")
    return client


# ---------------------------------------------------------------------------
# Top-level result structure
# ---------------------------------------------------------------------------

def test_returns_ok_envelope():
    result = check_config(_make_settings(), _ok_client())
    assert result["ok"] is True


def test_required_and_optional_keys_present():
    result = check_config(_make_settings(), _ok_client())
    assert "required" in result
    assert "optional" in result
    assert "capabilities" in result
    assert "paths" in result
    assert "nextSteps" in result


def test_each_check_item_has_expected_fields():
    result = check_config(_make_settings(), _ok_client())
    for item in result["required"] + result["optional"]:
        assert {"name", "var", "status", "detail"} == set(item.keys())


# ---------------------------------------------------------------------------
# allRequiredOk
# ---------------------------------------------------------------------------

def test_all_required_ok_when_control_plane_reachable_and_feishu_configured():
    result = check_config(_make_settings(), _ok_client())
    assert result["allRequiredOk"] is True


def test_all_required_ok_false_when_control_plane_unreachable():
    result = check_config(_make_settings(), _fail_client())
    assert result["allRequiredOk"] is False


# ---------------------------------------------------------------------------
# Control plane check
# ---------------------------------------------------------------------------

def test_control_plane_ok_status():
    result = check_config(_make_settings(), _ok_client())
    cp = next(i for i in result["required"] if "控制面" in i["name"])
    assert cp["status"] == "ok"


def test_control_plane_fail_status_when_unreachable():
    result = check_config(_make_settings(), _fail_client())
    cp = next(i for i in result["required"] if "控制面" in i["name"])
    assert cp["status"] == "fail"


# ---------------------------------------------------------------------------
# Feishu check
# ---------------------------------------------------------------------------

def test_feishu_ok_with_webhook():
    settings = _make_settings(feishu_webhook_url="https://example.com/hook")
    result = check_config(settings, _ok_client())
    fs = next(i for i in result["required"] if "飞书" in i["name"])
    assert fs["status"] == "ok"
    assert "Webhook" in fs["detail"]


def test_feishu_ok_with_app_bot_preferred_over_webhook():
    settings = _make_settings(
        feishu_webhook_url="https://example.com/hook",
        feishu_app_id="app123",
        feishu_app_secret="secret",
        feishu_chat_id="chat456",
    )
    result = check_config(settings, _ok_client())
    fs = next(i for i in result["required"] if "飞书" in i["name"])
    assert fs["status"] == "ok"
    assert "应用机器人" in fs["detail"]


def test_feishu_fail_when_neither_configured():
    settings = _make_settings(feishu_webhook_url=None)
    result = check_config(settings, _ok_client())
    fs = next(i for i in result["required"] if "飞书" in i["name"])
    assert fs["status"] == "fail"


# ---------------------------------------------------------------------------
# MLflow check
# ---------------------------------------------------------------------------

def test_mlflow_warn_when_sqlite_file_missing(tmp_path):
    uri = f"sqlite:////{tmp_path}/nonexistent.db"
    settings = _make_settings(mlflow_tracking_uri=uri)
    result = check_config(settings, _ok_client(), tracker=None)
    ml = next(i for i in result["optional"] if "MLflow" in i["name"])
    assert ml["status"] == "warn"
    assert "暂不存在" in ml["detail"]


def test_mlflow_ok_when_tracker_ping_succeeds(tmp_path):
    tracker = MagicMock()
    tracker.ping.return_value = (True, "可访问：yolo-auto")
    # Use a uri that points to an existing file to bypass the sqlite path check
    db = tmp_path / "mlflow.db"
    db.touch()
    settings = _make_settings(mlflow_tracking_uri=f"sqlite:////{db}")
    result = check_config(settings, _ok_client(), tracker=tracker)
    ml = next(i for i in result["optional"] if "MLflow" in i["name"])
    assert ml["status"] == "ok"


def test_mlflow_warn_when_tracker_ping_fails(tmp_path):
    tracker = MagicMock()
    tracker.ping.return_value = (False, "访问失败：connection refused")
    db = tmp_path / "mlflow.db"
    db.touch()
    settings = _make_settings(mlflow_tracking_uri=f"sqlite:////{db}")
    result = check_config(settings, _ok_client(), tracker=tracker)
    ml = next(i for i in result["optional"] if "MLflow" in i["name"])
    assert ml["status"] == "warn"


# ---------------------------------------------------------------------------
# MinIO check
# ---------------------------------------------------------------------------

def test_minio_skip_when_not_configured():
    with patch.dict("os.environ", {}, clear=False):
        # Ensure vars are unset
        import os
        os.environ.pop("YOLO_MINIO_ALIAS", None)
        os.environ.pop("YOLO_MINIO_EXPORT_BUCKET", None)
        result = check_config(_make_settings(), _ok_client())
    minio = next(i for i in result["optional"] if "MinIO" in i["name"])
    assert minio["status"] == "skip"


def test_minio_ok_when_both_vars_set():
    env = {"YOLO_MINIO_ALIAS": "minio", "YOLO_MINIO_EXPORT_BUCKET": "cvat-export"}
    with patch.dict("os.environ", env):
        result = check_config(_make_settings(), _ok_client())
    minio = next(i for i in result["optional"] if "MinIO" in i["name"])
    assert minio["status"] == "ok"
    assert "minio" in minio["detail"]


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

def test_capabilities_all_true_when_everything_configured(tmp_path):
    tracker = MagicMock()
    tracker.ping.return_value = (True, "可访问：yolo-auto")
    db = tmp_path / "mlflow.db"
    db.touch()
    settings = _make_settings(mlflow_tracking_uri=f"sqlite:////{db}")
    env = {"YOLO_MINIO_ALIAS": "minio", "YOLO_MINIO_EXPORT_BUCKET": "cvat-export"}
    with patch.dict("os.environ", env):
        result = check_config(settings, _ok_client(), tracker=tracker)
    caps = result["capabilities"]
    assert caps["canTrain"] is True
    assert caps["canNotify"] is True
    assert caps["canCompareExperiments"] is True
    assert caps["canSyncDataset"] is True


def test_capabilities_cannot_train_when_control_plane_down():
    result = check_config(_make_settings(), _fail_client())
    assert result["capabilities"]["canTrain"] is False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def test_paths_contain_all_expected_keys():
    result = check_config(_make_settings(), _ok_client())
    paths = result["paths"]
    assert "workDir" in paths
    assert "datasetsDir" in paths
    assert "jobsDir" in paths
    assert "modelsDir" in paths
    assert "stateFile" in paths


# ---------------------------------------------------------------------------
# nextSteps
# ---------------------------------------------------------------------------

def test_next_steps_include_required_fix_when_control_plane_down():
    result = check_config(_make_settings(), _fail_client())
    steps = result["nextSteps"]
    assert any("[必须]" in s for s in steps)


def test_next_steps_empty_when_everything_ok(tmp_path):
    tracker = MagicMock()
    tracker.ping.return_value = (True, "可访问：yolo-auto")
    db = tmp_path / "mlflow.db"
    db.touch()
    settings = _make_settings(mlflow_tracking_uri=f"sqlite:////{db}")
    env = {"YOLO_MINIO_ALIAS": "minio", "YOLO_MINIO_EXPORT_BUCKET": "cvat-export"}
    with patch.dict("os.environ", env):
        result = check_config(settings, _ok_client(), tracker=tracker)
    # No required failures, no warns — only optional skips are gone too
    steps = result["nextSteps"]
    assert not any("[必须]" in s or "[建议]" in s for s in steps)
