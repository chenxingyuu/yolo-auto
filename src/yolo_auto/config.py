from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class Settings:
    yolo_control_base_url: str
    yolo_control_bearer_token: str | None
    yolo_control_timeout_seconds: int
    feishu_webhook_url: str | None
    feishu_app_id: str | None
    feishu_app_secret: str | None
    feishu_chat_id: str | None
    feishu_report_enable: bool
    feishu_report_every_n_epochs: int
    feishu_card_img_key: str | None
    feishu_card_fallback_img_key: str | None
    primary_metric_key: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_external_url: str | None
    yolo_work_dir: str
    yolo_datasets_dir: str
    yolo_jobs_dir: str
    yolo_models_dir: str
    yolo_state_file: str
    yolo_notify_state_file: str
    watch_poll_interval_seconds: int
    watch_lock_file: str


def _env_truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required env: {name}")
    return value


def _get_env_optional(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def load_settings() -> Settings:
    # 自动加载项目根目录 .env；已存在的系统环境变量保持优先级
    dotenv_path = find_dotenv(filename=".env", usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)

    control_base_url = _get_env("YOLO_CONTROL_BASE_URL")
    control_bearer_token = _get_env_optional("YOLO_CONTROL_BEARER_TOKEN")
    control_timeout_seconds = max(1, int(_get_env("YOLO_CONTROL_TIMEOUT_SECONDS", "30")))
    feishu_webhook_url = _get_env_optional("FEISHU_WEBHOOK_URL")
    feishu_app_id = _get_env_optional("FEISHU_APP_ID")
    feishu_app_secret = _get_env_optional("FEISHU_APP_SECRET")
    feishu_chat_id = _get_env_optional("FEISHU_CHAT_ID")
    if not feishu_webhook_url and not (feishu_app_id and feishu_app_secret and feishu_chat_id):
        raise ValueError(
            "Missing Feishu config: set FEISHU_WEBHOOK_URL "
            "or (FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_CHAT_ID)"
        )
    return Settings(
        yolo_control_base_url=control_base_url,
        yolo_control_bearer_token=control_bearer_token,
        yolo_control_timeout_seconds=control_timeout_seconds,
        feishu_webhook_url=feishu_webhook_url,
        feishu_app_id=feishu_app_id,
        feishu_app_secret=feishu_app_secret,
        feishu_chat_id=feishu_chat_id,
        feishu_report_enable=_env_truthy(_get_env("FEISHU_REPORT_ENABLE", "true")),
        feishu_report_every_n_epochs=max(
            0, int(_get_env("FEISHU_REPORT_EVERY_N_EPOCHS", "5"))
        ),
        feishu_card_img_key=_get_env_optional("FEISHU_CARD_IMG_KEY"),
        feishu_card_fallback_img_key=_get_env_optional(
            "FEISHU_CARD_FALLBACK_IMG_KEY"
        ),
        primary_metric_key=_get_env("YOLO_PRIMARY_METRIC", "map5095"),
        mlflow_tracking_uri=_get_env(
            "MLFLOW_TRACKING_URI",
            "sqlite:////data/mlflow/mlflow.db",
        ),
        mlflow_experiment_name=_get_env("MLFLOW_EXPERIMENT_NAME", "yolo-auto"),
        mlflow_external_url=_get_env_optional("MLFLOW_EXTERNAL_URL"),
        yolo_work_dir=_get_env("YOLO_WORK_DIR", "/workspace/yolo-auto"),
        yolo_datasets_dir=_get_env("YOLO_DATASETS_DIR", "/workspace/datasets"),
        yolo_jobs_dir=_get_env("YOLO_JOBS_DIR", "/workspace/jobs"),
        yolo_models_dir=_get_env("YOLO_MODELS_DIR", "/workspace/models"),
        yolo_state_file=_get_env("YOLO_STATE_FILE", ".state/jobs.db"),
        yolo_notify_state_file=_get_env("YOLO_NOTIFY_STATE_FILE", ".state/notify.db"),
        watch_poll_interval_seconds=max(
            5, int(_get_env("YOLO_WATCH_POLL_INTERVAL_SECONDS", "30"))
        ),
        watch_lock_file=_get_env("YOLO_WATCH_LOCK_FILE", ".state/watch.lock"),
    )

