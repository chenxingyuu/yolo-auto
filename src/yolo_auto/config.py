from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class Settings:
    yolo_ssh_host: str
    yolo_ssh_port: int
    yolo_ssh_user: str
    yolo_ssh_key_path: str
    feishu_webhook_url: str
    feishu_report_enable: bool
    feishu_report_every_n_epochs: int
    feishu_message_mode: str
    primary_metric_key: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    yolo_work_dir: str
    yolo_datasets_dir: str
    yolo_jobs_dir: str
    yolo_state_file: str
    watch_poll_interval_seconds: int
    watch_lock_file: str


def _env_truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required env: {name}")
    return value


def load_settings() -> Settings:
    # 自动加载项目根目录 .env；已存在的系统环境变量保持优先级
    dotenv_path = find_dotenv(filename=".env", usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)

    return Settings(
        yolo_ssh_host=_get_env("YOLO_SSH_HOST"),
        yolo_ssh_port=int(_get_env("YOLO_SSH_PORT", "2222")),
        yolo_ssh_user=_get_env("YOLO_SSH_USER"),
        yolo_ssh_key_path=_get_env("YOLO_SSH_KEY_PATH"),
        feishu_webhook_url=_get_env("FEISHU_WEBHOOK_URL"),
        feishu_report_enable=_env_truthy(_get_env("FEISHU_REPORT_ENABLE", "true")),
        feishu_report_every_n_epochs=max(
            0, int(_get_env("FEISHU_REPORT_EVERY_N_EPOCHS", "5"))
        ),
        feishu_message_mode=_get_env("FEISHU_MESSAGE_MODE", "text").lower(),
        primary_metric_key=_get_env("YOLO_PRIMARY_METRIC", "map5095"),
        mlflow_tracking_uri=_get_env("MLFLOW_TRACKING_URI", "./mlruns"),
        mlflow_experiment_name=_get_env("MLFLOW_EXPERIMENT_NAME", "yolo-auto"),
        yolo_work_dir=_get_env("YOLO_WORK_DIR", "/workspace/yolo-auto"),
        yolo_datasets_dir=_get_env("YOLO_DATASETS_DIR", "/workspace/datasets"),
        yolo_jobs_dir=_get_env("YOLO_JOBS_DIR", "/workspace/jobs"),
        yolo_state_file=_get_env("YOLO_STATE_FILE", ".state/jobs.json"),
        watch_poll_interval_seconds=max(
            5, int(_get_env("YOLO_WATCH_POLL_INTERVAL_SECONDS", "30"))
        ),
        watch_lock_file=_get_env("YOLO_WATCH_LOCK_FILE", ".state/watch.lock"),
    )

