from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class Settings:
    yolo_ssh_host: str
    yolo_ssh_port: int
    yolo_ssh_user: str
    yolo_ssh_key_path: str
    yolo_ssh_envs: dict[str, SSHEnv]
    feishu_webhook_url: str | None
    feishu_app_id: str | None
    feishu_app_secret: str | None
    feishu_chat_id: str | None
    feishu_report_enable: bool
    feishu_report_every_n_epochs: int
    primary_metric_key: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_external_url: str | None
    yolo_work_dir: str
    yolo_datasets_dir: str
    yolo_jobs_dir: str
    yolo_models_dir: str
    yolo_state_file: str
    watch_poll_interval_seconds: int
    watch_lock_file: str
    cvat_url: str | None
    cvat_token: str | None
    cvat_org_slug: str | None


@dataclass(frozen=True)
class SSHEnv:
    host: str
    port: int
    user: str
    key_path: str


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

    raw_envs = os.getenv("YOLO_SSH_ENVS", "").strip()
    if raw_envs:
        parsed = json.loads(raw_envs)
        if not isinstance(parsed, dict):
            raise ValueError("YOLO_SSH_ENVS must be a JSON object")

        envs: dict[str, SSHEnv] = {}
        for env_id, cfg in parsed.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"YOLO_SSH_ENVS[{env_id}] must be an object")
            host = cfg.get("host")
            user = cfg.get("user")
            key_path = cfg.get("keyPath") or cfg.get("key_path")
            port = cfg.get("port", 2222)
            if not host or not user or not key_path:
                raise ValueError(f"YOLO_SSH_ENVS[{env_id}] missing host/user/keyPath")
            envs[str(env_id)] = SSHEnv(
                host=str(host),
                port=int(port),
                user=str(user),
                key_path=str(key_path),
            )

        if "default" not in envs:
            # 如果用户没有提供 default，则尝试回退到旧版 YOLO_SSH_* 变量。
            default_host = _get_env("YOLO_SSH_HOST")
            default_port = int(_get_env("YOLO_SSH_PORT", "2222"))
            default_user = _get_env("YOLO_SSH_USER")
            default_key_path = _get_env("YOLO_SSH_KEY_PATH")
            envs["default"] = SSHEnv(
                host=default_host,
                port=default_port,
                user=default_user,
                key_path=default_key_path,
            )
    else:
        envs = {
            "default": SSHEnv(
                host=_get_env("YOLO_SSH_HOST"),
                port=int(_get_env("YOLO_SSH_PORT", "2222")),
                user=_get_env("YOLO_SSH_USER"),
                key_path=_get_env("YOLO_SSH_KEY_PATH"),
            )
        }

    default_env = envs["default"]
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
        yolo_ssh_host=default_env.host,
        yolo_ssh_port=default_env.port,
        yolo_ssh_user=default_env.user,
        yolo_ssh_key_path=default_env.key_path,
        yolo_ssh_envs=envs,
        feishu_webhook_url=feishu_webhook_url,
        feishu_app_id=feishu_app_id,
        feishu_app_secret=feishu_app_secret,
        feishu_chat_id=feishu_chat_id,
        feishu_report_enable=_env_truthy(_get_env("FEISHU_REPORT_ENABLE", "true")),
        feishu_report_every_n_epochs=max(
            0, int(_get_env("FEISHU_REPORT_EVERY_N_EPOCHS", "5"))
        ),
        primary_metric_key=_get_env("YOLO_PRIMARY_METRIC", "map5095"),
        mlflow_tracking_uri=_get_env("MLFLOW_TRACKING_URI", "sqlite:////data/mlflow/mlflow.db"),
        mlflow_experiment_name=_get_env("MLFLOW_EXPERIMENT_NAME", "yolo-auto"),
        mlflow_external_url=_get_env_optional("MLFLOW_EXTERNAL_URL"),
        yolo_work_dir=_get_env("YOLO_WORK_DIR", "/workspace/yolo-auto"),
        yolo_datasets_dir=_get_env("YOLO_DATASETS_DIR", "/workspace/datasets"),
        yolo_jobs_dir=_get_env("YOLO_JOBS_DIR", "/workspace/jobs"),
        yolo_models_dir=_get_env("YOLO_MODELS_DIR", "/workspace/models"),
        yolo_state_file=_get_env("YOLO_STATE_FILE", ".state/jobs.json"),
        watch_poll_interval_seconds=max(
            5, int(_get_env("YOLO_WATCH_POLL_INTERVAL_SECONDS", "30"))
        ),
        watch_lock_file=_get_env("YOLO_WATCH_LOCK_FILE", ".state/watch.lock"),
        cvat_url=_get_env_optional("CVAT_URL"),
        cvat_token=_get_env_optional("CVAT_TOKEN"),
        cvat_org_slug=_get_env_optional("CVAT_ORG_SLUG"),
    )

