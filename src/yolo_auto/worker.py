from __future__ import annotations

# 后台 Watch Worker。
#
# 该脚本会不断轮询本地 `yolo_state_file`（默认 `.state/jobs.db`）里的任务状态：
# - 对状态为 `RUNNING` 的任务，调用 `get_status(...)` 拉取训练结果/指标，
#   更新 MLflow，并按配置向飞书推送训练里程碑。
#
# 建议同一时间只运行一个 Worker；本脚本使用 `YOLO_WATCH_LOCK_FILE` 做进程间排他锁
# （基于 `fcntl.flock`）。
#
# 启动方式（任选其一）：
# - `uv run yolo-auto-watch`
# - `uv run python -m yolo_auto.worker`
#
# 常用环境变量（来自 `.env`，由 `load_settings()` 读取）：
# - `YOLO_STATE_FILE`：任务状态文件路径（Worker 从这里读取 `RUNNING` 任务）
# - `YOLO_WATCH_LOCK_FILE`：Worker 锁文件路径（避免多实例抢锁）
# - `YOLO_WATCH_POLL_INTERVAL_SECONDS`：轮询间隔（秒）
# - `FEISHU_WEBHOOK_URL`：飞书机器人 Webhook
# - `FEISHU_REPORT_ENABLE` / `FEISHU_REPORT_EVERY_N_EPOCHS`：是否启用里程碑推送与推送频率
# - `YOLO_PRIMARY_METRIC`：主指标名（写入里程碑文案）
# - `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_NAME`：MLflow 追踪配置
#
# SSH 连接配置：
# - 优先使用 `YOLO_SSH_ENVS`（JSON 对象，支持为不同 `env_id` 配置不同主机）
# - 或回退使用 `YOLO_SSH_HOST` / `YOLO_SSH_PORT` / `YOLO_SSH_USER`
#   / `YOLO_SSH_KEY_PATH`（作为 `default` 环境）
import fcntl
import time
from pathlib import Path
from types import TracebackType

from yolo_auto.config import load_settings
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobStatus
from yolo_auto.ssh_client import SSHClient, SSHConfig
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.status import get_status
from yolo_auto.tracker import MLflowTracker, TrackerConfig


class LockFile:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = None

    def __enter__(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._path, "a+", encoding="utf-8")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()


def main() -> None:
    """轮询本地任务状态，并对 `RUNNING` 任务执行 `get_status(...)`。"""
    settings = load_settings()
    ssh_by_env: dict[str, SSHClient] = {}
    for env_id, ssh_env in settings.yolo_ssh_envs.items():
        ssh_by_env[env_id] = SSHClient(
            SSHConfig(
                host=ssh_env.host,
                port=ssh_env.port,
                user=ssh_env.user,
                key_path=ssh_env.key_path,
            )
        )
    ssh_default = ssh_by_env["default"]
    notifier = FeishuNotifier(
        webhook_url=settings.feishu_webhook_url,
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
        chat_id=settings.feishu_chat_id,
    )
    tracker = MLflowTracker(
        TrackerConfig(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            external_url=settings.mlflow_external_url,
        )
    )
    store = JobStateStore(settings.yolo_state_file)
    lock_path = Path(settings.watch_lock_file)
    while True:
        with LockFile(lock_path):
            for job in store.list_all():
                if job.status != JobStatus.RUNNING:
                    continue
                ssh_client = ssh_by_env.get(job.env_id, ssh_default)
                get_status(
                    job.job_id,
                    job.run_id,
                    store,
                    ssh_client,
                    tracker,
                    notifier,
                    feishu_report_enable=settings.feishu_report_enable,
                    feishu_report_every_n_epochs=settings.feishu_report_every_n_epochs,
                    primary_metric_key=settings.primary_metric_key,
                )
        time.sleep(settings.watch_poll_interval_seconds)


if __name__ == "__main__":
    main()
