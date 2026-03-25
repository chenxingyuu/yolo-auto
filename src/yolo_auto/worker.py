from __future__ import annotations

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
    settings = load_settings()
    ssh = SSHClient(
        SSHConfig(
            host=settings.yolo_ssh_host,
            port=settings.yolo_ssh_port,
            user=settings.yolo_ssh_user,
            key_path=settings.yolo_ssh_key_path,
        )
    )
    notifier = FeishuNotifier(settings.feishu_webhook_url, settings.feishu_message_mode)
    tracker = MLflowTracker(
        TrackerConfig(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
        )
    )
    store = JobStateStore(settings.yolo_state_file)
    lock_path = Path(settings.watch_lock_file)
    while True:
        with LockFile(lock_path):
            for job in store.list_all():
                if job.status != JobStatus.RUNNING:
                    continue
                get_status(
                    job.job_id,
                    job.run_id,
                    store,
                    ssh,
                    tracker,
                    notifier,
                    feishu_report_enable=settings.feishu_report_enable,
                    feishu_report_every_n_epochs=settings.feishu_report_every_n_epochs,
                    primary_metric_key=settings.primary_metric_key,
                )
        time.sleep(settings.watch_poll_interval_seconds)


if __name__ == "__main__":
    main()
