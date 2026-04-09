from __future__ import annotations

# 后台 Watch Worker。
#
# 该脚本会不断轮询本地 `yolo_state_file`（默认 `.state/jobs.db`）里的任务状态：
# - 对状态为 `RUNNING` 的任务，调用 `get_status(...)` 拉取训练结果/指标，
#   并按配置向飞书推送训练里程碑。
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
import fcntl
import logging
import time
from pathlib import Path
from types import TracebackType

from yolo_auto.config import load_settings
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobStatus
from yolo_auto.remote_control import HttpControlClient, RemoteControlConfig
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.status import get_status

logger = logging.getLogger(__name__)


def _format_ok_status_line(result: dict[str, object]) -> str:
    """将一次成功的 get_status 返回体整理成单行诊断信息（不含密钥）。"""
    status = result.get("status", "?")
    progress = result.get("progress", "?")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return f"status={status} progress={progress}"
    epoch = metrics.get("epoch", "?")
    parts = [
        f"status={status}",
        f"progress={progress}",
        f"epoch={epoch}",
    ]
    for key in ("loss", "map5095", "map50", "primaryMetric"):
        val = metrics.get(key)
        if val is not None:
            parts.append(f"{key}={val}")
    return " ".join(parts)


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
            logger.debug("released watch lock path=%s", self._path)


def main() -> None:
    """轮询本地任务状态，并对 `RUNNING` 任务执行 `get_status(...)`。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    settings = load_settings()
    control_client = HttpControlClient(
        RemoteControlConfig(
            base_url=settings.yolo_control_base_url,
            bearer_token=settings.yolo_control_bearer_token,
            timeout_seconds=settings.yolo_control_timeout_seconds,
        )
    )
    notifier = FeishuNotifier(
        webhook_url=settings.feishu_webhook_url,
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
        chat_id=settings.feishu_chat_id,
    )
    store = JobStateStore(settings.yolo_state_file)
    lock_path = Path(settings.watch_lock_file)
    logger.info(
        "yolo-auto-watch starting state_file=%s lock_file=%s poll_interval_s=%s "
        "feishu_report=%s every_n_epochs=%s primary_metric=%s",
        settings.yolo_state_file,
        settings.watch_lock_file,
        settings.watch_poll_interval_seconds,
        settings.feishu_report_enable,
        settings.feishu_report_every_n_epochs,
        settings.primary_metric_key,
    )
    poll_seq = 0
    while True:
        poll_seq += 1
        logger.debug(
            "poll #%s waiting for watch lock path=%s",
            poll_seq,
            lock_path,
        )
        with LockFile(lock_path):
            logger.debug("poll #%s acquired watch lock", poll_seq)
            jobs = store.list_all()
            running_jobs = [j for j in jobs if j.status == JobStatus.RUNNING]
            by_status: dict[str, int] = {}
            for j in jobs:
                by_status[j.status.value] = by_status.get(j.status.value, 0) + 1
            logger.debug(
                "poll #%s store snapshot jobs=%s by_status=%s running_ids=%s",
                poll_seq,
                len(jobs),
                by_status,
                [j.job_id for j in running_jobs],
            )
            if not running_jobs:
                logger.debug(
                    "poll #%s no RUNNING jobs sleep_s=%s",
                    poll_seq,
                    settings.watch_poll_interval_seconds,
                )
            else:
                logger.info(
                    "poll #%s refreshing %s RUNNING job(s): %s",
                    poll_seq,
                    len(running_jobs),
                    [j.job_id for j in running_jobs],
                )
            for job in jobs:
                if job.status != JobStatus.RUNNING:
                    continue
                logger.info(
                    "get_status begin poll=#%s job_id=%s run_id=%s env_id=%r "
                    "pid=%s metricsPath=%s",
                    poll_seq,
                    job.job_id,
                    job.run_id,
                    job.env_id,
                    job.pid,
                    (job.paths.get("metricsPath") or "").strip() or "(missing)",
                )
                try:
                    t0 = time.monotonic()
                    result = get_status(
                        job.job_id,
                        job.run_id,
                        store,
                        None,
                        notifier,
                        feishu_report_enable=settings.feishu_report_enable,
                        feishu_report_every_n_epochs=settings.feishu_report_every_n_epochs,
                        primary_metric_key=settings.primary_metric_key,
                        feishu_card_img_key=settings.feishu_card_img_key,
                        feishu_card_fallback_img_key=settings.feishu_card_fallback_img_key,
                        control_client=control_client,
                    )
                    elapsed_ms = int((time.monotonic() - t0) * 1000)
                    if not bool(result.get("ok", True)):
                        logger.warning(
                            "get_status failed poll=#%s job_id=%s elapsed_ms=%s "
                            "errorCode=%s message=%s hint=%s retryable=%s",
                            poll_seq,
                            job.job_id,
                            elapsed_ms,
                            result.get("errorCode"),
                            result.get("error"),
                            result.get("hint"),
                            result.get("retryable"),
                        )
                    else:
                        logger.info(
                            "get_status ok poll=#%s job_id=%s elapsed_ms=%s %s",
                            poll_seq,
                            job.job_id,
                            elapsed_ms,
                            _format_ok_status_line(result),
                        )
                except Exception:
                    logger.exception(
                        "get_status raised poll=#%s job_id=%s run_id=%s env_id=%r",
                        poll_seq,
                        job.job_id,
                        job.run_id,
                        job.env_id,
                    )
        logger.debug(
            "poll #%s sleeping %ss",
            poll_seq,
            settings.watch_poll_interval_seconds,
        )
        time.sleep(settings.watch_poll_interval_seconds)


if __name__ == "__main__":
    main()
