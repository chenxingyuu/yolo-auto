from __future__ import annotations

import csv
import time
from io import StringIO

from yolo_auto.errors import err, ok
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobStatus
from yolo_auto.ssh_client import SSHClient
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.metrics_csv import metric_value_from_parsed, parse_training_row
from yolo_auto.tracker import MLflowTracker


def get_status(
    job_id: str,
    run_id: str,
    state_store: JobStateStore,
    ssh_client: SSHClient,
    tracker: MLflowTracker,
    notifier: FeishuNotifier,
    *,
    feishu_report_enable: bool = True,
    feishu_report_every_n_epochs: int = 5,
    primary_metric_key: str = "map5095",
) -> dict[str, object]:
    now = int(time.time())
    record = state_store.get(job_id)
    if not record:
        return err(
            error_code="JOB_NOT_FOUND",
            message=f"job not found: {job_id}",
            retryable=False,
            hint="请先调用 yolo_start_training 创建任务",
            payload={"jobId": job_id},
        )

    effective_run_id = record.run_id or run_id
    metrics_path = record.paths.get("metricsPath", "")
    content, stderr_text, exit_code = ssh_client.execute(f"cat {metrics_path}")
    if exit_code != 0:
        if ssh_client.process_alive(record.pid):
            return ok(
                {
                    "jobId": job_id,
                    "runId": effective_run_id,
                    "status": record.status.value,
                    "progress": 0.0,
                    "error": stderr_text.strip() or "results.csv not ready",
                }
            )
        log_path = record.paths.get("logPath", "")
        log_content, _, _ = ssh_client.tail_file(log_path, lines=80)
        failed = "error" in log_content.lower() or "traceback" in log_content.lower()
        target = JobStatus.FAILED if failed else JobStatus.COMPLETED
        updated = state_store.update_status(job_id, target, now)
        if updated.last_notified_state != target:
            title = "[YOLO] 状态变更"
            body = f"job={job_id}\n状态={target.value}\nrunId={effective_run_id}"
            notifier.send_training_update(title, body)
            state_store.mark_notified(job_id, target, now)
        if target == JobStatus.FAILED:
            return err(
                error_code="TRAIN_FAILED",
                message="training process exited with errors",
                retryable=False,
                hint="查看远程 train.log 定位错误并修复数据或参数",
                payload=updated.to_dict(),
            )
        tracker.finish_run(effective_run_id, updated.paths.get("bestPath"))
        return ok(updated.to_dict())

    rows = list(csv.DictReader(StringIO(content)))
    if not rows:
        return ok(
            {
                "jobId": job_id,
                "runId": effective_run_id,
                "status": record.status.value,
                "progress": 0.0,
            }
        )

    last_row = rows[-1]
    parsed = parse_training_row(last_row)
    epoch = int(parsed["epoch"])
    map50 = float(parsed["map50"])
    map5095 = float(parsed["map5095"])
    precision = float(parsed["precision"])
    recall = float(parsed["recall"])
    loss = float(parsed["loss"])
    primary_value = metric_value_from_parsed(parsed, primary_metric_key)

    tracker.log_epoch(
        run_id=effective_run_id,
        metrics={
            "loss": loss,
            "map50": map50,
            "map5095": map5095,
            "precision": precision,
            "recall": recall,
        },
        step=epoch,
    )
    record = state_store.mark_metrics(job_id, now)
    total_epochs = max(record.train_epochs or 100, 1)
    progress = round(min(1.0, epoch / total_epochs), 4)

    if ssh_client.process_alive(record.pid) and feishu_report_enable:
        n = feishu_report_every_n_epochs
        if n > 0 and epoch > 0 and epoch >= record.last_reported_epoch + n:
            title = "[YOLO] 训练里程碑"
            body = (
                f"job={job_id}\n"
                f"epoch={epoch}/{total_epochs}\n"
                f"{primary_metric_key}={primary_value:.4f}\n"
                f"runId={effective_run_id}"
            )
            notifier.send_training_update(title, body)
            record = state_store.mark_milestone_epoch(job_id, epoch, now)

    if not ssh_client.process_alive(record.pid):
        updated = state_store.update_status(job_id, JobStatus.COMPLETED, now)
        tracker.finish_run(effective_run_id, updated.paths.get("bestPath"))
        if updated.last_notified_state != JobStatus.COMPLETED:
            title = "[YOLO] 任务完成"
            body = (
                f"job={job_id}\n"
                f"mAP50-95={map5095:.4f}\n"
                f"runId={effective_run_id}"
            )
            notifier.send_training_update(title, body)
            state_store.mark_notified(job_id, JobStatus.COMPLETED, now)
        status_value = updated.status.value
    else:
        status_value = JobStatus.RUNNING.value

    return ok(
        {
            "jobId": job_id,
            "runId": effective_run_id,
            "status": status_value,
            "progress": progress,
            "metrics": {
                "epoch": epoch,
                "loss": loss,
                "map50": map50,
                "map5095": map5095,
                "precision": precision,
                "recall": recall,
                "primaryMetric": primary_value,
            },
            "artifacts": {
                "best": record.paths.get("bestPath", ""),
                "last": record.paths.get("lastPath", ""),
            },
        }
    )
