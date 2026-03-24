from __future__ import annotations

import csv
import time
from io import StringIO

from yolo_auto.errors import err, ok
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobStatus
from yolo_auto.ssh_client import SSHClient
from yolo_auto.state_store import JobStateStore
from yolo_auto.tracker import MLflowTracker


def get_status(
    job_id: str,
    run_id: str,
    state_store: JobStateStore,
    ssh_client: SSHClient,
    tracker: MLflowTracker,
    notifier: FeishuNotifier,
) -> dict[str, object]:
    now = int(time.time())
    record = state_store.get(job_id)
    if not record:
        return err(
            error_code="JOB_NOT_FOUND",
            message=f"job not found: {job_id}",
            retryable=False,
            hint="请先调用 yolo.start_training 创建任务",
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
            notifier.send_text(f"[YOLO] 任务 {job_id} 状态变更: {target.value}")
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
                "runId": run_id,
                "status": record.status.value,
                "progress": 0.0,
            }
        )

    last_row = rows[-1]
    epoch = int(float(last_row.get("epoch", "0")))
    map50 = float(last_row.get("metrics/mAP50(B)", "0"))
    map5095 = float(last_row.get("metrics/mAP50-95(B)", "0"))
    precision = float(last_row.get("metrics/precision(B)", "0"))
    recall = float(last_row.get("metrics/recall(B)", "0"))
    loss = float(last_row.get("train/box_loss", "0"))

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
    state_store.mark_metrics(job_id, now)

    if not ssh_client.process_alive(record.pid):
        updated = state_store.update_status(job_id, JobStatus.COMPLETED, now)
        tracker.finish_run(updated.run_id, updated.paths.get("bestPath"))
        if updated.last_notified_state != JobStatus.COMPLETED:
            notifier.send_text(f"[YOLO] 任务完成 job={job_id}, mAP50-95={map5095:.4f}")
            state_store.mark_notified(job_id, JobStatus.COMPLETED, now)
        status_value = updated.status.value
    else:
        status_value = JobStatus.RUNNING.value

    return ok(
        {
            "jobId": job_id,
            "runId": effective_run_id,
            "status": status_value,
            "progress": round(min(1.0, epoch / 100), 4),
            "metrics": {
                "epoch": epoch,
                "loss": loss,
                "map50": map50,
                "map5095": map5095,
                "precision": precision,
                "recall": recall,
            },
            "artifacts": {
                "best": record.paths.get("bestPath", ""),
                "last": record.paths.get("lastPath", ""),
            },
        }
    )

