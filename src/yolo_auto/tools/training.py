from __future__ import annotations

import time
from dataclasses import dataclass

from yolo_auto.errors import err, ok
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.ssh_client import SSHClient
from yolo_auto.state_store import JobStateStore
from yolo_auto.tracker import MLflowTracker


@dataclass(frozen=True)
class TrainRequest:
    job_id: str
    model: str
    data_config_path: str
    epochs: int
    img_size: int
    batch: int
    learning_rate: float
    work_dir: str
    jobs_dir: str


def start_training(
    req: TrainRequest,
    ssh_client: SSHClient,
    notifier: FeishuNotifier,
    tracker: MLflowTracker,
    state_store: JobStateStore,
) -> dict[str, object]:
    now = int(time.time())
    existing = state_store.get(req.job_id)
    if existing and existing.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
        return ok(existing.to_dict())

    run_id = tracker.start_run(
        job_id=req.job_id,
        config={
            "model": req.model,
            "data_config_path": req.data_config_path,
            "epochs": req.epochs,
            "img_size": req.img_size,
            "batch": req.batch,
            "learning_rate": req.learning_rate,
        },
    )
    job_dir = f"{req.jobs_dir}/{req.job_id}"
    log_path = f"{job_dir}/train.log"
    train_cmd = (
        f"mkdir -p {job_dir} && cd {req.work_dir} && "
        f"yolo detect train model={req.model} data={req.data_config_path} "
        f"epochs={req.epochs} imgsz={req.img_size} batch={req.batch} lr0={req.learning_rate} "
        f"project={req.jobs_dir} name={req.job_id} > {log_path} 2>&1"
    )
    try:
        pid, _ = ssh_client.execute_background(train_cmd)
    except Exception as exc:
        return err(
            error_code="START_FAILED",
            message=str(exc),
            retryable=True,
            hint="检查 SSH 连通性、数据路径和远程 yolo 命令可用性",
            payload={"jobId": req.job_id},
        )
    record = JobRecord(
        job_id=req.job_id,
        run_id=run_id,
        status=JobStatus.RUNNING,
        pid=str(pid),
        paths={
            "jobDir": job_dir,
            "logPath": log_path,
            "metricsPath": f"{job_dir}/results.csv",
            "bestPath": f"{job_dir}/weights/best.pt",
            "lastPath": f"{job_dir}/weights/last.pt",
        },
        created_at=now if not existing else existing.created_at,
        updated_at=now,
        last_notified_state=existing.last_notified_state if existing else None,
        last_metrics_at=existing.last_metrics_at if existing else None,
        train_epochs=req.epochs,
        last_reported_epoch=0,
    )
    state_store.upsert(record)
    notifier.send_training_update(
        "[YOLO] 训练已启动",
        f"job={req.job_id}\npid={pid}\nrunId={run_id}\nepochs={req.epochs}",
    )
    return ok(record.to_dict())


def stop_training(
    job_id: str,
    run_id: str,
    ssh_client: SSHClient,
    notifier: FeishuNotifier,
    tracker: MLflowTracker,
    state_store: JobStateStore,
) -> dict[str, object]:
    now = int(time.time())
    record = state_store.get(job_id)
    effective_run_id = record.run_id if record else run_id
    if record and record.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED}:
        return ok(record.to_dict())

    pid = record.pid if record else ""
    if pid and not ssh_client.process_alive(pid):
        updated = state_store.update_status(job_id, JobStatus.STOPPED, now) if record else None
        tracker.kill_run(effective_run_id)
        if updated:
            return ok(updated.to_dict())
        return ok({"jobId": job_id, "runId": effective_run_id, "status": JobStatus.STOPPED.value})

    kill_cmd = f"pkill -f \"name={job_id}\""
    _, stderr_text, exit_code = ssh_client.execute(kill_cmd)
    if exit_code != 0:
        return err(
            error_code="STOP_FAILED",
            message=stderr_text.strip() or "unable to stop training process",
            retryable=True,
            hint="确认 jobId 是否正确，或稍后重试 stop",
            payload={"jobId": job_id, "runId": effective_run_id},
        )

    tracker.kill_run(effective_run_id)
    if record:
        updated = state_store.update_status(job_id, JobStatus.STOPPED, now)
        if updated.last_notified_state != JobStatus.STOPPED:
            notifier.send_training_update(
                "[YOLO] 训练已停止",
                f"job={job_id}\nrunId={effective_run_id}",
            )
            state_store.mark_notified(job_id, JobStatus.STOPPED, now)
        return ok(updated.to_dict())
    notifier.send_training_update(
        "[YOLO] 训练已停止",
        f"job={job_id}\nrunId={effective_run_id}",
    )
    return ok({"jobId": job_id, "runId": effective_run_id, "status": JobStatus.STOPPED.value})

