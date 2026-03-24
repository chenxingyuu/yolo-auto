from __future__ import annotations

import time
from dataclasses import dataclass

from yolo_auto.feishu import FeishuNotifier
from yolo_auto.ssh_client import SSHClient
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
) -> dict[str, object]:
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
    pid, _ = ssh_client.execute_background(train_cmd)
    notifier.send_text(f"[YOLO] 训练已启动 job={req.job_id}, pid={pid}, run_id={run_id}")
    return {
        "jobId": req.job_id,
        "runId": run_id,
        "status": "running",
        "pid": pid,
        "paths": {
            "jobDir": job_dir,
            "logPath": log_path,
            "metricsPath": f"{job_dir}/results.csv",
            "bestPath": f"{job_dir}/weights/best.pt",
            "lastPath": f"{job_dir}/weights/last.pt",
        },
        "startedAt": int(time.time()),
    }


def stop_training(
    job_id: str,
    run_id: str,
    ssh_client: SSHClient,
    notifier: FeishuNotifier,
    tracker: MLflowTracker,
) -> dict[str, object]:
    kill_cmd = f"pkill -f \"name={job_id}\""
    _, stderr_text, exit_code = ssh_client.execute(kill_cmd)
    if exit_code != 0:
        return {"jobId": job_id, "status": "failed", "error": stderr_text.strip()}
    tracker.kill_run(run_id)
    notifier.send_text(f"[YOLO] 训练已停止 job={job_id}, run_id={run_id}")
    return {"jobId": job_id, "status": "stopped"}

