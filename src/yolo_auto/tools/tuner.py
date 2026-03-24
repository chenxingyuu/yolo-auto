from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from typing import Any

from yolo_auto.errors import err, ok
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobStatus
from yolo_auto.ssh_client import SSHClient
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.status import get_status
from yolo_auto.tools.training import TrainRequest, start_training
from yolo_auto.tracker import MLflowTracker


@dataclass(frozen=True)
class TuneRequest:
    env_id: str
    base_job_id: str
    model: str
    data_config_path: str
    epochs: int
    search_space: dict[str, list[float | int]]
    max_trials: int
    work_dir: str
    jobs_dir: str
    trial_timeout_seconds: int = 1800
    poll_interval_seconds: int = 10


def auto_tune(
    req: TuneRequest,
    ssh_client: SSHClient,
    notifier: FeishuNotifier,
    tracker: MLflowTracker,
    state_store: JobStateStore,
) -> dict[str, Any]:
    learning_rates = req.search_space.get("learningRate", [0.01])
    batches = req.search_space.get("batch", [16])
    img_sizes = req.search_space.get("imgSize", [640])
    candidates = list(product(learning_rates, batches, img_sizes))
    trials: list[dict[str, Any]] = []

    for trial_index, (lr, batch, img_size) in enumerate(candidates, start=1):
        if trial_index > req.max_trials:
            break
        job_id = f"{req.base_job_id}-t{trial_index}"
        train_req = TrainRequest(
            job_id=job_id,
            model=req.model,
            data_config_path=req.data_config_path,
            epochs=req.epochs,
            img_size=int(img_size),
            batch=int(batch),
            learning_rate=float(lr),
            work_dir=req.work_dir,
            jobs_dir=req.jobs_dir,
        )
        train_result = start_training(train_req, ssh_client, notifier, tracker, state_store)
        if not train_result.get("ok", False):
            trials.append(
                {
                    "jobId": job_id,
                    "runId": "",
                    "params": {"learningRate": lr, "batch": batch, "imgSize": img_size},
                    "metric": 0.0,
                    "status": JobStatus.FAILED.value,
                    "error": train_result.get("error", "start failed"),
                }
            )
            continue
        run_id = str(train_result["runId"])
        deadline = int(time.time()) + req.trial_timeout_seconds
        latest_status: dict[str, Any] = {}
        while int(time.time()) <= deadline:
            latest_status = get_status(job_id, run_id, state_store, ssh_client, tracker, notifier)
            if not latest_status.get("ok", False):
                break
            status_value = str(latest_status.get("status", JobStatus.RUNNING.value))
            if status_value in {
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.STOPPED.value,
            }:
                break
            time.sleep(req.poll_interval_seconds)

        metric_value = float(latest_status.get("metrics", {}).get("map5095", 0.0))
        trial_status = str(latest_status.get("status", JobStatus.FAILED.value))
        if int(time.time()) > deadline and trial_status == JobStatus.RUNNING.value:
            trial_status = JobStatus.FAILED.value
            latest_status = err(
                error_code="TRIAL_TIMEOUT",
                message=f"trial timeout after {req.trial_timeout_seconds}s",
                retryable=True,
                hint="提高 trial_timeout_seconds 或减少 epochs",
                payload={"jobId": job_id, "runId": run_id},
            )

        trials.append(
            {
                "jobId": job_id,
                "runId": run_id,
                "params": {"learningRate": lr, "batch": batch, "imgSize": img_size},
                "metric": metric_value,
                "status": trial_status,
                "error": latest_status.get("error") if not latest_status.get("ok", True) else None,
            }
        )
        progress_message = (
            f"[YOLO][TUNE] trial={trial_index}/{req.max_trials}, "
            f"job={job_id}, map50-95={metric_value:.4f}"
        )
        notifier.send_text(
            progress_message
        )

    succeeded_trials = [item for item in trials if item["status"] == JobStatus.COMPLETED.value]
    if not succeeded_trials:
        return err(
            error_code="NO_SUCCESSFUL_TRIALS",
            message="all tuning trials failed",
            retryable=True,
            hint="缩小 searchSpace 或先降低模型/epoch 确认可训练",
            payload={"envId": req.env_id, "baseJobId": req.base_job_id, "trials": trials},
        )
    best_trial = max(succeeded_trials, key=lambda item: item["metric"])
    return ok(
        {
            "envId": req.env_id,
            "baseJobId": req.base_job_id,
            "bestJobId": best_trial["jobId"],
            "bestMetrics": {"map5095": best_trial["metric"]},
            "bestParams": best_trial["params"],
            "trials": trials,
        }
    )

