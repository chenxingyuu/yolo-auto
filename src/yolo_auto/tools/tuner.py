from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from yolo_auto.feishu import FeishuNotifier
from yolo_auto.ssh_client import SSHClient
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


def auto_tune(
    req: TuneRequest,
    ssh_client: SSHClient,
    notifier: FeishuNotifier,
    tracker: MLflowTracker,
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
        train_result = start_training(train_req, ssh_client, notifier, tracker)
        run_id = str(train_result["runId"])
        status = get_status(job_id, run_id, req.jobs_dir, ssh_client, tracker)
        metric_value = float(status.get("metrics", {}).get("map5095", 0.0))
        trials.append(
            {
                "jobId": job_id,
                "runId": run_id,
                "params": {"learningRate": lr, "batch": batch, "imgSize": img_size},
                "metric": metric_value,
            }
        )
        progress_message = (
            f"[YOLO][TUNE] trial={trial_index}/{req.max_trials}, "
            f"job={job_id}, map50-95={metric_value:.4f}"
        )
        notifier.send_text(
            progress_message
        )

    if not trials:
        raise ValueError("No tuning trials executed")
    best_trial = max(trials, key=lambda item: item["metric"])
    return {
        "envId": req.env_id,
        "baseJobId": req.base_job_id,
        "bestJobId": best_trial["jobId"],
        "bestMetrics": {"map5095": best_trial["metric"]},
        "bestParams": best_trial["params"],
        "trials": trials,
    }

