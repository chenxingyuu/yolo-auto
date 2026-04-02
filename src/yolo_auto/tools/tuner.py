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
from yolo_auto.tools.mlflow_grouping import mlflow_filter_same_training_scope
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
    primary_metric_key: str = "map5095"
    feishu_report_enable: bool = True
    feishu_report_every_n_epochs: int = 5


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
            env_id=req.env_id,
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
            latest_status = get_status(
                job_id,
                run_id,
                state_store,
                ssh_client,
                tracker,
                notifier,
                feishu_report_enable=req.feishu_report_enable,
                feishu_report_every_n_epochs=req.feishu_report_every_n_epochs,
                primary_metric_key=req.primary_metric_key,
            )
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

        metrics_payload = latest_status.get("metrics", {})
        if isinstance(metrics_payload, dict):
            raw_metric = metrics_payload.get(
                req.primary_metric_key,
                metrics_payload.get("primaryMetric", 0.0),
            )
            metric_value = float(raw_metric)
        else:
            metric_value = 0.0
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
        notifier.send_training_update(
            "[YOLO] 调参 trial 完成",
            f"trial={trial_index}/{req.max_trials}\njob={job_id}\n"
            f"{req.primary_metric_key}={metric_value:.4f}",
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
    scope_filter = mlflow_filter_same_training_scope(
        env_id=req.env_id,
        model_path=req.model,
        data_config_path=req.data_config_path,
    )
    mlflow_top = tracker.summarize_top_runs(
        req.primary_metric_key,
        limit=5,
        filter_string=scope_filter,
    )
    best_mlflow = mlflow_top[0] if mlflow_top else None
    disagreement = False
    if best_mlflow is not None:
        disagreement = abs(float(best_mlflow["metric"]) - float(best_trial["metric"])) > 1e-4

    try:
        exp_url = tracker.get_experiment_url()
        notifier.send_rich_card(
            title="[YOLO] 调参完成",
            md_text=(
                f"baseJobId={req.base_job_id}\n"
                f"bestJobId={best_trial['jobId']}\n"
                f"{req.primary_metric_key}={float(best_trial['metric']):.4f}\n"
                f"trials={len(trials)}"
            ),
            header_color="green",
            actions=(
                [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "查看 MLflow 实验"},
                        "type": "url",
                        "url": exp_url,
                    }
                ]
                if exp_url
                else None
            ),
        )
    except Exception:
        pass
    return ok(
        {
            "envId": req.env_id,
            "baseJobId": req.base_job_id,
            "bestJobId": best_trial["jobId"],
            "bestMetrics": {req.primary_metric_key: best_trial["metric"]},
            "bestParams": best_trial["params"],
            "bestFromTrials": {
                "jobId": best_trial["jobId"],
                "metricKey": req.primary_metric_key,
                "metric": best_trial["metric"],
            },
            "bestFromMlflow": best_mlflow,
            "mlflowTopRuns": mlflow_top,
            "disagreement": disagreement,
            "trials": trials,
        }
    )

