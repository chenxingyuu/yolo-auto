from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Run


@dataclass(frozen=True)
class TrackerConfig:
    tracking_uri: str
    experiment_name: str


class MLflowTracker:
    def __init__(self, config: TrackerConfig) -> None:
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        self._experiment_name = config.experiment_name

    def start_run(self, job_id: str, config: dict[str, Any]) -> str:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="KILLED")
        run = mlflow.start_run(run_name=job_id)
        mlflow.log_params(config)
        return run.info.run_id

    def log_epoch(self, run_id: str, metrics: dict[str, float], step: int) -> None:
        # 训练启动后 `start_run()` 不会自动 end_run，因此此时
        # `mlflow.active_run()` 可能已经等于当前 run_id。
        # 此时再次以 nested=False 开始同一个 run 会抛异常：
        # "Run with UUID is already active".
        active = mlflow.active_run()
        if active is not None and active.info.run_id == run_id:
            mlflow.log_metrics(metrics, step=step)
            return

        # 若当前没有对应 run 处于 active，显式打开并记录指标。
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_metrics(metrics, step=step)

    def finish_run(self, run_id: str, best_model_path: str | None = None) -> None:
        active = mlflow.active_run()
        if active is not None and active.info.run_id == run_id:
            if best_model_path:
                model_path = Path(best_model_path)
                if model_path.exists():
                    mlflow.log_artifact(best_model_path)
            mlflow.end_run(status="FINISHED")
            return

        with mlflow.start_run(run_id=run_id, nested=True):
            if best_model_path:
                model_path = Path(best_model_path)
                if model_path.exists():
                    mlflow.log_artifact(best_model_path)
            mlflow.end_run(status="FINISHED")

    def kill_run(self, run_id: str) -> None:
        active = mlflow.active_run()
        if active is not None and active.info.run_id == run_id:
            mlflow.end_run(status="KILLED")
            return

        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.end_run(status="KILLED")

    def compare_runs(self, metric_key: str, descending: bool = True) -> list[Run]:
        order = "DESC" if descending else "ASC"
        return mlflow.search_runs(
            experiment_names=[self._experiment_name],
            order_by=[f"metrics.{metric_key} {order}"],
            output_format="list",
        )

    def summarize_top_runs(self, metric_key: str, limit: int = 5) -> list[dict[str, Any]]:
        try:
            runs = self.compare_runs(metric_key, descending=True)
        except Exception:
            return []
        result: list[dict[str, Any]] = []
        for run in runs[:limit]:
            metrics = run.data.metrics or {}
            metric_val = float(metrics.get(metric_key, 0.0))
            result.append(
                {
                    "runId": run.info.run_id,
                    "runName": run.info.run_name,
                    "metricKey": metric_key,
                    "metric": metric_val,
                }
            )
        return result

