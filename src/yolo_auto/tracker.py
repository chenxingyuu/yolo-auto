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
        run = mlflow.start_run(run_name=job_id)
        mlflow.log_params(config)
        return run.info.run_id

    def log_epoch(self, run_id: str, metrics: dict[str, float], step: int) -> None:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics, step=step)

    def finish_run(self, run_id: str, best_model_path: str | None = None) -> None:
        with mlflow.start_run(run_id=run_id):
            if best_model_path:
                model_path = Path(best_model_path)
                if model_path.exists():
                    mlflow.log_artifact(best_model_path)
            mlflow.end_run(status="FINISHED")

    def kill_run(self, run_id: str) -> None:
        with mlflow.start_run(run_id=run_id):
            mlflow.end_run(status="KILLED")

    def compare_runs(self, metric_key: str, descending: bool = True) -> list[Run]:
        order = "DESC" if descending else "ASC"
        return mlflow.search_runs(
            experiment_names=[self._experiment_name],
            order_by=[f"metrics.{metric_key} {order}"],
            output_format="list",
        )

