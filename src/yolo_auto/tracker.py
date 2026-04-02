from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Run


def _mlflow_param_strings(config: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, raw in config.items():
        if raw is None:
            continue
        out[key] = str(raw)
    return out


@dataclass(frozen=True)
class TrackerConfig:
    tracking_uri: str
    experiment_name: str
    external_url: str | None = None


class MLflowTracker:
    def __init__(self, config: TrackerConfig) -> None:
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        self._experiment_name = config.experiment_name
        self._external_url = (config.external_url or "").strip() or None
        self._experiment_id: str | None = None

    def _get_experiment_id(self) -> str | None:
        if self._experiment_id is not None:
            return self._experiment_id
        try:
            exp = mlflow.get_experiment_by_name(self._experiment_name)
        except Exception:
            exp = None
        self._experiment_id = exp.experiment_id if exp else None
        return self._experiment_id

    def get_run_url(self, run_id: str) -> str | None:
        if not self._external_url:
            return None
        exp_id = self._get_experiment_id()
        if not exp_id:
            return None
        base = self._external_url.rstrip("/")
        return f"{base}/#/experiments/{exp_id}/runs/{run_id}"

    def get_experiment_url(self) -> str | None:
        if not self._external_url:
            return None
        exp_id = self._get_experiment_id()
        if not exp_id:
            return None
        base = self._external_url.rstrip("/")
        return f"{base}/#/experiments/{exp_id}"

    def start_run(
        self,
        job_id: str,
        config: dict[str, Any],
        *,
        tags: dict[str, str] | None = None,
    ) -> str:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="KILLED")
        run = mlflow.start_run(run_name=job_id)
        mlflow.log_params(_mlflow_param_strings(config))
        for tag_key, tag_val in (tags or {}).items():
            val = str(tag_val).strip()
            if val:
                mlflow.set_tag(tag_key, val)
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

    def compare_runs(
        self,
        metric_key: str,
        descending: bool = True,
        *,
        filter_string: str | None = None,
    ) -> list[Run]:
        order = "DESC" if descending else "ASC"
        kwargs: dict[str, Any] = {
            "experiment_names": [self._experiment_name],
            "order_by": [f"metrics.{metric_key} {order}"],
            "output_format": "list",
        }
        if filter_string:
            kwargs["filter_string"] = filter_string
        return mlflow.search_runs(**kwargs)

    def summarize_top_runs(
        self,
        metric_key: str,
        limit: int = 5,
        *,
        filter_string: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            runs = self.compare_runs(
                metric_key,
                descending=True,
                filter_string=filter_string,
            )
        except Exception:
            return []
        result: list[dict[str, Any]] = []
        for run in runs[:limit]:
            metrics = run.data.metrics or {}
            metric_val = float(metrics.get(metric_key, 0.0))
            tags = run.data.tags or {}
            group_tags = {k: tags[k] for k in sorted(tags) if k.startswith("yolo_")}
            result.append(
                {
                    "runId": run.info.run_id,
                    "runName": run.info.run_name,
                    "metricKey": metric_key,
                    "metric": metric_val,
                    "groupTags": group_tags,
                }
            )
        return result

