from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
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
        self._last_logged_step_by_run: dict[str, int] = {}

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
        self._try_log_dataset_input(config)
        for tag_key, tag_val in (tags or {}).items():
            val = str(tag_val).strip()
            if val:
                mlflow.set_tag(tag_key, val)
        return run.info.run_id

    def _try_log_dataset_input(self, config: dict[str, Any]) -> None:
        raw_path = config.get("data_config_path")
        if raw_path is None:
            return
        dataset_path = str(raw_path).strip()
        if not dataset_path:
            return

        try:
            dataset = mlflow.data.from_pandas(
                pd.DataFrame({"data_config_path": [dataset_path]}),
                source=dataset_path,
                name="yolo_dataset_config",
            )
            mlflow.log_input(dataset, context="training")
        except Exception:
            # Dataset input 记录失败时降级，不影响训练主流程。
            return

    def log_epoch(self, run_id: str, metrics: dict[str, float], step: int) -> None:
        # 高频轮询 get_status 时，同一 epoch 可能被重复上报。
        # 这里做单进程去重，减少 MLflow backend（尤其 sqlite）写放大。
        last_step = self._last_logged_step_by_run.get(run_id)
        if last_step is not None and step <= last_step:
            return

        # 训练启动后 `start_run()` 不会自动 end_run，因此此时
        # `mlflow.active_run()` 可能已经等于当前 run_id。
        # 此时再次以 nested=False 开始同一个 run 会抛异常：
        # "Run with UUID is already active".
        active = mlflow.active_run()
        if active is not None and active.info.run_id == run_id:
            mlflow.log_metrics(metrics, step=step)
            self._last_logged_step_by_run[run_id] = step
            return

        # 若当前没有对应 run 处于 active，显式打开并记录指标。
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_metrics(metrics, step=step)
        self._last_logged_step_by_run[run_id] = step

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

