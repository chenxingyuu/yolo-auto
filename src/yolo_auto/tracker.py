from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run


@dataclass(frozen=True)
class TrackerConfig:
    tracking_uri: str
    experiment_name: str
    external_url: str | None = None


class MLflowTracker:
    """Read-only MLflow helper for leaderboard and registry views."""

    def __init__(self, config: TrackerConfig) -> None:
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        self._experiment_name = config.experiment_name
        self._external_url = (config.external_url or "").strip() or None
        self._experiment_id: str | None = None
        self._client = MlflowClient()

    def _get_experiment_id(self) -> str | None:
        if self._experiment_id is not None:
            return self._experiment_id
        try:
            exp = mlflow.get_experiment_by_name(self._experiment_name)
        except Exception:
            exp = None
        self._experiment_id = exp.experiment_id if exp else None
        return self._experiment_id

    def get_experiment_url(self) -> str | None:
        if not self._external_url:
            return None
        exp_id = self._get_experiment_id()
        if not exp_id:
            return None
        base = self._external_url.rstrip("/")
        return f"{base}/#/experiments/{exp_id}"

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

    def list_experiments(self, *, max_results: int = 200) -> list[dict[str, Any]]:
        experiments = self._client.search_experiments(max_results=max_results)
        out: list[dict[str, Any]] = []
        for exp in experiments:
            out.append(
                {
                    "experimentId": str(exp.experiment_id),
                    "name": exp.name,
                    "lifecycleStage": exp.lifecycle_stage,
                    "artifactLocation": exp.artifact_location,
                }
            )
        return out

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

    def get_registered_model_ui_url(self, model_name: str) -> str | None:
        if not self._external_url:
            return None
        base = self._external_url.rstrip("/")
        enc = quote(model_name, safe="")
        return f"{base}/#/models/{enc}"

    def list_registered_models(self, *, max_results: int = 50) -> list[dict[str, Any]]:
        models = self._client.search_registered_models(max_results=max_results)
        out: list[dict[str, Any]] = []
        for m in models:
            latest = sorted(m.latest_versions or [], key=lambda x: int(x.version), reverse=True)
            latest_v = latest[0] if latest else None
            out.append(
                {
                    "name": m.name,
                    "aliases": dict(m.aliases or {}),
                    "description": m.description or "",
                    "latestVersion": str(latest_v.version) if latest_v else None,
                    "latestRunId": latest_v.run_id if latest_v else None,
                    "url": self.get_registered_model_ui_url(m.name),
                }
            )
        return out

    def list_model_versions(
        self,
        *,
        model_name: str,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        escaped_name = model_name.replace("'", "''")
        filter_expr = f"name = '{escaped_name}'"
        versions = self._client.search_model_versions(
            filter_string=filter_expr,
            max_results=max_results,
        )
        ordered_versions = sorted(versions, key=lambda x: int(x.version), reverse=True)
        out: list[dict[str, Any]] = []
        for v in ordered_versions:
            metric_val: float | None = None
            try:
                run = self._client.get_run(v.run_id)
                metrics = run.data.metrics or {}
                metric_val = float(metrics.get("map5095")) if "map5095" in metrics else None
            except Exception:
                metric_val = None
            out.append(
                {
                    "name": v.name,
                    "version": str(v.version),
                    "runId": v.run_id,
                    "source": v.source,
                    "aliases": list(v.aliases or []),
                    "tags": dict(v.tags or {}),
                    "metricMap5095": metric_val,
                    "createdAt": int(v.creation_timestamp or 0),
                }
            )
        return out
