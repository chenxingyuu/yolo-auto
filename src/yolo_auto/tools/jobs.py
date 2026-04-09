from __future__ import annotations

import csv
from io import StringIO
from typing import Any

from yolo_auto.errors import err, ok
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobStatus
from yolo_auto.notifier_state_store import NotifierStateStore
from yolo_auto.remote_control import HttpControlClient
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.metrics_csv import parse_training_row
from yolo_auto.tools.status import get_status
from yolo_auto.tracker import MLflowTracker


def _job_status_from_mlflow(run_status: str) -> JobStatus:
    s = (run_status or "").strip().upper()
    if s == "RUNNING":
        return JobStatus.RUNNING
    if s == "FAILED":
        return JobStatus.FAILED
    if s == "KILLED":
        return JobStatus.STOPPED
    # FINISHED or unknown -> completed (best-effort)
    return JobStatus.COMPLETED


def _job_id_from_mlflow_run(run_name: str | None, tags: dict[str, Any] | None) -> str:
    t = tags or {}
    jid = str(t.get("yolo_job_id", "")).strip()
    if jid:
        return jid
    rn = (run_name or "").strip()
    return rn


def _epoch_hint(record) -> dict[str, Any]:
    path = record.paths.get("metricsPath", "")
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fp:
            content = fp.read()
    except Exception:
        return {}
    if not content.strip():
        return {}
    rows = list(csv.DictReader(StringIO(content)))
    if not rows:
        return {}
    parsed = parse_training_row(rows[-1])
    return {"epoch": parsed["epoch"]}


def list_jobs(
    state_store: JobStateStore,
    legacy_clients_by_env: dict[str, Any] | None,
    tracker: MLflowTracker | None = None,
    limit: int = 20,
    control_client: HttpControlClient | None = None,
) -> dict[str, Any]:
    capped = max(1, min(limit, 100))
    if tracker is not None:
        runs = tracker.list_recent_runs(limit=capped)
        items: list[dict[str, Any]] = []
        for run in runs:
            job_id = _job_id_from_mlflow_run(run.info.run_name, run.data.tags or {})
            status = _job_status_from_mlflow(run.info.status)
            metrics = run.data.metrics or {}
            epoch_hint = metrics.get("epoch") if isinstance(metrics, dict) else None
            items.append(
                {
                    "jobId": job_id,
                    "runId": run.info.run_id,
                    "status": status.value,
                    "createdAt": int((run.info.start_time or 0) // 1000),
                    "updatedAt": int((run.info.end_time or run.info.start_time or 0) // 1000),
                    "epochHint": epoch_hint,
                    "mlflow": {
                        "runName": run.info.run_name,
                        "metrics": dict(metrics),
                        "tags": dict(run.data.tags or {}),
                    },
                }
            )
        return ok({"jobs": items, "count": len(items)})

    if control_client is not None:
        try:
            remote = control_client.list_jobs(limit=capped)
            if isinstance(remote, dict) and isinstance(remote.get("jobs"), list):
                return ok({"jobs": remote["jobs"], "count": int(remote.get("count", 0))})
        except Exception:
            pass
    records = state_store.list_all()[:capped]
    items: list[dict[str, Any]] = []
    for record in records:
        _ = legacy_clients_by_env
        hint = _epoch_hint(record)
        merged = record.to_dict()
        merged["epochHint"] = hint.get("epoch")
        items.append(merged)
    return ok({"jobs": items, "count": len(items)})


def get_job(
    job_id: str,
    state_store: JobStateStore,
    legacy_clients_by_env: dict[str, Any] | None,
    notifier: FeishuNotifier,
    tracker: MLflowTracker | None = None,
    notifier_store: NotifierStateStore | None = None,
    *,
    refresh: bool = False,
    feishu_report_enable: bool = True,
    feishu_report_every_n_epochs: int = 5,
    primary_metric_key: str = "map5095",
    feishu_card_img_key: str | None = None,
    feishu_card_fallback_img_key: str | None = None,
    control_client: HttpControlClient | None = None,
) -> dict[str, Any]:
    record = state_store.get(job_id)
    payload: dict[str, Any] = {}
    if record:
        payload["record"] = record.to_dict()

    if tracker is not None:
        run = tracker.find_latest_run_for_job_id(job_id)
        if run is not None:
            status = _job_status_from_mlflow(run.info.status)
            payload["mlflow"] = {
                "jobId": _job_id_from_mlflow_run(run.info.run_name, run.data.tags or {}),
                "runId": run.info.run_id,
                "status": status.value,
                "runName": run.info.run_name,
                "createdAt": int((run.info.start_time or 0) // 1000),
                "updatedAt": int((run.info.end_time or run.info.start_time or 0) // 1000),
                "metrics": dict(run.data.metrics or {}),
                "tags": dict(run.data.tags or {}),
            }
            payload["mlflowUrl"] = tracker.get_run_url(run.info.run_id)

    if not record and "mlflow" not in payload:
        return err(
            error_code="JOB_NOT_FOUND",
            message=f"job not found: {job_id}",
            retryable=False,
            hint="请先启动训练或检查 jobId",
            payload={"jobId": job_id},
        )

    if refresh:
        if not record:
            return err(
                error_code="JOB_NOT_FOUND",
                message=f"job not found in local state: {job_id}",
                retryable=False,
                hint=(
                    "refresh 需要本地状态（用于 pid/paths）；"
                    "可先调用 yolo_start_training 创建任务"
                ),
                payload={"jobId": job_id},
            )
        _ = legacy_clients_by_env
        status_payload = get_status(
            job_id,
            record.run_id,
            state_store,
            None,
            notifier,
            tracker=tracker,
            notifier_store=notifier_store,
            feishu_report_enable=feishu_report_enable,
            feishu_report_every_n_epochs=feishu_report_every_n_epochs,
            primary_metric_key=primary_metric_key,
            feishu_card_img_key=feishu_card_img_key,
            feishu_card_fallback_img_key=feishu_card_fallback_img_key,
            control_client=control_client,
        )
        payload["liveStatus"] = status_payload
        refreshed = state_store.get(job_id) or record
        payload["record"] = refreshed.to_dict()
    return ok(payload)


def delete_job(job_id: str, state_store: JobStateStore) -> dict[str, Any]:
    record = state_store.get(job_id)
    if not record:
        return err(
            error_code="JOB_NOT_FOUND",
            message=f"job not found: {job_id}",
            retryable=False,
            hint="请先检查 jobId，或先调用 yolo_list_jobs 查看任务列表",
            payload={"jobId": job_id},
        )
    if record.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
        return err(
            error_code="JOB_DELETE_FORBIDDEN",
            message=f"job is {record.status.value}, stop it before deletion",
            retryable=False,
            hint="请先调用 yolo_stop_training 停止任务，再删除状态记录",
            payload={"jobId": job_id, "status": record.status.value},
        )
    deleted = state_store.delete(job_id)
    return ok({"jobId": job_id, "deleted": deleted})
