from __future__ import annotations

import csv
from io import StringIO
from typing import Any

from yolo_auto.errors import err, ok
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.ssh_client import SSHClient
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.metrics_csv import parse_training_row
from yolo_auto.tools.status import get_status
from yolo_auto.tracker import MLflowTracker


def _ssh_for_record(
    record: JobRecord,
    ssh_clients_by_env: dict[str, SSHClient],
    *,
    default_env_id: str = "default",
) -> SSHClient:
    ssh = ssh_clients_by_env.get(record.env_id)
    if ssh:
        return ssh
    # 回退到 default；保证旧数据或未知 envId 不会导致整个请求失败。
    return ssh_clients_by_env[default_env_id]


def _epoch_hint(record: JobRecord, ssh_clients_by_env: dict[str, SSHClient]) -> dict[str, Any]:
    path = record.paths.get("metricsPath", "")
    if not path:
        return {}
    ssh_client = _ssh_for_record(record, ssh_clients_by_env)
    try:
        content, _, code = ssh_client.execute(f"cat {path}")
    except Exception:
        # list_jobs 是轻量查看接口：SSH 不可用时也应尽量返回本地状态。
        return {}
    if code != 0 or not content.strip():
        return {}
    rows = list(csv.DictReader(StringIO(content)))
    if not rows:
        return {}
    parsed = parse_training_row(rows[-1])
    return {"epoch": parsed["epoch"]}


def list_jobs(
    state_store: JobStateStore,
    ssh_clients_by_env: dict[str, SSHClient],
    limit: int = 20,
) -> dict[str, Any]:
    capped = max(1, min(limit, 100))
    records = state_store.list_all()[:capped]
    items: list[dict[str, Any]] = []
    for record in records:
        hint = _epoch_hint(record, ssh_clients_by_env)
        merged = record.to_dict()
        merged["epochHint"] = hint.get("epoch")
        items.append(merged)
    return ok({"jobs": items, "count": len(items)})


def get_job(
    job_id: str,
    state_store: JobStateStore,
    ssh_clients_by_env: dict[str, SSHClient],
    tracker: MLflowTracker,
    notifier: FeishuNotifier,
    *,
    refresh: bool = False,
    feishu_report_enable: bool = True,
    feishu_report_every_n_epochs: int = 5,
    primary_metric_key: str = "map5095",
    feishu_card_img_key: str | None = None,
    feishu_card_fallback_img_key: str | None = None,
) -> dict[str, Any]:
    record = state_store.get(job_id)
    if not record:
        return err(
            error_code="JOB_NOT_FOUND",
            message=f"job not found: {job_id}",
            retryable=False,
            hint="请先启动训练或检查 jobId",
            payload={"jobId": job_id},
        )
    payload: dict[str, Any] = {"record": record.to_dict()}
    if refresh:
        ssh_client = _ssh_for_record(record, ssh_clients_by_env)
        status_payload = get_status(
            job_id,
            record.run_id,
            state_store,
            ssh_client,
            tracker,
            notifier,
            feishu_report_enable=feishu_report_enable,
            feishu_report_every_n_epochs=feishu_report_every_n_epochs,
            primary_metric_key=primary_metric_key,
            feishu_card_img_key=feishu_card_img_key,
            feishu_card_fallback_img_key=feishu_card_fallback_img_key,
        )
        payload["liveStatus"] = status_payload
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
