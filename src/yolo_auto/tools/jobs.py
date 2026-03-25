from __future__ import annotations

import csv
from io import StringIO
from typing import Any

from yolo_auto.errors import err, ok
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobRecord
from yolo_auto.ssh_client import SSHClient
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.metrics_csv import parse_training_row
from yolo_auto.tools.status import get_status
from yolo_auto.tracker import MLflowTracker


def _epoch_hint(record: JobRecord, ssh_client: SSHClient) -> dict[str, Any]:
    path = record.paths.get("metricsPath", "")
    content, _, code = ssh_client.execute(f"cat {path}")
    if code != 0 or not content.strip():
        return {}
    rows = list(csv.DictReader(StringIO(content)))
    if not rows:
        return {}
    parsed = parse_training_row(rows[-1])
    return {"epoch": parsed["epoch"]}


def list_jobs(
    state_store: JobStateStore,
    ssh_client: SSHClient,
    limit: int = 20,
) -> dict[str, Any]:
    capped = max(1, min(limit, 100))
    records = state_store.list_all()[:capped]
    items: list[dict[str, Any]] = []
    for record in records:
        hint = _epoch_hint(record, ssh_client)
        merged = record.to_dict()
        merged["epochHint"] = hint.get("epoch")
        items.append(merged)
    return ok({"jobs": items, "count": len(items)})


def get_job(
    job_id: str,
    state_store: JobStateStore,
    ssh_client: SSHClient,
    tracker: MLflowTracker,
    notifier: FeishuNotifier,
    *,
    refresh: bool = False,
    feishu_report_enable: bool = True,
    feishu_report_every_n_epochs: int = 5,
    primary_metric_key: str = "map5095",
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
        )
        payload["liveStatus"] = status_payload
    return ok(payload)
