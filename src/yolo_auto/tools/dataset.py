from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from typing import Any

from yolo_auto.cvat_client import CVATClient
from yolo_auto.errors import err, ok
from yolo_auto.ssh_client import SSHClient
from yolo_auto.tools.sync import sync_dataset


def list_cvat_projects(cvat_client: CVATClient) -> dict[str, Any]:
    projects = cvat_client.list_projects()
    return ok({"projects": projects, "count": len(projects)})


def list_cvat_tasks(
    cvat_client: CVATClient,
    *,
    project_id: int | None = None,
) -> dict[str, Any]:
    tasks = cvat_client.list_tasks(project_id=project_id)
    return ok({"tasks": tasks, "count": len(tasks)})


def list_cvat_formats(cvat_client: CVATClient) -> dict[str, Any]:
    formats = cvat_client.list_formats()
    exporters = formats.get("exporters", [])
    importers = formats.get("importers", [])
    return ok(
        {
            "exporters": exporters,
            "importers": importers,
            "exporterCount": len(exporters),
            "importerCount": len(importers),
        }
    )


def get_cvat_task_detail(cvat_client: CVATClient, task_id: int) -> dict[str, Any]:
    detail = cvat_client.get_task_details(task_id)
    return ok({"task": detail})


def analyze_cvat_task(cvat_client: CVATClient, task_id: int) -> dict[str, Any]:
    analysis = cvat_client.analyze_task(task_id)
    return ok(analysis)


def list_cvat_request_queue(
    cvat_client: CVATClient,
    *,
    project_id: int | None = None,
    task_id: int | None = None,
    status: str | None = None,
    page: int | None = None,
    page_size: int = 20,
) -> dict[str, Any]:
    """列出 CVAT /api/requests 队列（含导出任务进度）。"""
    return ok(
        cvat_client.list_requests(
            project_id=project_id,
            task_id=task_id,
            status=status,
            page=page,
            page_size=page_size,
        )
    )


def export_cvat_dataset(
    cvat_client: CVATClient,
    *,
    task_id: int,
    dataset_name: str,
    format_name: str = "Ultralytics YOLO Detection 1.0",
    include_images: bool = False,
    cloud_storage_id: int | None = None,
    cloud_filename: str | None = None,
    status_check_period: int | None = None,
) -> dict[str, Any]:
    del status_check_period  # 云导出已异步；轮询请用 cvat_get_request / cvat_list_requests

    safe_name = _sanitize_dataset_name(dataset_name)
    if not safe_name:
        return err(
            error_code="INVALID_DATASET_NAME",
            message="datasetName is empty after sanitization",
            retryable=False,
            hint="datasetName 只能包含字母、数字、短横线和下划线",
        )

    if cloud_storage_id is None:
        return err(
            error_code="CVAT_CLOUD_STORAGE_REQUIRED",
            message="cloudStorageId is required for CVAT cloud export",
            retryable=False,
            hint="请配置 CVAT_CLOUD_STORAGE_ID 或在入参中提供 cloudStorageId",
        )

    export_token = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    try:
        cloud_target = cloud_filename or f"exports/{safe_name}-{export_token}.zip"
        rq_id = cvat_client.export_task_dataset_to_cloud(
            task_id,
            filename=cloud_target,
            cloud_storage_id=cloud_storage_id,
            format_name=format_name,
            include_images=include_images,
        )
    except Exception as exc:
        return err(
            error_code="CVAT_CLOUD_EXPORT_FAILED",
            message=str(exc),
            retryable=True,
            hint="检查 cloudStorageId、导出格式和 CVAT Cloud Storage 配置",
            payload={"taskId": task_id, "cloudStorageId": cloud_storage_id},
        )

    return ok(
        {
            "taskId": task_id,
            "datasetName": safe_name,
            "format": format_name,
            "rqId": rq_id,
            "cloudExport": {
                "enabled": True,
                "cloudStorageId": cloud_storage_id,
                "filename": cloud_target,
            },
            "pollHint": (
                "导出已进入后台队列。请用 cvat_get_request(rqId) 或 cvat_list_requests 查看进度；"
                "status 为 finished 后再执行 yolo_sync_dataset（filename=cloudExport.filename）。"
            ),
        }
    )


def export_and_sync_cvat_dataset(
    cvat_client: CVATClient,
    ssh_client: SSHClient,
    *,
    task_id: int,
    dataset_name: str,
    format_name: str = "Ultralytics YOLO Detection 1.0",
    include_images: bool = False,
    cloud_storage_id: int | None = None,
    cloud_filename: str | None = None,
    status_check_period: int | None = None,
    poll_seconds: float | None = None,
    max_wait_seconds: float | None = None,
    minio_alias: str,
    minio_bucket: str,
    minio_prefix: str,
    datasets_dir: str,
) -> dict[str, Any]:
    export_result = export_cvat_dataset(
        cvat_client,
        task_id=task_id,
        dataset_name=dataset_name,
        format_name=format_name,
        include_images=include_images,
        cloud_storage_id=cloud_storage_id,
        cloud_filename=cloud_filename,
        status_check_period=status_check_period,
    )
    if not export_result.get("ok", False):
        return export_result

    rq_id = str(export_result.get("rqId", "")).strip()
    if not rq_id:
        return err(
            error_code="MISSING_RQ_ID",
            message="export response missing rqId",
            retryable=False,
            hint="请检查 CVAT 版本与云导出接口是否返回 rq_id",
            payload=export_result,
        )

    eff_poll = float(
        poll_seconds
        if poll_seconds is not None
        else (status_check_period if status_check_period is not None else 5.0)
    )
    eff_poll = max(eff_poll, 0.1)
    eff_max = float(max_wait_seconds if max_wait_seconds is not None else 7200.0)

    wait_result = wait_for_cvat_export_request(
        cvat_client,
        rq_id,
        poll_seconds=eff_poll,
        max_wait_seconds=eff_max,
    )
    if not wait_result.get("ok", False):
        return err(
            error_code=str(wait_result.get("errorCode", "CVAT_EXPORT_WAIT_FAILED")),
            message=str(wait_result.get("error", "export wait failed")),
            retryable=bool(wait_result.get("retryable", True)),
            hint=str(wait_result.get("hint", "")),
            payload={"export": export_result, "wait": wait_result},
        )

    cloud_export = export_result.get("cloudExport", {})
    cloud_file = str(cloud_export.get("filename", "")).strip()
    if not cloud_file:
        return err(
            error_code="MISSING_CLOUD_FILENAME",
            message="cloud export filename missing",
            retryable=False,
            hint="请确认 cvat_export_dataset 返回 cloudExport.filename",
            payload=export_result,
        )

    sync_result = sync_dataset(
        ssh_client,
        minio_alias=minio_alias,
        minio_bucket=minio_bucket,
        minio_prefix=minio_prefix,
        filename=cloud_file,
        dataset_name=dataset_name,
        datasets_dir=datasets_dir,
    )
    if not sync_result.get("ok", False):
        return err(
            error_code="CVAT_EXPORT_SYNC_FAILED",
            message="dataset exported to cloud but sync to training container failed",
            retryable=True,
            hint="检查 MinIO 配置、训练容器 mc/unzip 和远程路径权限",
            payload={
                "export": export_result,
                "sync": sync_result,
            },
        )

    return ok(
        {
            "taskId": task_id,
            "datasetName": export_result.get("datasetName"),
            "format": export_result.get("format"),
            "rqId": rq_id,
            "request": wait_result.get("request"),
            "cloudExport": cloud_export,
            "sync": {
                "source": sync_result.get("source"),
                "extractedDir": sync_result.get("extractedDir"),
                "dataConfigPath": sync_result.get("dataConfigPath"),
                "files": sync_result.get("files", []),
            },
        }
    )


def export_cvat_project_dataset(
    cvat_client: CVATClient,
    *,
    project_id: int,
    dataset_name: str,
    format_name: str = "Ultralytics YOLO Detection 1.0",
    include_images: bool = False,
    cloud_storage_id: int | None = None,
    cloud_filename: str | None = None,
    status_check_period: int | None = None,
) -> dict[str, Any]:
    safe_name = _sanitize_dataset_name(dataset_name)
    if not safe_name:
        return err(
            error_code="INVALID_DATASET_NAME",
            message="datasetName is empty after sanitization",
            retryable=False,
            hint="datasetName 只能包含字母、数字、短横线和下划线",
        )

    if cloud_storage_id is None:
        return err(
            error_code="CVAT_CLOUD_STORAGE_REQUIRED",
            message="cloudStorageId is required for CVAT cloud export",
            retryable=False,
            hint="请配置 CVAT_CLOUD_STORAGE_ID 或在入参中提供 cloudStorageId",
        )

    del status_check_period
    export_token = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    try:
        cloud_target = cloud_filename or f"exports/{safe_name}-proj{project_id}-{export_token}.zip"
        rq_id = cvat_client.export_project_dataset_to_cloud(
            project_id,
            filename=cloud_target,
            cloud_storage_id=cloud_storage_id,
            format_name=format_name,
            include_images=include_images,
        )
    except Exception as exc:
        return err(
            error_code="CVAT_CLOUD_EXPORT_FAILED",
            message=str(exc),
            retryable=True,
            hint="检查 cloudStorageId、导出格式和 CVAT Cloud Storage 配置；项目导出可能较慢",
            payload={"projectId": project_id, "cloudStorageId": cloud_storage_id},
        )

    return ok(
        {
            "projectId": project_id,
            "datasetName": safe_name,
            "format": format_name,
            "rqId": rq_id,
            "cloudExport": {
                "enabled": True,
                "cloudStorageId": cloud_storage_id,
                "filename": cloud_target,
            },
            "pollHint": (
                "导出已进入后台队列。请用 cvat_get_request(rqId) 或 cvat_list_requests 查看进度；"
                "status 为 finished 后再执行 yolo_sync_dataset。"
            ),
        }
    )


def export_and_sync_cvat_project_dataset(
    cvat_client: CVATClient,
    ssh_client: SSHClient,
    *,
    project_id: int,
    dataset_name: str,
    format_name: str = "Ultralytics YOLO Detection 1.0",
    include_images: bool = False,
    cloud_storage_id: int | None = None,
    cloud_filename: str | None = None,
    status_check_period: int | None = None,
    poll_seconds: float | None = None,
    max_wait_seconds: float | None = None,
    minio_alias: str,
    minio_bucket: str,
    minio_prefix: str,
    datasets_dir: str,
) -> dict[str, Any]:
    export_result = export_cvat_project_dataset(
        cvat_client,
        project_id=project_id,
        dataset_name=dataset_name,
        format_name=format_name,
        include_images=include_images,
        cloud_storage_id=cloud_storage_id,
        cloud_filename=cloud_filename,
        status_check_period=status_check_period,
    )
    if not export_result.get("ok", False):
        return export_result

    rq_id = str(export_result.get("rqId", "")).strip()
    if not rq_id:
        return err(
            error_code="MISSING_RQ_ID",
            message="export response missing rqId",
            retryable=False,
            hint="请检查 CVAT 版本与云导出接口是否返回 rq_id",
            payload=export_result,
        )

    eff_poll = float(
        poll_seconds
        if poll_seconds is not None
        else (status_check_period if status_check_period is not None else 5.0)
    )
    eff_poll = max(eff_poll, 0.1)
    eff_max = float(max_wait_seconds if max_wait_seconds is not None else 7200.0)

    wait_result = wait_for_cvat_export_request(
        cvat_client,
        rq_id,
        poll_seconds=eff_poll,
        max_wait_seconds=eff_max,
    )
    if not wait_result.get("ok", False):
        return err(
            error_code=str(wait_result.get("errorCode", "CVAT_EXPORT_WAIT_FAILED")),
            message=str(wait_result.get("error", "export wait failed")),
            retryable=bool(wait_result.get("retryable", True)),
            hint=str(wait_result.get("hint", "")),
            payload={"export": export_result, "wait": wait_result},
        )

    cloud_export = export_result.get("cloudExport", {})
    cloud_file = str(cloud_export.get("filename", "")).strip()
    if not cloud_file:
        return err(
            error_code="MISSING_CLOUD_FILENAME",
            message="cloud export filename missing",
            retryable=False,
            hint="请确认 cvat_export_project_dataset 返回 cloudExport.filename",
            payload=export_result,
        )

    sync_result = sync_dataset(
        ssh_client,
        minio_alias=minio_alias,
        minio_bucket=minio_bucket,
        minio_prefix=minio_prefix,
        filename=cloud_file,
        dataset_name=dataset_name,
        datasets_dir=datasets_dir,
    )
    if not sync_result.get("ok", False):
        return err(
            error_code="CVAT_EXPORT_SYNC_FAILED",
            message="project exported to cloud but sync to training container failed",
            retryable=True,
            hint="检查 MinIO 配置、训练容器 mc/unzip 和远程路径权限",
            payload={
                "export": export_result,
                "sync": sync_result,
            },
        )

    return ok(
        {
            "projectId": project_id,
            "datasetName": export_result.get("datasetName"),
            "format": export_result.get("format"),
            "rqId": rq_id,
            "request": wait_result.get("request"),
            "cloudExport": cloud_export,
            "sync": {
                "source": sync_result.get("source"),
                "extractedDir": sync_result.get("extractedDir"),
                "dataConfigPath": sync_result.get("dataConfigPath"),
                "files": sync_result.get("files", []),
            },
        }
    )


def wait_for_cvat_export_request(
    cvat_client: CVATClient,
    rq_id: str,
    *,
    poll_seconds: float,
    max_wait_seconds: float,
) -> dict[str, Any]:
    """轮询 /api/requests/{id}，直到 finished / failed 或超时。"""
    deadline = time.monotonic() + max_wait_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            last = cvat_client.get_request(rq_id)
        except Exception as exc:
            return err(
                error_code="CVAT_REQUEST_POLL_FAILED",
                message=str(exc),
                retryable=True,
                hint="检查 rqId 是否与 CVAT 服务一致",
                payload={"rqId": rq_id},
            )

        status = _request_status_value(last)
        if status == "finished":
            return ok({"request": last, "rqId": rq_id})
        if status == "failed":
            return err(
                error_code="CVAT_EXPORT_FAILED",
                message=str(last.get("message", "export failed")),
                retryable=False,
                hint="查看 request.message；修复数据后重试导出",
                payload={"request": last, "rqId": rq_id},
            )

        time.sleep(poll_seconds)

    return err(
        error_code="CVAT_EXPORT_TIMEOUT",
        message=(
            "export not finished in time "
            f"(maxWaitSeconds={max_wait_seconds}, "
            f"lastStatus={_request_status_value(last) or 'unknown'})"
        ),
        retryable=True,
        hint=(
            "项目/大数据集导出较慢：增大 maxWaitSeconds，或保留 rqId 稍后 cvat_get_request，"
            "finished 后再 yolo_sync_dataset"
        ),
        payload={"request": last, "rqId": rq_id},
    )


def _request_status_value(request_payload: dict[str, Any]) -> str:
    raw = request_payload.get("status")
    if isinstance(raw, str):
        return raw.strip().lower()
    if isinstance(raw, dict):
        val = raw.get("value")
        if isinstance(val, str):
            return val.strip().lower()
    return ""


def _sanitize_dataset_name(raw_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw_name.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-_")
    return normalized
