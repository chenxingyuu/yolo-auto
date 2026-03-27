from __future__ import annotations

import re
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
        cvat_client.export_task_dataset_to_cloud(
            task_id,
            filename=cloud_target,
            cloud_storage_id=cloud_storage_id,
            format_name=format_name,
            include_images=include_images,
            status_check_period=status_check_period,
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
            "cloudExport": {
                "enabled": True,
                "cloudStorageId": cloud_storage_id,
                "filename": cloud_target,
            },
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
            "cloudExport": cloud_export,
            "sync": {
                "source": sync_result.get("source"),
                "extractedDir": sync_result.get("extractedDir"),
                "dataConfigPath": sync_result.get("dataConfigPath"),
                "files": sync_result.get("files", []),
            },
        }
    )


def _sanitize_dataset_name(raw_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw_name.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-_")
    return normalized
