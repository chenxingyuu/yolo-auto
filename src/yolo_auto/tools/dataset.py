from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from shlex import quote
from typing import Any

from yolo_auto.cvat_client import CVATClient
from yolo_auto.errors import err, ok
from yolo_auto.ssh_client import SSHClient


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


def get_cvat_task_detail(cvat_client: CVATClient, task_id: int) -> dict[str, Any]:
    detail = cvat_client.get_task_details(task_id)
    return ok({"task": detail})


def analyze_cvat_task(cvat_client: CVATClient, task_id: int) -> dict[str, Any]:
    analysis = cvat_client.analyze_task(task_id)
    return ok(analysis)


def export_cvat_dataset(
    cvat_client: CVATClient,
    ssh_client: SSHClient,
    *,
    task_id: int,
    dataset_name: str,
    datasets_dir: str,
    format_name: str = "YOLO 1.1",
    include_images: bool = True,
) -> dict[str, Any]:
    safe_name = _sanitize_dataset_name(dataset_name)
    if not safe_name:
        return err(
            error_code="INVALID_DATASET_NAME",
            message="datasetName is empty after sanitization",
            retryable=False,
            hint="datasetName 只能包含字母、数字、短横线和下划线",
        )

    try:
        dataset_bytes = cvat_client.export_task_dataset(
            task_id,
            format_name=format_name,
            include_images=include_images,
        )
    except Exception as exc:
        return err(
            error_code="CVAT_EXPORT_FAILED",
            message=str(exc),
            retryable=True,
            hint="检查 taskId、导出格式和 CVAT 服务连通性",
            payload={"taskId": task_id, "format": format_name},
        )

    export_token = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    remote_zip_path = f"/tmp/cvat-task-{task_id}-{export_token}.zip"
    target_dir = f"{datasets_dir.rstrip('/')}/{safe_name}"
    data_yaml_path = f"{target_dir}/data.yaml"
    train_images = f"{target_dir}/images/train"
    val_images = f"{target_dir}/images/val"

    try:
        ssh_client.upload_bytes(dataset_bytes, remote_zip_path)
    except Exception as exc:
        return err(
            error_code="UPLOAD_FAILED",
            message=str(exc),
            retryable=True,
            hint="检查 SSH 连接和远程 /tmp 可写权限",
            payload={"remoteZipPath": remote_zip_path},
        )

    yaml_content = (
        f"path: {target_dir}\n"
        f"train: {train_images}\n"
        f"val: {val_images}\n"
        "test: \n\n"
        "names: []\n"
    )
    setup_cmd = (
        f"set -e; "
        f"rm -rf {quote(target_dir)}; "
        f"mkdir -p {quote(target_dir)}; "
        f"unzip -oq {quote(remote_zip_path)} -d {quote(target_dir)}; "
        f"rm -f {quote(remote_zip_path)}; "
        f"cat > {quote(data_yaml_path)} <<'EOF'\n{yaml_content}EOF\n"
    )
    _, stderr_text, code = ssh_client.execute(setup_cmd, timeout=180)
    if code != 0:
        return err(
            error_code="REMOTE_PREPARE_FAILED",
            message=stderr_text.strip() or "failed to unzip dataset on remote host",
            retryable=True,
            hint="确认远程容器已安装 unzip，并检查 datasets 目录权限",
            payload={"targetDir": target_dir, "remoteZipPath": remote_zip_path},
        )

    labels = _collect_labels(cvat_client, task_id)
    if labels:
        final_yaml = _build_data_yaml(target_dir, labels)
        _, write_err, write_code = ssh_client.execute(
            f"cat > {quote(data_yaml_path)} <<'EOF'\n{final_yaml}EOF\n",
            timeout=30,
        )
        if write_code != 0:
            return err(
                error_code="WRITE_DATA_YAML_FAILED",
                message=write_err.strip() or "failed to write data.yaml",
                retryable=True,
                hint="检查远程目录权限",
                payload={"dataConfigPath": data_yaml_path},
            )

    return ok(
        {
            "taskId": task_id,
            "datasetName": safe_name,
            "format": format_name,
            "targetDir": target_dir,
            "dataConfigPath": data_yaml_path,
            "labels": labels,
        }
    )


def _sanitize_dataset_name(raw_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw_name.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-_")
    return normalized


def _collect_labels(cvat_client: CVATClient, task_id: int) -> list[str]:
    try:
        detail = cvat_client.get_task_details(task_id)
    except Exception:
        return []
    labels = detail.get("labels", [])
    if not isinstance(labels, list):
        return []
    names: list[str] = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _build_data_yaml(target_dir: str, labels: list[str]) -> str:
    names_json = json.dumps(labels, ensure_ascii=False)
    return (
        f"path: {target_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: \n\n"
        f"names: {names_json}\n"
    )
