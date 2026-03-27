from __future__ import annotations

import os
import re
import shlex
import time
from typing import Any

from yolo_auto.errors import err, ok
from yolo_auto.ssh_client import SSHClient

_DATASET_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")


def sync_dataset(
    ssh_client: SSHClient,
    *,
    minio_alias: str,
    minio_bucket: str,
    minio_prefix: str,
    filename: str,
    dataset_name: str,
    datasets_dir: str,
) -> dict[str, Any]:
    safe_filename = _sanitize_filename(filename)
    if not safe_filename:
        return err(
            error_code="INVALID_MINIO_FILENAME",
            message="filename is empty or invalid",
            retryable=False,
            hint="filename 应为 MinIO 中已存在的 zip 文件名，如 task8-20260327075925.zip",
        )
    if not safe_filename.lower().endswith(".zip"):
        return err(
            error_code="INVALID_MINIO_FILENAME",
            message="filename must be a .zip file",
            retryable=False,
            hint="请传入 zip 文件名，例如 exports/task9_export_sync_20260327.zip",
        )

    safe_dataset_name = _sanitize_dataset_name(dataset_name)
    if not safe_dataset_name:
        return err(
            error_code="INVALID_DATASET_NAME",
            message="datasetName is empty after sanitization",
            retryable=False,
            hint="datasetName 只能包含字母、数字、点、短横线和下划线",
        )

    object_name = _normalize_object_name(
        safe_filename,
        bucket=minio_bucket,
        prefix=minio_prefix,
    )
    remote_source = _join_remote_path(minio_alias, minio_bucket, minio_prefix, object_name)
    extracted_dir = f"{datasets_dir.rstrip('/')}/{safe_dataset_name}"
    tmp_zip = f"/tmp/yolo-auto-sync-{int(time.time())}-{safe_filename}"

    check_mc_cmd = "command -v mc >/dev/null 2>&1"
    _, _, check_mc_code = ssh_client.execute(check_mc_cmd, timeout=10)
    if check_mc_code != 0:
        return err(
            error_code="MC_NOT_FOUND",
            message="mc command not found on remote host",
            retryable=False,
            hint="请在训练容器安装 MinIO Client (mc) 并确保在 PATH 中",
        )

    check_unzip_cmd = "command -v unzip >/dev/null 2>&1"
    _, _, check_unzip_code = ssh_client.execute(check_unzip_cmd, timeout=10)
    if check_unzip_code != 0:
        return err(
            error_code="UNZIP_NOT_FOUND",
            message="unzip command not found on remote host",
            retryable=False,
            hint="请在训练容器安装 unzip（apt-get install -y unzip）",
        )

    check_source_cmd = f"mc stat {shlex.quote(remote_source)} >/dev/null 2>&1"
    _, _, source_code = ssh_client.execute(check_source_cmd, timeout=30)
    if source_code != 0:
        return err(
            error_code="MINIO_OBJECT_NOT_FOUND",
            message=f"object not found: {remote_source}",
            retryable=False,
            hint="请确认文件名和 MinIO 路径是否正确",
            payload={"source": remote_source},
        )

    sync_cmd = (
        "set -euo pipefail && "
        f"rm -f {shlex.quote(tmp_zip)} && "
        f"mkdir -p {shlex.quote(extracted_dir)} && "
        f"mc cp {shlex.quote(remote_source)} {shlex.quote(tmp_zip)} && "
        f"unzip -o {shlex.quote(tmp_zip)} -d {shlex.quote(extracted_dir)} >/dev/null && "
        f"rm -f {shlex.quote(tmp_zip)}"
    )
    _, stderr_sync, sync_code = ssh_client.execute(sync_cmd, timeout=300)
    if sync_code != 0:
        return err(
            error_code="SYNC_DATASET_FAILED",
            message=stderr_sync.strip() or "failed to sync and extract dataset zip",
            retryable=True,
            hint="检查 MinIO 连通性、压缩包完整性和容器磁盘空间",
            payload={"source": remote_source, "targetDir": extracted_dir},
        )

    detect_data_yaml_cmd = (
        f"cd {shlex.quote(extracted_dir)} && "
        "(test -f data.yaml && echo data.yaml) || "
        "(test -f dataset.yaml && echo dataset.yaml) || "
        "(test -f data.yml && echo data.yml) || "
        "(test -f dataset.yml && echo dataset.yml) || "
        "find . -maxdepth 6 -type f "
        "\\( -name 'data.yaml' -o -name 'dataset.yaml' -o -name 'data.yml' "
        "-o -name 'dataset.yml' \\) "
        "-print | head -1"
    )
    data_yaml_rel, _, data_yaml_code = ssh_client.execute(detect_data_yaml_cmd, timeout=20)
    data_yaml_rel = data_yaml_rel.strip().splitlines()[0] if data_yaml_rel.strip() else ""
    if data_yaml_code != 0 or not data_yaml_rel:
        return err(
            error_code="DATA_YAML_NOT_FOUND",
            message="cannot find data.yaml after extraction",
            retryable=False,
            hint="请确认 CVAT 导出格式为 Ultralytics YOLO，且压缩包内包含 data.yaml",
            payload={"targetDir": extracted_dir},
        )

    data_config_path = _normalize_join(extracted_dir, data_yaml_rel)
    list_files_cmd = (
        f"cd {shlex.quote(extracted_dir)} && "
        "find . -maxdepth 3 -type f | head -50"
    )
    files_out, _, _ = ssh_client.execute(list_files_cmd, timeout=20)
    files = [
        f"{extracted_dir}/{line.strip().lstrip('./')}"
        for line in files_out.splitlines()
        if line.strip()
    ]

    return ok(
        {
            "source": remote_source,
            "filename": safe_filename,
            "datasetName": safe_dataset_name,
            "extractedDir": extracted_dir,
            "dataConfigPath": data_config_path,
            "files": files,
        }
    )


def _sanitize_filename(raw: str) -> str:
    name = raw.strip().replace("\\", "/")
    return name.lstrip("/").replace("..", "").strip()


def _sanitize_dataset_name(raw: str) -> str:
    normalized = _DATASET_NAME_PATTERN.sub("-", raw.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-._")
    return normalized


def _join_remote_path(alias: str, bucket: str, prefix: str, filename: str) -> str:
    parts = [alias.strip("/"), bucket.strip("/"), prefix.strip("/"), filename.strip("/")]
    return "/".join(p for p in parts if p)


def _normalize_object_name(filename: str, *, bucket: str, prefix: str) -> str:
    normalized = filename.strip("/").replace("\\", "/")
    bucket_part = bucket.strip("/")
    prefix_part = prefix.strip("/")

    if bucket_part and normalized.startswith(f"{bucket_part}/"):
        normalized = normalized[len(bucket_part) + 1 :]
    if prefix_part and normalized.startswith(f"{prefix_part}/"):
        normalized = normalized[len(prefix_part) + 1 :]
    return normalized


def _normalize_join(base: str, rel: str) -> str:
    cleaned_rel = rel.strip().lstrip("./")
    return os.path.normpath(f"{base.rstrip('/')}/{cleaned_rel}")
