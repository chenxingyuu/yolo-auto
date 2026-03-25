from __future__ import annotations

import re
import shlex
from typing import Any

from yolo_auto.errors import err, ok
from yolo_auto.models import JobStatus
from yolo_auto.ssh_client import SSHClient
from yolo_auto.state_store import JobStateStore

_ALL_LINE_RE = re.compile(
    r"^\s*all\s+\d+\s+\d+\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s*$",
    re.MULTILINE,
)


def _format_yolo_value(value: Any) -> str:
    # Ultralytics CLI 参数使用 key=value 形式，字符串需要 quote，数字/布尔可直接输出。
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        list_repr = "[" + ",".join(_format_yolo_value(item) for item in value) + "]"
        return shlex.quote(list_repr)
    return shlex.quote(str(value))


def _build_extra_cli_args(extra_args: dict[str, Any] | None) -> str:
    if not extra_args:
        return ""
    parts: list[str] = []
    for key, raw_value in extra_args.items():
        if raw_value is None:
            continue
        parts.append(f"{key}={_format_yolo_value(raw_value)}")
    return " ".join(parts)


def _parse_val_stdout(stdout: str) -> dict[str, float] | None:
    match = _ALL_LINE_RE.search(stdout)
    if not match:
        return None
    precision = float(match.group(1))
    recall = float(match.group(2))
    map50 = float(match.group(3))
    map5095 = float(match.group(4))
    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map5095": map5095,
    }


def run_validation(
    job_id: str,
    state_store: JobStateStore,
    ssh_client: SSHClient,
    jobs_dir: str,
    work_dir: str,
    *,
    data_config_path: str | None = None,
    img_size: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    extra_args: dict[str, Any] | None = None,
) -> dict[str, object]:
    record = state_store.get(job_id)
    if not record:
        return err(
            error_code="JOB_NOT_FOUND",
            message=f"job not found: {job_id}",
            retryable=False,
            hint="请先启动训练或检查 jobId",
            payload={"jobId": job_id},
        )

    if record.status != JobStatus.COMPLETED:
        return err(
            error_code="JOB_NOT_COMPLETED",
            message=f"job not completed: {job_id}",
            retryable=False,
            hint="请先等待训练完成后再调用 yolo_validate",
            payload=record.to_dict(),
        )

    best_path = record.paths.get("bestPath", "")
    if not best_path:
        return err(
            error_code="MISSING_BEST_MODEL",
            message="bestPath missing from job record",
            retryable=False,
            hint="请确认训练过程中 best 权重已生成，或检查 JobStateStore 的 paths 字段",
            payload=record.to_dict(),
        )

    if not ssh_client.file_exists(best_path):
        return err(
            error_code="BEST_MODEL_NOT_FOUND",
            message="best model file not found on remote host",
            retryable=False,
            hint="请确认远程权重路径可访问，并且权限允许读取",
            payload={"jobId": job_id, "modelPath": best_path},
        )

    effective_data_config = data_config_path or record.paths.get("dataConfigPath", "")
    if not effective_data_config:
        return err(
            error_code="MISSING_DATA_CONFIG",
            message="dataConfigPath missing for validation",
            retryable=False,
            hint="请传入 dataConfigPath 参数，或在训练后确保 job record 中写入 dataConfigPath",
            payload={"jobId": job_id},
        )

    extra_cli_args = _build_extra_cli_args(extra_args)
    cmd = [
        f"cd {shlex.quote(work_dir)}",
        "&&",
        "yolo detect val",
        f"model={shlex.quote(best_path)}",
        f"data={shlex.quote(effective_data_config)}",
        f"project={shlex.quote(jobs_dir)}",
        f"name={shlex.quote(f'val-{job_id}')}",
    ]
    if img_size is not None:
        cmd.append(f"imgsz={img_size}")
    if batch is not None:
        cmd.append(f"batch={batch}")
    if device is not None:
        cmd.append(f"device={shlex.quote(device)}")
    if extra_cli_args:
        cmd.append(extra_cli_args)

    val_cmd = " ".join(cmd)
    stdout_text, stderr_text, exit_code = ssh_client.execute(val_cmd, timeout=300)
    if exit_code != 0:
        return err(
            error_code="VALIDATION_FAILED",
            message=stderr_text.strip() or stdout_text.strip() or "yolo val failed",
            retryable=False,
            hint="检查远程路径、数据集 YAML 与权重文件是否可用",
            payload={"jobId": job_id},
        )

    metrics = _parse_val_stdout(stdout_text)
    if not metrics:
        return err(
            error_code="VALIDATION_PARSE_FAILED",
            message="unable to parse yolo val stdout",
            retryable=False,
            hint="检查远程输出格式是否与 Ultralytics 版本一致（all 行可能不同）",
            payload={"jobId": job_id},
        )

    return ok(
        {
            "jobId": job_id,
            "modelPath": best_path,
            "metrics": metrics,
            "rawOutput": stdout_text,
        }
    )

