from __future__ import annotations

import json
import shlex
import time
from typing import TYPE_CHECKING, Any

from yolo_auto.errors import err, ok
from yolo_auto.models import JobStatus
from yolo_auto.ssh_client import SSHClient
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.model_ref import parse_model_ref, resolve_model_ref_to_record

if TYPE_CHECKING:
    from yolo_auto.tracker import MLflowTracker


def _format_yolo_value(value: Any) -> str:
    # Ultralytics CLI uses key=value. For strings, quote; for numbers/bools, keep as-is.
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


_FORMAT_TO_FIND_QUERY: dict[str, str] = {
    "onnx": r"*.onnx",
    "engine": r"*.engine",
    "coreml": r"*.mlpackage",
}


def _find_export_artifacts(
    ssh_client: SSHClient,
    job_dir: str,
    *,
    formats: list[str],
) -> list[dict[str, str]]:
    # 尽量在 job_dir 下搜，避免依赖 ulralytics exporter 的输出目录规则。
    artifacts: list[dict[str, str]] = []
    for fmt in formats:
        pattern = _FORMAT_TO_FIND_QUERY.get(fmt)
        if not pattern:
            continue
        cmd = (
            f"cd {shlex.quote(job_dir)} && "
            f"find . -maxdepth 4 \\( -name {shlex.quote(pattern)} \\) -print | head -50"
        )
        stdout, _, code = ssh_client.execute(cmd, timeout=10)
        if code != 0 or not stdout.strip():
            continue
        for line in stdout.splitlines():
            p = line.strip()
            if not p:
                continue
            artifacts.append({"format": fmt, "path": f"{job_dir}/{p.lstrip('./')}"})
    return artifacts


def run_export(
    job_id: str | None,
    state_store: JobStateStore,
    ssh_client: SSHClient,
    jobs_dir: str,
    work_dir: str,
    *,
    formats: list[str] | None = None,
    img_size: int | None = None,
    half: bool | None = None,
    int8: bool | None = None,
    device: str | None = None,
    extra_args: dict[str, Any] | None = None,
    model_ref: str | None = None,
    registry_tracker: MLflowTracker | None = None,
) -> dict[str, object]:
    model_ref_meta: dict[str, str] | None = None
    effective_job_id = (job_id or "").strip() or None
    raw_ref = (model_ref or "").strip()

    if raw_ref:
        if registry_tracker is None:
            return err(
                error_code="MODEL_REF_UNSUPPORTED",
                message="modelRef 需要启用模型注册并传入 tracker",
                retryable=False,
                hint="设置 MLFLOW_MODEL_REGISTRY_ENABLE=true",
                payload={},
            )
        parsed = parse_model_ref(raw_ref)
        if not parsed:
            return err(
                error_code="INVALID_MODEL_REF",
                message=f"无法解析 modelRef: {raw_ref}",
                retryable=False,
                hint="示例：`yolo-default-coco-yolo11n:approved`",
                payload={"modelRef": raw_ref},
            )
        name, alias = parsed
        resolved = resolve_model_ref_to_record(
            registry_tracker,
            state_store,
            model_name=name,
            alias=alias,
        )
        if not resolved:
            return err(
                error_code="MODEL_REF_UNRESOLVED",
                message="registry alias 无法解析到本地 job",
                retryable=False,
                hint="请改用 jobId，或确认该版本训练任务仍在 YOLO_STATE_FILE 中",
                payload={"modelRef": raw_ref, "modelName": name, "alias": alias},
            )
        effective_job_id = resolved.job_id
        model_ref_meta = {
            "modelName": name,
            "alias": alias,
            "resolvedJobId": effective_job_id,
        }
    elif not effective_job_id:
        return err(
            error_code="MISSING_JOB_OR_MODEL_REF",
            message="jobId 与 modelRef 至少填一个",
            retryable=False,
            hint='示例 modelRef：`yolo-default-coco-yolo11n:approved`',
            payload={},
        )

    record = state_store.get(effective_job_id)
    if not record:
        return err(
            error_code="JOB_NOT_FOUND",
            message=f"job not found: {effective_job_id}",
            retryable=False,
            hint="请先启动训练并等待完成",
            payload={"jobId": effective_job_id},
        )
    if record.status != JobStatus.COMPLETED:
        return err(
            error_code="JOB_NOT_COMPLETED",
            message=f"job not completed: {effective_job_id}",
            retryable=False,
            hint="请先等待训练完成再调用 yolo_export",
            payload=record.to_dict(),
        )

    best_path = record.paths.get("bestPath", "")
    if not best_path:
        return err(
            error_code="MISSING_BEST_MODEL",
            message="bestPath missing from job record",
            retryable=False,
            hint="请确认 JobRecord.paths.bestPath 字段可用",
            payload=record.to_dict(),
        )
    if not ssh_client.file_exists(best_path):
        return err(
            error_code="BEST_MODEL_NOT_FOUND",
            message="best model file not found on remote host",
            retryable=False,
            hint="请确认远程权重路径可访问",
            payload={"jobId": effective_job_id, "modelPath": best_path},
        )

    job_dir = record.paths.get("jobDir", f"{jobs_dir}/{effective_job_id}")

    effective_formats = formats or ["onnx", "engine", "coreml"]
    # 降噪：格式名标准化
    normalized_formats = [str(f).strip().lower() for f in effective_formats if str(f).strip()]

    extra_cli_args = _build_extra_cli_args(extra_args)

    export_errors: list[dict[str, Any]] = []
    # 为了尽量减少重复搜索，每次只找到当前 format 的 artefacts。
    for fmt in normalized_formats:
        project_arg = (
            f"project={shlex.quote(job_dir)} "
            f"name={shlex.quote(f'export-{fmt}-{effective_job_id}')}"
        )
        cmd_common = (
            f"cd {shlex.quote(work_dir)} && "
            f"yolo export model={shlex.quote(best_path)} format={shlex.quote(fmt)}"
        )
        if img_size is not None:
            cmd_common += f" imgsz={img_size}"
        if half is not None:
            cmd_common += f" half={_format_yolo_value(half)}"
        if int8 is not None:
            cmd_common += f" int8={_format_yolo_value(int8)}"
        if device is not None:
            cmd_common += f" device={shlex.quote(str(device))}"
        if extra_cli_args:
            cmd_common += f" {extra_cli_args}"

        # 优先尝试把输出写到 job_dir
        cmd_with_project = f"{cmd_common} {project_arg}"
        stdout, stderr, exit_code = ssh_client.execute(cmd_with_project, timeout=1800)
        if exit_code != 0:
            # fallback：不带 project/name（有些版本可能不支持）
            cmd_without_project = cmd_common
            stdout2, stderr2, exit_code2 = ssh_client.execute(cmd_without_project, timeout=1800)
            if exit_code2 != 0:
                export_errors.append(
                    {
                        "format": fmt,
                        "error": stderr2.strip() or stderr.strip() or "yolo export failed",
                    }
                )
                continue

        # 每个 format 成功后再搜一遍当前 job_dir 的产物
        # 为了减少时间消耗，这里不做严格去重（find + head 已限制结果）。
        _ = stdout  # noqa: F841 (用于调试时可保留扩展)

    artifacts = _find_export_artifacts(ssh_client, job_dir, formats=normalized_formats)
    if export_errors and not artifacts:
        return err(
            error_code="EXPORT_FAILED",
            message="yolo export failed for all formats",
            retryable=False,
            hint="检查远程权重路径、Ultralytics exporter 可用性与显存/权限",
            payload={"jobId": effective_job_id, "errors": export_errors},
        )

    exported_at = int(time.time())
    out: dict[str, object] = {
        "jobId": effective_job_id,
        "modelPath": best_path,
        "artifacts": artifacts,
        "errors": export_errors,
        "exportedAt": exported_at,
    }
    if model_ref_meta:
        out["modelRef"] = model_ref_meta
    manifest: dict[str, Any] = {
        "schema": "yolo-auto.export-manifest/v1",
        "jobId": effective_job_id,
        "runId": record.run_id,
        "exportedAt": exported_at,
        "modelPath": best_path,
        "formatsRequested": normalized_formats,
        "artifacts": artifacts,
        "errors": export_errors,
    }
    manifest_name = "export-manifest.json"
    manifest_remote = f"{job_dir.rstrip('/')}/{manifest_name}"
    try:
        payload = json.dumps(manifest, ensure_ascii=True, indent=2).encode("utf-8")
        ssh_client.upload_bytes(payload, manifest_remote)
        out["exportManifestPath"] = manifest_remote
        out["exportManifest"] = manifest
    except Exception as exc:
        out["exportManifestWarning"] = str(exc)
    return ok(out)

