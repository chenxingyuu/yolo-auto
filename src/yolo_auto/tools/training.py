from __future__ import annotations

import shlex
import time
from dataclasses import dataclass
from typing import Any

from yolo_auto.errors import err, ok
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.notifier_state_store import NotifierStateStore
from yolo_auto.remote_control import HttpControlClient, RemoteControlError
from yolo_auto.ssh_client import SSHClient, SSHRemoteExecutionError
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.status import (
    build_schema_card_with_mlflow_button,
    build_training_started_schema_card,
)


@dataclass(frozen=True)
class TrainRequest:
    job_id: str
    model: str
    data_config_path: str
    epochs: int
    img_size: int
    batch: int | float
    learning_rate: float | None
    work_dir: str
    jobs_dir: str
    env_id: str = "default"
    extra_args: dict[str, Any] | None = None
    minio_export_zip: str | None = None
    dataset_slug: str | None = None
    dataset_version_note: str | None = None


def _resolve_remote_path(path: str, work_dir: str) -> str:
    if path.startswith("/"):
        return path
    return f"{work_dir.rstrip('/')}/{path.lstrip('./')}"


def _format_yolo_value(value: Any) -> str:
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
    arg_parts: list[str] = []
    for key, raw_value in extra_args.items():
        if raw_value is None:
            continue
        arg_parts.append(f"{key}={_format_yolo_value(raw_value)}")
    return " ".join(arg_parts)


_START_CARD_ROW2_KEYS = frozenset({"optimizer", "device", "workers", "patience", "amp"})

_OPTIMIZER_AUTO_NOTICE = (
    "提示：optimizer=auto 时 Ultralytics 会忽略命令行中的 lr0、momentum 等，"
    "由训练器自动选定优化器与学习率；若未传 learningRate 则未写 lr0，"
    "卡片 lr0 列为占位；实际数值以 train.log 为准。"
)


def _is_optimizer_auto(extra_args: dict[str, Any] | None) -> bool:
    if not extra_args:
        return False
    raw = extra_args.get("optimizer")
    if raw is None:
        return False
    return str(raw).strip().casefold() == "auto"


def _format_scalar_for_card(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _extra_params_line_for_start_card(extra_args: dict[str, Any] | None) -> str | None:
    if not extra_args:
        return None
    pairs: list[tuple[str, str]] = []
    for key, raw in sorted(extra_args.items()):
        if raw is None or key in _START_CARD_ROW2_KEYS:
            continue
        pairs.append((key, _format_scalar_for_card(raw)))
    if not pairs:
        return None
    shown = pairs[:5]
    line = "其他：" + "，".join(f"{k}={v}" for k, v in shown)
    if len(pairs) > 5:
        line += " …"
    return line


def _dataset_provenance_from_request(req: TrainRequest) -> dict[str, Any] | None:
    parts: dict[str, Any] = {}
    z = (req.minio_export_zip or "").strip()
    if z:
        parts["minioExportZip"] = z
    s = (req.dataset_slug or "").strip()
    if s:
        parts["datasetSlug"] = s
    n = (req.dataset_version_note or "").strip()
    if n:
        parts["datasetVersionNote"] = n
    return parts or None


def _fourth_metric_for_start_card(extra_args: dict[str, Any] | None) -> tuple[str, str]:
    ex = extra_args or {}
    if ex.get("patience") is not None:
        return "Patience", str(ex["patience"])
    if ex.get("amp") is not None:
        return "AMP", _format_scalar_for_card(ex["amp"])
    return "Patience", ""


def start_training(
    req: TrainRequest,
    ssh_client: SSHClient | None,
    notifier: FeishuNotifier,
    state_store: JobStateStore,
    *,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    notifier_store: NotifierStateStore | None = None,
    mlflow_url: str | None = None,
    feishu_card_img_key: str | None = None,
    feishu_card_fallback_img_key: str | None = None,
    control_client: HttpControlClient | None = None,
) -> dict[str, object]:
    now = int(time.time())
    existing = state_store.get(req.job_id)
    if existing and existing.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
        return ok(existing.to_dict())

    model_abs_path = _resolve_remote_path(req.model, req.work_dir)
    model_q = shlex.quote(model_abs_path)
    ds_prov = _dataset_provenance_from_request(req)
    run_id = req.job_id
    job_dir = f"{req.jobs_dir}/{req.job_id}"
    log_path = f"{job_dir}/train.log"
    data_abs_path = _resolve_remote_path(req.data_config_path, req.work_dir)
    extra_cli_args = _build_extra_cli_args(req.extra_args)
    lr_cli = f" lr0={req.learning_rate}" if req.learning_rate is not None else ""
    tracking_uri = (mlflow_tracking_uri or "").strip()
    experiment_name = (mlflow_experiment_name or "").strip()
    if not tracking_uri:
        return err(
            error_code="MISSING_MLFLOW_TRACKING_URI",
            message="mlflow tracking uri is empty",
            retryable=False,
            hint="请在服务端环境变量配置 MLFLOW_TRACKING_URI（训练端需可访问同一 Tracking Server）",
            payload={"jobId": req.job_id},
        )
    if not experiment_name:
        return err(
            error_code="MISSING_MLFLOW_EXPERIMENT_NAME",
            message="mlflow experiment name is empty",
            retryable=False,
            hint="请在服务端环境变量配置 MLFLOW_EXPERIMENT_NAME",
            payload={"jobId": req.job_id},
        )
    train_cmd = (
        f"mkdir -p {job_dir} && cd {req.work_dir} && "
        f"export MLFLOW_TRACKING_URI={shlex.quote(tracking_uri)} && "
        f"export MLFLOW_EXPERIMENT_NAME={shlex.quote(experiment_name)} && "
        f"export MLFLOW_RUN_NAME={shlex.quote(req.job_id)} && "
        f"yolo detect train model={req.model} data={req.data_config_path} "
        f"epochs={req.epochs} imgsz={req.img_size} batch={req.batch}{lr_cli} "
        f"project={req.jobs_dir} name={req.job_id} exist_ok=True {extra_cli_args} > {log_path} 2>&1"
    )
    try:
        if control_client is not None:
            remote = control_client.start_training(
                {
                    "jobId": req.job_id,
                    "runName": req.job_id,
                    "modelPath": model_abs_path,
                    "dataPath": data_abs_path,
                    "project": req.jobs_dir,
                    "name": req.job_id,
                    "device": (req.extra_args or {}).get("device"),
                    "epochs": req.epochs,
                    "imgsz": req.img_size,
                    "batch": req.batch,
                    "extraArgs": req.extra_args,
                    "workDir": req.work_dir,
                    "mlflowTrackingUri": tracking_uri,
                    "mlflowExperimentName": experiment_name,
                }
            )
            pid = str(remote.get("pid") or remote.get("executionId") or "")
            if not pid:
                raise RemoteControlError(
                    "REMOTE_INVALID_RESPONSE",
                    "missing pid/executionId in start response",
                    retryable=False,
                    hint="检查控制面 train/start 返回格式",
                )
        else:
            if ssh_client is None:
                return err(
                    error_code="REMOTE_CLIENT_MISSING",
                    message="ssh client is required when control client is disabled",
                    retryable=False,
                    hint="请检查 YOLO_REMOTE_MODE 配置",
                    payload={"jobId": req.job_id},
                )
            _, _, model_exists_code = ssh_client.execute(f"test -f {model_q}")
            if model_exists_code != 0:
                return err(
                    error_code="MODEL_NOT_FOUND",
                    message=f"model not found: {model_abs_path}",
                    retryable=False,
                    hint="请先确认模型路径存在，或先调用 yolo_setup_env 做环境预检查",
                    payload={"jobId": req.job_id, "modelPath": model_abs_path},
                )
            pid, _ = ssh_client.execute_background(train_cmd)
    except SSHRemoteExecutionError as exc:
        return err(
            error_code=exc.error_code,
            message=str(exc),
            retryable=exc.retryable,
            hint=exc.hint,
            payload={"jobId": req.job_id},
        )
    except RemoteControlError as exc:
        return err(
            error_code=exc.error_code,
            message=str(exc),
            retryable=exc.retryable,
            hint=exc.hint,
            payload={"jobId": req.job_id, **exc.payload},
        )
    except Exception as exc:
        return err(
            error_code="START_FAILED",
            message=str(exc),
            retryable=True,
            hint="检查 SSH 连通性、数据路径和远程 yolo 命令可用性",
            payload={"jobId": req.job_id},
        )
    record = JobRecord(
        job_id=req.job_id,
        run_id=run_id,
        status=JobStatus.RUNNING,
        pid=str(pid),
        paths={
            "jobDir": job_dir,
            "logPath": log_path,
            "metricsPath": f"{job_dir}/results.csv",
            "bestPath": f"{job_dir}/weights/best.pt",
            "lastPath": f"{job_dir}/weights/last.pt",
            "modelPath": model_abs_path,
            "dataConfigPath": data_abs_path,
        },
        created_at=now if not existing else existing.created_at,
        updated_at=now,
        env_id=req.env_id,
        last_notified_state=existing.last_notified_state if existing else None,
        last_metrics_at=existing.last_metrics_at if existing else None,
        train_epochs=req.epochs,
        last_reported_epoch=0,
        dataset_provenance=ds_prov,
    )
    state_store.upsert(record)
    ex = req.extra_args or {}
    fourth_label, fourth_val = _fourth_metric_for_start_card(req.extra_args)
    optimizer_auto = _is_optimizer_auto(req.extra_args)
    training_hints = [_OPTIMIZER_AUTO_NOTICE] if optimizer_auto else []
    lr_text = f"{req.learning_rate:g}" if req.learning_rate is not None else ""
    if isinstance(req.batch, float) and req.batch == int(req.batch):
        batch_text = str(int(req.batch))
    else:
        batch_text = str(req.batch)
    started_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    card = build_training_started_schema_card(
        epochs=req.epochs,
        imgsz=req.img_size,
        batch_display=batch_text,
        lr_display=lr_text,
        optimizer_display=_format_scalar_for_card(ex.get("optimizer")),
        device_display=_format_scalar_for_card(ex.get("device")),
        workers_display=_format_scalar_for_card(ex.get("workers")),
        fourth_metric_label=fourth_label,
        fourth_metric_value=fourth_val,
        extra_params_line=_extra_params_line_for_start_card(req.extra_args),
        optimizer_auto_notice=_OPTIMIZER_AUTO_NOTICE if optimizer_auto else None,
        started_time_text=started_text,
        mlflow_url=mlflow_url,
        top_img_key=feishu_card_img_key,
        top_img_fallback_key=feishu_card_fallback_img_key,
    )
    message_id = notifier.send_schema_card_with_message_id(card=card)
    if notifier_store is not None and message_id:
        notifier_store.upsert(
            job_id=req.job_id,
            now_ts=now,
            feishu_message_id=message_id,
            last_notified_state=JobStatus.RUNNING,
        )
    payload = record.to_dict()
    if training_hints:
        payload = {**payload, "trainingHints": training_hints}
    return ok(payload)


def stop_training(
    job_id: str,
    run_id: str,
    ssh_client: SSHClient | None,
    notifier: FeishuNotifier,
    state_store: JobStateStore,
    *,
    notifier_store: NotifierStateStore | None = None,
    mlflow_url: str | None = None,
    control_client: HttpControlClient | None = None,
) -> dict[str, object]:
    now = int(time.time())
    record = state_store.get(job_id)
    effective_run_id = record.run_id if record else run_id
    if record and record.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED}:
        return ok(record.to_dict())

    pid = record.pid if record else ""
    if (
        control_client is None
        and pid
        and ssh_client is not None
        and not ssh_client.process_alive(pid)
    ):
        updated = state_store.update_status(job_id, JobStatus.STOPPED, now) if record else None
        if updated:
            return ok(updated.to_dict())
        return ok({"jobId": job_id, "runId": effective_run_id, "status": JobStatus.STOPPED.value})
    if control_client is not None:
        try:
            stopped = control_client.stop_training(
                {"jobId": job_id, "executionId": pid if pid else None}
            )
            if str(stopped.get("status", "")).strip() not in {"stopped", "already_stopped"}:
                return err(
                    error_code="STOP_FAILED",
                    message=f"unexpected stop response: {stopped}",
                    retryable=True,
                    hint="稍后重试 stop",
                    payload={"jobId": job_id, "runId": effective_run_id},
                )
        except RemoteControlError as exc:
            return err(
                error_code=exc.error_code,
                message=str(exc),
                retryable=exc.retryable,
                hint=exc.hint,
                payload={"jobId": job_id, "runId": effective_run_id, **exc.payload},
            )
    else:
        if ssh_client is None:
            return err(
                error_code="REMOTE_CLIENT_MISSING",
                message="ssh client is required when control client is disabled",
                retryable=False,
                hint="请检查 YOLO_REMOTE_MODE 配置",
                payload={"jobId": job_id, "runId": effective_run_id},
            )
        kill_cmd = f"pkill -f \"name={job_id}\""
        _, stderr_text, exit_code = ssh_client.execute(kill_cmd)
        if exit_code != 0:
            return err(
                error_code="STOP_FAILED",
                message=stderr_text.strip() or "unable to stop training process",
                retryable=True,
                hint="确认 jobId 是否正确，或稍后重试 stop",
                payload={"jobId": job_id, "runId": effective_run_id},
            )

    if record:
        updated = state_store.update_status(job_id, JobStatus.STOPPED, now)
        last_notified = updated.last_notified_state
        if notifier_store is not None:
            st = notifier_store.get(job_id)
            last_notified = st.last_notified_state if st else last_notified
        if last_notified != JobStatus.STOPPED:
            card = build_schema_card_with_mlflow_button(
                title="[YOLO] 训练已停止",
                header_template="orange",
                md_text=f"job={job_id}\nrunId={effective_run_id}",
                mlflow_url=mlflow_url,
                button_element_id="mlflow_stop_btn",
            )
            message_id = notifier.send_schema_card_with_message_id(card=card)
            if notifier_store is not None:
                notifier_store.upsert(
                    job_id=job_id,
                    now_ts=now,
                    feishu_message_id=message_id,
                    last_notified_state=JobStatus.STOPPED,
                )
            else:
                if message_id:
                    state_store.mark_feishu_message(job_id, message_id, now)
                state_store.mark_notified(job_id, JobStatus.STOPPED, now)
        return ok(updated.to_dict())
    card = build_schema_card_with_mlflow_button(
        title="[YOLO] 训练已停止",
        header_template="orange",
        md_text=f"job={job_id}\nrunId={effective_run_id}",
        mlflow_url=mlflow_url,
        button_element_id="mlflow_stop_btn",
    )
    _ = notifier.send_schema_card_with_message_id(card=card)
    return ok({"jobId": job_id, "runId": effective_run_id, "status": JobStatus.STOPPED.value})

