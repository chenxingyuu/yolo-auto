from __future__ import annotations

import base64
import csv
import logging
import shlex
import time
from io import BytesIO, StringIO
from typing import Any, Literal

from yolo_auto.errors import err, ok
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobStatus
from yolo_auto.notifier_state_store import NotifierStateStore
from yolo_auto.remote_control import HttpControlClient, RemoteControlError
from yolo_auto.ssh_client import SSHClient, SSHRemoteExecutionError
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.metrics_csv import metric_value_from_parsed, parse_training_row
from yolo_auto.tracker import MLflowTracker

logger = logging.getLogger(__name__)


def _build_mlflow_button_elements(
    mlflow_url: str | None,
    *,
    element_id: str = "mlflow_open_btn",
) -> list[dict[str, Any]]:
    if not mlflow_url:
        return []
    return [
        {
            "tag": "button",
            "element_id": element_id,
            "type": "primary_filled",
            "size": "small",
            "text": {"tag": "plain_text", "content": "Chrome浏览器打开"},
            "behaviors": [
                {
                    "type": "open_url",
                    "default_url": mlflow_url,
                    "pc_url": mlflow_url,
                    "android_url": mlflow_url,
                    "ios_url": mlflow_url,
                }
            ],
        }
    ]


def _build_url_button_element(
    url: str,
    *,
    label: str,
    element_id: str,
) -> dict[str, Any]:
    return {
        "tag": "button",
        "element_id": element_id,
        "type": "primary_filled",
        "size": "small",
        "text": {"tag": "plain_text", "content": label},
        "behaviors": [
            {
                "type": "open_url",
                "default_url": url,
                "pc_url": url,
                "android_url": url,
                "ios_url": url,
            }
        ],
    }


def _right_aligned_button_column_set(
    button_elements: list[dict[str, Any]],
) -> dict[str, Any]:
    """把按钮组件右侧对齐到卡片容器右边。"""
    return {
        "tag": "column_set",
        "flex_mode": "flow",
        "horizontal_align": "right",
        "horizontal_spacing": "default",
        "margin": "0px",
        "columns": [
            {
                "tag": "column",
                "width": "auto",
                "vertical_align": "top",
                "elements": [],
            },
            {
                "tag": "column",
                "width": "auto",
                "vertical_align": "top",
                "elements": button_elements,
            },
        ],
    }


def build_schema_card_with_mlflow_button(
    *,
    title: str,
    header_template: Literal["blue", "green", "red", "orange"],
    md_text: str,
    mlflow_url: str | None,
    button_element_id: str = "mlflow_open_btn",
) -> dict[str, Any]:
    """构建简单 Schema 2.0 卡片：markdown + 右对齐按钮 + 更新时间。"""
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": md_text,
        }
    ]
    button_elements = _build_mlflow_button_elements(
        mlflow_url, element_id=button_element_id
    )
    if button_elements:
        elements.append(_right_aligned_button_column_set(button_elements))
    now_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    elements.append(
        {
            "tag": "markdown",
            "content": f"<font color='grey'>更新时间：{now_text}</font>",
            "text_align": "right",
        }
    )
    return {
        "schema": "2.0",
        "header": {
            "template": header_template,
            "padding": "12px 12px 12px 12px",
            "icon": {"tag": "standard_icon", "token": "code", "color": header_template},
            "title": {"tag": "plain_text", "content": title},
        },
        "body": {"elements": elements},
    }


def _training_rows_to_chart_spec(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    values: list[dict[str, Any]] = []
    for row in rows:
        try:
            parsed = parse_training_row(row)
            ep = int(parsed["epoch"])
        except Exception:
            continue
        if ep <= 0:
            continue
        ep_s = str(ep)
        m50 = float(parsed["map50"])
        m95 = float(parsed["map5095"])
        rec = float(parsed["recall"])
        values.append({"epoch": ep_s, "series": "mAP50", "value": m50})
        values.append({"epoch": ep_s, "series": "mAP50-95", "value": m95})
        values.append({"epoch": ep_s, "series": "Recall", "value": rec})
    if not values:
        return None
    return {
        "type": "line",
        "title": {"text": "训练指标"},
        "data": {"values": values},
        "xField": "epoch",
        "yField": "value",
        "seriesField": "series",
        "legends": {"visible": True, "orient": "bottom"},
        # 关闭折线点/圆点显示，避免在点较多时视觉拥挤
        "point": {"visible": False},
        # 关闭点/折线的数值标签（如在点上显示 value 文本）
        "label": {"visible": False},
    }


def _metric_column(
    *,
    value: str,
    label: str,
    value_color: str,
    background_style: str,
    weight: int = 1,
) -> dict[str, Any]:
    return {
        "tag": "column",
        "width": "weighted",
        "weight": weight,
        "padding": "12px",
        "vertical_spacing": "2px",
        "background_style": background_style,
        "elements": [
            {
                "tag": "markdown",
                "content": f"## <font color='{value_color}'>{value}</font>",
                "text_align": "center",
            },
            {
                "tag": "markdown",
                "content": f"<font color='grey'>{label}</font>",
                "text_align": "center",
                "text_size": "normal",
            },
        ],
    }


def _build_training_schema_card(
    *,
    title: str,
    header_template: str,
    epoch_text: str,
    elapsed_text: str,
    eta_text: str,
    updated_time_text: str,
    mlflow_url: str | None = None,
    top_img_key: str | None = None,
    top_img_fallback_key: str | None = None,
    training_rows: list[dict[str, Any]] | None = None,
    registry_markdown: str | None = None,
    registry_button_specs: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = []
    if top_img_key:
        image_element: dict[str, Any] = {
            "tag": "img",
            "img_key": top_img_key,
            "alt": {"tag": "plain_text", "content": "training cover"},
        }
        if top_img_fallback_key:
            image_element["fallback_img_key"] = top_img_fallback_key
        elements.append(image_element)
    chart_spec = (
        _training_rows_to_chart_spec(training_rows)
        if training_rows
        else None
    )
    if chart_spec is not None:
        elements.append(
            {
                "tag": "chart",
                "element_id": "train_metrics_chart",
                "aspect_ratio": "16:9",
                "color_theme": "brand",
                "chart_spec": chart_spec,
            }
        )
    elements.append(
        {
            "tag": "column_set",
            "flex_mode": "stretch",
            "horizontal_spacing": "12px",
            "margin": "0px",
            "columns": [
                _metric_column(
                    value=epoch_text,
                    label="Epoch",
                    value_color="blue",
                    background_style="blue-50",
                ),
                _metric_column(
                    value=f"{elapsed_text} / {eta_text}",
                    label="Elapsed / ETA",
                    value_color="blue",
                    background_style="blue-50",
                    weight=2,
                ),
            ],
        }
    )
    if registry_markdown:
        elements.append({"tag": "markdown", "content": registry_markdown})
    mlflow_buttons = _build_mlflow_button_elements(mlflow_url)
    extra_btns: list[dict[str, Any]] = []
    if registry_button_specs:
        for bid, label, url in registry_button_specs:
            if url.strip():
                extra_btns.append(
                    _build_url_button_element(url.strip(), label=label, element_id=bid)
                )
    all_buttons = mlflow_buttons + extra_btns
    if all_buttons:
        # 采用 bisect 双列：左列留空，按钮放右列，确保右侧对齐。
        # 这里使用 flex_mode=flow，尽量贴近飞书官方示例的兼容写法。
        elements.append(
            {
                "tag": "column_set",
                "flex_mode": "flow",
                "horizontal_align": "right",
                "horizontal_spacing": "default",
                "margin": "0px",
                "columns": [
                    {
                        "tag": "column",
                        "width": "auto",
                        "vertical_align": "top",
                        "elements": [],
                    },
                    {
                        "tag": "column",
                        "width": "auto",
                        "vertical_align": "top",
                        "elements": all_buttons,
                    }
                ],
            }
        )
    elements.append(
        {
            "tag": "markdown",
            "content": f"<font color='grey'>更新时间：{updated_time_text}</font>",
            "text_align": "right",
        }
    )
    return {
        "schema": "2.0",
        "header": {
            "template": header_template,
            "padding": "12px 12px 12px 12px",
            "icon": {"tag": "standard_icon", "token": "code", "color": header_template},
            "title": {"tag": "plain_text", "content": title},
        },
        "body": {"elements": elements},
    }


def _training_started_cell(value: str) -> str:
    t = str(value).strip()
    return t if t else "—"


def build_training_started_schema_card(
    *,
    epochs: int,
    imgsz: int,
    batch_display: str,
    lr_display: str,
    optimizer_display: str,
    device_display: str,
    workers_display: str,
    fourth_metric_label: str,
    fourth_metric_value: str,
    extra_params_line: str | None = None,
    optimizer_auto_notice: str | None = None,
    started_time_text: str,
    mlflow_url: str | None = None,
    top_img_key: str | None = None,
    top_img_fallback_key: str | None = None,
    title: str = "YOLO 训练已启动",
    header_template: str = "blue",
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = []
    if top_img_key:
        img_el: dict[str, Any] = {
            "tag": "img",
            "img_key": top_img_key,
            "alt": {"tag": "plain_text", "content": "training cover"},
        }
        if top_img_fallback_key:
            img_el["fallback_img_key"] = top_img_fallback_key
        elements.append(img_el)
    if optimizer_auto_notice:
        elements.append(
            {
                "tag": "markdown",
                "content": f"<font color='grey'>{optimizer_auto_notice}</font>",
            }
        )
    elements.extend(
        [
            {
                "tag": "column_set",
                "flex_mode": "stretch",
                "horizontal_spacing": "12px",
                "margin": "0px",
                "columns": [
                    _metric_column(
                        value=_training_started_cell(str(epochs)),
                        label="Epochs",
                        value_color="blue",
                        background_style="blue-50",
                    ),
                    _metric_column(
                        value=_training_started_cell(str(imgsz)),
                        label="imgsz",
                        value_color="blue",
                        background_style="blue-50",
                    ),
                    _metric_column(
                        value=_training_started_cell(batch_display),
                        label="Batch",
                        value_color="blue",
                        background_style="blue-50",
                    ),
                    _metric_column(
                        value=_training_started_cell(lr_display),
                        label="lr0",
                        value_color="blue",
                        background_style="blue-50",
                    ),
                ],
            },
            {
                "tag": "column_set",
                "flex_mode": "stretch",
                "horizontal_spacing": "12px",
                "margin": "0px",
                "columns": [
                    _metric_column(
                        value=_training_started_cell(optimizer_display),
                        label="优化器",
                        value_color="violet",
                        background_style="violet-50",
                    ),
                    _metric_column(
                        value=_training_started_cell(device_display),
                        label="Device",
                        value_color="violet",
                        background_style="violet-50",
                    ),
                    _metric_column(
                        value=_training_started_cell(workers_display),
                        label="Workers",
                        value_color="violet",
                        background_style="violet-50",
                    ),
                    _metric_column(
                        value=_training_started_cell(fourth_metric_value),
                        label=fourth_metric_label,
                        value_color="violet",
                        background_style="violet-50",
                    ),
                ],
            },
        ]
    )
    if extra_params_line:
        elements.append(
            {
                "tag": "markdown",
                "content": f"<font color='grey'>{extra_params_line}</font>",
            }
        )
    start_buttons = _build_mlflow_button_elements(
        mlflow_url, element_id="mlflow_start_btn"
    )
    if start_buttons:
        elements.append(
            {
                "tag": "column_set",
                "flex_mode": "flow",
                "horizontal_align": "right",
                "horizontal_spacing": "default",
                "margin": "0px",
                "columns": [
                    {
                        "tag": "column",
                        "width": "auto",
                        "vertical_align": "top",
                        "elements": [],
                    },
                    {
                        "tag": "column",
                        "width": "auto",
                        "vertical_align": "top",
                        "elements": start_buttons,
                    }
                ],
            }
        )
    elements.append(
        {
            "tag": "markdown",
            "content": f"<font color='grey'>启动时间：{started_time_text}</font>",
            "text_align": "right",
        }
    )
    return {
        "schema": "2.0",
        "header": {
            "template": header_template,
            "padding": "12px 12px 12px 12px",
            "icon": {"tag": "standard_icon", "token": "code", "color": header_template},
            "title": {"tag": "plain_text", "content": title},
        },
        "body": {"elements": elements},
    }


def _format_duration(seconds: int) -> str:
    seconds = max(int(seconds), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_card_timestamp(now_ts: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))


def _format_signed(value: float) -> str:
    return f"{value:+.4f}"


def _estimate_eta_seconds(*, epoch: int, total_epochs: int, elapsed_seconds: int) -> int | None:
    if epoch <= 0 or total_epochs <= epoch:
        return None
    avg_epoch_seconds = elapsed_seconds / max(epoch, 1)
    remaining_epochs = max(total_epochs - epoch, 0)
    return int(avg_epoch_seconds * remaining_epochs)


def _pick_row_for_epoch(rows: list[dict[str, Any]], target_epoch: int) -> dict[str, Any] | None:
    if target_epoch <= 0:
        return None
    candidate: dict[str, Any] | None = None
    for row in rows:
        try:
            parsed = parse_training_row(row)
            row_epoch = int(parsed["epoch"])
        except Exception:
            continue
        if row_epoch <= target_epoch:
            candidate = row
        else:
            break
    return candidate


def _generate_loss_map_chart_png_b64(
    rows: list[dict[str, Any]],
    *,
    primary_metric_key: str,
) -> str | None:
    # matplotlib 仅在训练完成时才需要，避免影响训练过程性能与依赖缺失导致整个链路失败。
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    epochs: list[int] = []
    losses: list[float] = []
    maps: list[float] = []

    for row in rows:
        parsed = parse_training_row(row)
        epoch = int(parsed["epoch"])
        if epoch <= 0:
            continue
        epochs.append(epoch)
        losses.append(float(parsed["loss"]))
        maps.append(float(metric_value_from_parsed(parsed, primary_metric_key)))

    if len(epochs) < 2:
        return None

    # 控制点数，避免 base64 过大。
    max_points = 50
    if len(epochs) > max_points:
        step = max(1, len(epochs) // max_points)
        epochs = epochs[::step]
        losses = losses[::step]
        maps = maps[::step]

    fig, (ax_loss, ax_map) = plt.subplots(
        2,
        1,
        figsize=(6, 5),
        dpi=100,
        sharex=True,
    )
    ax_loss.plot(epochs, losses, linewidth=1.5)
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linewidth=0.3)

    ax_map.plot(epochs, maps, linewidth=1.5)
    ax_map.set_ylabel(primary_metric_key)
    ax_map.set_xlabel("epoch")
    ax_map.grid(True, linewidth=0.3)

    fig.tight_layout()

    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    finally:
        plt.close(fig)

    png_bytes = buf.getvalue()
    if not png_bytes:
        return None
    return base64.b64encode(png_bytes).decode("ascii")


def _generate_loss_map_chart_png(
    rows: list[dict[str, Any]],
    *,
    primary_metric_key: str,
) -> bytes | None:
    b64 = _generate_loss_map_chart_png_b64(rows, primary_metric_key=primary_metric_key)
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def _upsert_training_card(
    *,
    record,
    now_ts: int,
    state_store: JobStateStore,
    notifier_store: NotifierStateStore | None,
    notifier: FeishuNotifier,
    card: dict[str, Any],
):
    message_id = None
    if notifier_store is not None:
        st = notifier_store.get(record.job_id)
        message_id = st.feishu_message_id if st else None
    if not message_id:
        message_id = record.feishu_message_id

    if message_id:
        updated_ok = notifier.update_schema_card(
            message_id=message_id,
            card=card,
        )
        if updated_ok:
            return record
        logger.warning(
            "Feishu card update failed, fallback to resend: job_id=%s message_id=%s",
            record.job_id,
            message_id,
        )
    new_message_id = notifier.send_schema_card_with_message_id(card=card)
    if not new_message_id:
        logger.warning(
            "Feishu card send failed: job_id=%s has_existing_message=%s",
            record.job_id,
            bool(message_id),
        )
        return record
    if notifier_store is not None:
        notifier_store.upsert(
            job_id=record.job_id,
            now_ts=now_ts,
            feishu_message_id=new_message_id,
        )
        return record
    return state_store.mark_feishu_message(record.job_id, new_message_id, now_ts)


def get_status(
    job_id: str,
    run_id: str,
    state_store: JobStateStore,
    ssh_client: SSHClient | None,
    notifier: FeishuNotifier,
    *,
    tracker: MLflowTracker | None = None,
    notifier_store: NotifierStateStore | None = None,
    mlflow_url: str | None = None,
    feishu_report_enable: bool = True,
    feishu_report_every_n_epochs: int = 5,
    primary_metric_key: str = "map5095",
    feishu_card_img_key: str | None = None,
    feishu_card_fallback_img_key: str | None = None,
    control_client: HttpControlClient | None = None,
) -> dict[str, object]:
    now = int(time.time())
    record = state_store.get(job_id)
    if not record:
        return err(
            error_code="JOB_NOT_FOUND",
            message=f"job not found: {job_id}",
            retryable=False,
            hint="请先调用 yolo_start_training 创建任务",
            payload={"jobId": job_id},
        )

    effective_run_id = record.run_id or run_id

    if control_client is not None:
        try:
            remote = control_client.get_training_status(
                {
                    "jobId": job_id,
                    "pid": record.pid,
                    "metricsPath": record.paths.get("metricsPath", ""),
                    "logPath": record.paths.get("logPath", ""),
                    "totalEpochs": record.train_epochs or 0,
                    "createdAt": record.created_at,
                }
            )
        except RemoteControlError as exc:
            return err(
                error_code=exc.error_code,
                message=str(exc),
                retryable=exc.retryable,
                hint=exc.hint,
                payload={"jobId": job_id, "runId": effective_run_id, **exc.payload},
            )
        status_text = str(remote.get("status", record.status.value))
        target_status = None
        if status_text == JobStatus.COMPLETED.value:
            target_status = JobStatus.COMPLETED
        elif status_text == JobStatus.FAILED.value:
            target_status = JobStatus.FAILED
        elif status_text == JobStatus.STOPPED.value:
            target_status = JobStatus.STOPPED
        elif status_text == JobStatus.RUNNING.value:
            target_status = JobStatus.RUNNING
        if target_status is not None and record.status != target_status:
            state_store.update_status(job_id, target_status, now)
        return ok(
            {
                "jobId": job_id,
                "runId": effective_run_id,
                "status": status_text,
                "progress": float(remote.get("progress", 0.0) or 0.0),
                "metrics": dict(remote.get("metrics", {})),
                "artifacts": {
                    "best": record.paths.get("bestPath", ""),
                    "last": record.paths.get("lastPath", ""),
                },
            }
        )

    if ssh_client is None:
        return err(
            error_code="REMOTE_CLIENT_MISSING",
            message="ssh client is required when control client is disabled",
            retryable=False,
            hint="请检查 YOLO_REMOTE_MODE 配置",
            payload={"jobId": job_id, "runId": effective_run_id},
        )

    # 优先从 MLflow 获取最新指标与进度；读取失败再回退到 results.csv。
    if tracker is not None:
        ml_run = tracker.find_latest_run_for_job_id(job_id)
        if ml_run is not None:
            metrics = ml_run.data.metrics or {}

            def _pick_float(*keys: str) -> float | None:
                for key in keys:
                    if key in metrics:
                        try:
                            return float(metrics[key])
                        except Exception:
                            return None
                return None

            epoch_val = _pick_float("epoch", "train/epoch", "metrics/epoch")
            epoch_i = int(epoch_val) if epoch_val is not None else 0
            loss_val = _pick_float("loss", "train/loss", "metrics/loss")
            map50_val = _pick_float("map50", "metrics/map50", "val/map50")
            map5095_val = _pick_float(
                "map5095",
                "metrics/map5095",
                "val/map5095",
                "val/map50-95",
            )
            precision_val = _pick_float("precision", "metrics/precision", "val/precision")
            recall_val = _pick_float("recall", "metrics/recall", "val/recall")

            total_epochs = max(record.train_epochs or 100, 1)
            progress = round(min(1.0, epoch_i / total_epochs), 4) if epoch_i > 0 else 0.0
            elapsed_seconds = max(now - record.created_at, 0)
            eta_seconds = _estimate_eta_seconds(
                epoch=epoch_i,
                total_epochs=total_epochs,
                elapsed_seconds=elapsed_seconds,
            )

            # MLflow 本身不提供进程存活；仍用 SSH pid 判断是否还在跑。
            process_alive = ssh_client.process_alive(record.pid) if record.pid else False
            status_value = JobStatus.RUNNING.value if process_alive else JobStatus.COMPLETED.value

            primary_value = None
            if epoch_i > 0:
                primary_value = metric_value_from_parsed(
                    {
                        "epoch": epoch_i,
                        "loss": loss_val or 0.0,
                        "map50": map50_val or 0.0,
                        "map5095": map5095_val or 0.0,
                        "precision": precision_val or 0.0,
                        "recall": recall_val or 0.0,
                    },
                    primary_metric_key,
                )

            # 里程碑推送（不强依赖 results.csv 的曲线图）。
            if process_alive and feishu_report_enable:
                milestone_n = feishu_report_every_n_epochs
                last_reported_epoch = record.last_reported_epoch
                if notifier_store is not None:
                    st = notifier_store.get(job_id)
                    if st is not None:
                        last_reported_epoch = st.last_reported_epoch
                should_report_milestone = (
                    milestone_n > 0
                    and epoch_i > 0
                    and epoch_i >= last_reported_epoch + milestone_n
                )
                if should_report_milestone:
                    card = _build_training_schema_card(
                        title="YOLO模型训练里程碑",
                        header_template="blue",
                        epoch_text=f"{epoch_i}/{total_epochs}",
                        elapsed_text=_format_duration(elapsed_seconds),
                        eta_text=(
                            _format_duration(eta_seconds)
                            if eta_seconds is not None
                            else "n/a"
                        ),
                        updated_time_text=_format_card_timestamp(now),
                        mlflow_url=mlflow_url,
                        top_img_key=feishu_card_img_key,
                        top_img_fallback_key=feishu_card_fallback_img_key,
                        training_rows=None,
                    )
                    record = _upsert_training_card(
                        record=record,
                        now_ts=now,
                        state_store=state_store,
                        notifier_store=notifier_store,
                        notifier=notifier,
                        card=card,
                    )
                    if notifier_store is not None:
                        notifier_store.upsert(
                            job_id=job_id,
                            now_ts=now,
                            last_reported_epoch=epoch_i,
                        )
                    else:
                        _ = state_store.mark_milestone_epoch(job_id, epoch_i, now)

            # 完成态：如果进程结束，就落库并发完成卡片。
            if not process_alive and record.status != JobStatus.COMPLETED:
                updated = state_store.update_status(job_id, JobStatus.COMPLETED, now)
                card = _build_training_schema_card(
                    title="YOLO模型训练完成",
                    header_template="green",
                    epoch_text=f"{epoch_i}/{total_epochs}" if epoch_i > 0 else "--/--",
                    elapsed_text=_format_duration(elapsed_seconds),
                    eta_text="0s",
                    updated_time_text=_format_card_timestamp(now),
                    mlflow_url=mlflow_url,
                    top_img_key=feishu_card_img_key,
                    top_img_fallback_key=feishu_card_fallback_img_key,
                    training_rows=None,
                )
                updated = _upsert_training_card(
                    record=updated,
                    now_ts=now,
                    state_store=state_store,
                    notifier_store=notifier_store,
                    notifier=notifier,
                    card=card,
                )
                last_notified_state = updated.last_notified_state
                if notifier_store is not None:
                    st = notifier_store.get(job_id)
                    if st is not None:
                        last_notified_state = st.last_notified_state
                if last_notified_state != JobStatus.COMPLETED:
                    if notifier_store is not None:
                        notifier_store.upsert(
                            job_id=job_id,
                            now_ts=now,
                            last_notified_state=JobStatus.COMPLETED,
                        )
                    else:
                        state_store.mark_notified(job_id, JobStatus.COMPLETED, now)
                status_value = updated.status.value

            return ok(
                {
                    "jobId": job_id,
                    "runId": ml_run.info.run_id or effective_run_id,
                    "status": status_value,
                    "progress": progress,
                    "metrics": {
                        "epoch": epoch_i if epoch_i > 0 else None,
                        "loss": loss_val,
                        "map50": map50_val,
                        "map5095": map5095_val,
                        "precision": precision_val,
                        "recall": recall_val,
                        "primaryMetric": primary_value,
                        "mlflowMetrics": dict(metrics),
                    },
                    "artifacts": {
                        "best": record.paths.get("bestPath", ""),
                        "last": record.paths.get("lastPath", ""),
                    },
                }
            )
    metrics_path = (record.paths.get("metricsPath") or "").strip()
    if not metrics_path:
        return err(
            error_code="INVALID_JOB_RECORD",
            message="job record missing metricsPath",
            retryable=False,
            hint="状态文件中的任务路径不完整，请重新 start_training 或修复 YOLO_STATE_FILE",
            payload={"jobId": job_id},
        )
    try:
        content, stderr_text, exit_code = ssh_client.execute(
            f"cat {shlex.quote(metrics_path)}"
        )
    except SSHRemoteExecutionError as exc:
        return err(
            error_code=exc.error_code,
            message=str(exc),
            retryable=exc.retryable,
            hint=exc.hint,
            payload={"jobId": job_id, "runId": effective_run_id},
        )
    if exit_code != 0:
        if ssh_client.process_alive(record.pid):
            return ok(
                {
                    "jobId": job_id,
                    "runId": effective_run_id,
                    "status": record.status.value,
                    "progress": 0.0,
                    "error": stderr_text.strip() or "results.csv not ready",
                }
            )
        log_path = (record.paths.get("logPath") or "").strip()
        if log_path:
            try:
                log_content, _, _ = ssh_client.tail_file(log_path, lines=80)
            except SSHRemoteExecutionError:
                log_content = ""
        else:
            log_content = ""
        failed = "error" in log_content.lower() or "traceback" in log_content.lower()
        target = JobStatus.FAILED if failed else JobStatus.COMPLETED
        updated = state_store.update_status(job_id, target, now)
        card = _build_training_schema_card(
            title="YOLO模型训练失败" if target == JobStatus.FAILED else "YOLO模型训练完成",
            header_template="red" if target == JobStatus.FAILED else "green",
            epoch_text="--/--",
            elapsed_text="n/a",
            eta_text="n/a",
            updated_time_text=_format_card_timestamp(now),
            mlflow_url=mlflow_url,
            top_img_key=feishu_card_img_key,
            top_img_fallback_key=feishu_card_fallback_img_key,
            training_rows=None,
        )
        updated = _upsert_training_card(
            record=updated,
            now_ts=now,
            state_store=state_store,
            notifier_store=notifier_store,
            notifier=notifier,
            card=card,
        )
        last_notified_state = updated.last_notified_state
        if notifier_store is not None:
            st = notifier_store.get(job_id)
            if st is not None:
                last_notified_state = st.last_notified_state
        if last_notified_state != target:
            if notifier_store is not None:
                notifier_store.upsert(job_id=job_id, now_ts=now, last_notified_state=target)
            else:
                state_store.mark_notified(job_id, target, now)
        if target == JobStatus.FAILED:
            return err(
                error_code="TRAIN_FAILED",
                message="training process exited with errors",
                retryable=False,
                hint="查看远程 train.log 定位错误并修复数据或参数",
                payload=updated.to_dict(),
            )
        return ok(updated.to_dict())

    rows = list(csv.DictReader(StringIO(content)))
    if not rows:
        return ok(
            {
                "jobId": job_id,
                "runId": effective_run_id,
                "status": record.status.value,
                "progress": 0.0,
            }
        )

    last_row = rows[-1]
    parsed = parse_training_row(last_row)
    epoch = int(parsed["epoch"])
    map50 = float(parsed["map50"])
    map5095 = float(parsed["map5095"])
    precision = float(parsed["precision"])
    recall = float(parsed["recall"])
    loss = float(parsed["loss"])
    primary_value = metric_value_from_parsed(parsed, primary_metric_key)

    record = state_store.mark_metrics(job_id, now)
    total_epochs = max(record.train_epochs or 100, 1)
    progress = round(min(1.0, epoch / total_epochs), 4)
    elapsed_seconds = max(now - record.created_at, 0)
    eta_seconds = _estimate_eta_seconds(
        epoch=epoch,
        total_epochs=total_epochs,
        elapsed_seconds=elapsed_seconds,
    )
    try:
        process_alive = ssh_client.process_alive(record.pid)
    except SSHRemoteExecutionError as exc:
        return err(
            error_code=exc.error_code,
            message=str(exc),
            retryable=exc.retryable,
            hint=exc.hint,
            payload={"jobId": job_id, "runId": effective_run_id},
        )

    if process_alive and feishu_report_enable:
        milestone_n = feishu_report_every_n_epochs
        last_reported_epoch = record.last_reported_epoch
        if notifier_store is not None:
            st = notifier_store.get(job_id)
            if st is not None:
                last_reported_epoch = st.last_reported_epoch
        should_report_milestone = (
            milestone_n > 0
            and epoch > 0
            and epoch >= last_reported_epoch + milestone_n
        )
        if should_report_milestone:
            card = _build_training_schema_card(
                title="YOLO模型训练里程碑",
                header_template="blue",
                epoch_text=f"{epoch}/{total_epochs}",
                elapsed_text=_format_duration(elapsed_seconds),
                eta_text=(
                    _format_duration(eta_seconds)
                    if eta_seconds is not None
                    else "n/a"
                ),
                updated_time_text=_format_card_timestamp(now),
                mlflow_url=mlflow_url,
                top_img_key=feishu_card_img_key,
                top_img_fallback_key=feishu_card_fallback_img_key,
                training_rows=rows,
            )
            record = _upsert_training_card(
                record=record,
                now_ts=now,
                state_store=state_store,
                notifier_store=notifier_store,
                notifier=notifier,
                card=card,
            )
            if notifier_store is not None:
                notifier_store.upsert(job_id=job_id, now_ts=now, last_reported_epoch=epoch)
            else:
                record = state_store.mark_milestone_epoch(job_id, epoch, now)

    if not process_alive:
        updated = state_store.update_status(job_id, JobStatus.COMPLETED, now)
        card = _build_training_schema_card(
            title="YOLO模型训练完成",
            header_template="green",
            epoch_text=f"{epoch}/{total_epochs}",
            elapsed_text=_format_duration(elapsed_seconds),
            eta_text="0s",
            updated_time_text=_format_card_timestamp(now),
            mlflow_url=mlflow_url,
            top_img_key=feishu_card_img_key,
            top_img_fallback_key=feishu_card_fallback_img_key,
            training_rows=rows,
        )
        updated = _upsert_training_card(
            record=updated,
            now_ts=now,
            state_store=state_store,
            notifier_store=notifier_store,
            notifier=notifier,
            card=card,
        )
        last_notified_state = updated.last_notified_state
        if notifier_store is not None:
            st = notifier_store.get(job_id)
            if st is not None:
                last_notified_state = st.last_notified_state
        if last_notified_state != JobStatus.COMPLETED:
            if notifier_store is not None:
                notifier_store.upsert(
                    job_id=job_id,
                    now_ts=now,
                    last_notified_state=JobStatus.COMPLETED,
                )
            else:
                state_store.mark_notified(job_id, JobStatus.COMPLETED, now)
        status_value = updated.status.value
    else:
        status_value = JobStatus.RUNNING.value

    return ok(
        {
            "jobId": job_id,
            "runId": effective_run_id,
            "status": status_value,
            "progress": progress,
            "metrics": {
                "epoch": epoch,
                "loss": loss,
                "map50": map50,
                "map5095": map5095,
                "precision": precision,
                "recall": recall,
                "primaryMetric": primary_value,
            },
            "artifacts": {
                "best": record.paths.get("bestPath", ""),
                "last": record.paths.get("lastPath", ""),
            },
        }
    )
