from __future__ import annotations

import base64
import csv
import time
from io import BytesIO, StringIO
from typing import Any

from yolo_auto.errors import err, ok
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobStatus
from yolo_auto.ssh_client import SSHClient
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.metrics_csv import metric_value_from_parsed, parse_training_row
from yolo_auto.tracker import MLflowTracker


def _build_mlflow_actions(mlflow_url: str | None) -> list[dict[str, Any]] | None:
    if not mlflow_url:
        return None
    return [
        {
            "tag": "button",
            "text": {"tag": "plain_text", "content": "查看 MLflow"},
            "type": "url",
            "url": mlflow_url,
        }
    ]


def _to_percent(progress: float) -> float:
    return max(0.0, min(100.0, progress * 100.0))


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
    progress: float,
    elapsed_text: str,
    eta_text: str,
    map50: float,
    map5095: float,
    recall: float,
    loss: float,
    updated_time_text: str,
    top_img_key: str | None = None,
    top_img_fallback_key: str | None = None,
) -> dict[str, Any]:
    progress_pct = _to_percent(progress)
    elements: list[dict[str, Any]] = []
    if top_img_key:
        image_element: dict[str, Any] = {
            "tag": "img",
            "img_key": top_img_key,
            "alt": {"tag": "plain_text", "content": "training cover"},
            "scale_type": "fit_horizontal",
        }
        if top_img_fallback_key:
            image_element["fallback_img_key"] = top_img_fallback_key
        elements.append(image_element)
    elements.extend(
        [
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
                        value=f"{progress_pct:.1f}%",
                        label="训练进度",
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
            },
            {
                "tag": "column_set",
                "flex_mode": "stretch",
                "horizontal_spacing": "12px",
                "margin": "0px",
                "columns": [
                    _metric_column(
                        value=f"{map50:.3f}",
                        label="mAP50",
                        value_color="violet",
                        background_style="violet-50",
                    ),
                    _metric_column(
                        value=f"{map5095:.3f}",
                        label="mAP50-95",
                        value_color="violet",
                        background_style="violet-50",
                    ),
                    _metric_column(
                        value=f"{recall:.3f}",
                        label="Recall",
                        value_color="purple",
                        background_style="purple-50",
                    ),
                    _metric_column(
                        value=f"{loss:.3f}",
                        label="Loss",
                        value_color="purple",
                        background_style="purple-50",
                    ),
                ],
            },
            {
                "tag": "markdown",
                "content": f"<font color='grey'>更新时间：{updated_time_text}</font>",
                "text_align": "right",
            },
        ]
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
    notifier: FeishuNotifier,
    card: dict[str, Any],
):
    if record.feishu_message_id:
        updated_ok = notifier.update_schema_card(
            message_id=record.feishu_message_id,
            card=card,
        )
        if updated_ok:
            return record
    new_message_id = notifier.send_schema_card_with_message_id(card=card)
    if not new_message_id:
        return record
    return state_store.mark_feishu_message(record.job_id, new_message_id, now_ts)


def get_status(
    job_id: str,
    run_id: str,
    state_store: JobStateStore,
    ssh_client: SSHClient,
    tracker: MLflowTracker,
    notifier: FeishuNotifier,
    *,
    feishu_report_enable: bool = True,
    feishu_report_every_n_epochs: int = 5,
    primary_metric_key: str = "map5095",
    feishu_card_img_key: str | None = None,
    feishu_card_fallback_img_key: str | None = None,
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
    metrics_path = record.paths.get("metricsPath", "")
    content, stderr_text, exit_code = ssh_client.execute(f"cat {metrics_path}")
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
        log_path = record.paths.get("logPath", "")
        log_content, _, _ = ssh_client.tail_file(log_path, lines=80)
        failed = "error" in log_content.lower() or "traceback" in log_content.lower()
        target = JobStatus.FAILED if failed else JobStatus.COMPLETED
        updated = state_store.update_status(job_id, target, now)
        card = _build_training_schema_card(
            title="YOLO模型训练失败" if target == JobStatus.FAILED else "YOLO模型训练完成",
            header_template="red" if target == JobStatus.FAILED else "green",
            epoch_text="--/--",
            progress=1.0,
            elapsed_text="n/a",
            eta_text="n/a",
            map50=0.0,
            map5095=0.0,
            recall=0.0,
            loss=0.0,
            updated_time_text=_format_card_timestamp(now),
            top_img_key=feishu_card_img_key,
            top_img_fallback_key=feishu_card_fallback_img_key,
        )
        updated = _upsert_training_card(
            record=updated,
            now_ts=now,
            state_store=state_store,
            notifier=notifier,
            card=card,
        )
        if updated.last_notified_state != target:
            state_store.mark_notified(job_id, target, now)
        if target == JobStatus.FAILED:
            return err(
                error_code="TRAIN_FAILED",
                message="training process exited with errors",
                retryable=False,
                hint="查看远程 train.log 定位错误并修复数据或参数",
                payload=updated.to_dict(),
            )
        tracker.finish_run(effective_run_id, updated.paths.get("bestPath"))
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

    tracker.log_epoch(
        run_id=effective_run_id,
        metrics={
            "loss": loss,
            "map50": map50,
            "map5095": map5095,
            "precision": precision,
            "recall": recall,
        },
        step=epoch,
    )
    record = state_store.mark_metrics(job_id, now)
    total_epochs = max(record.train_epochs or 100, 1)
    progress = round(min(1.0, epoch / total_epochs), 4)
    elapsed_seconds = max(now - record.created_at, 0)
    eta_seconds = _estimate_eta_seconds(
        epoch=epoch,
        total_epochs=total_epochs,
        elapsed_seconds=elapsed_seconds,
    )
    process_alive = ssh_client.process_alive(record.pid)

    if process_alive and feishu_report_enable:
        if epoch > 0:
            card = _build_training_schema_card(
                title="YOLO模型训练里程碑",
                header_template="blue",
                epoch_text=f"{epoch}/{total_epochs}",
                progress=progress,
                elapsed_text=_format_duration(elapsed_seconds),
                eta_text=_format_duration(eta_seconds) if eta_seconds is not None else "n/a",
                map50=map50,
                map5095=map5095,
                recall=recall,
                loss=loss,
                updated_time_text=_format_card_timestamp(now),
                top_img_key=feishu_card_img_key,
                top_img_fallback_key=feishu_card_fallback_img_key,
            )
            record = _upsert_training_card(
                record=record,
                now_ts=now,
                state_store=state_store,
                notifier=notifier,
                card=card,
            )
            if epoch > record.last_reported_epoch:
                record = state_store.mark_milestone_epoch(job_id, epoch, now)

    if not process_alive:
        updated = state_store.update_status(job_id, JobStatus.COMPLETED, now)
        tracker.finish_run(effective_run_id, updated.paths.get("bestPath"))
        card = _build_training_schema_card(
            title="YOLO模型训练完成",
            header_template="green",
            epoch_text=f"{epoch}/{total_epochs}",
            progress=1.0,
            elapsed_text=_format_duration(elapsed_seconds),
            eta_text="0s",
            map50=map50,
            map5095=map5095,
            recall=recall,
            loss=loss,
            updated_time_text=_format_card_timestamp(now),
            top_img_key=feishu_card_img_key,
            top_img_fallback_key=feishu_card_fallback_img_key,
        )
        updated = _upsert_training_card(
            record=updated,
            now_ts=now,
            state_store=state_store,
            notifier=notifier,
            card=card,
        )
        if updated.last_notified_state != JobStatus.COMPLETED:
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
