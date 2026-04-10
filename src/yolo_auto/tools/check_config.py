from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from yolo_auto.config import Settings
from yolo_auto.errors import ok
from yolo_auto.remote_control import HttpControlClient
from yolo_auto.tracker import MLflowTracker

CheckStatus = Literal["ok", "warn", "fail", "skip"]


@dataclass(frozen=True)
class _CheckItem:
    name: str
    var: str
    status: CheckStatus
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "var": self.var, "status": self.status, "detail": self.detail}


def _check_control_plane(client: HttpControlClient) -> _CheckItem:
    reachable, detail = client.health_check()
    return _CheckItem(
        name="控制面连接",
        var="YOLO_CONTROL_BASE_URL",
        status="ok" if reachable else "fail",
        detail=detail,
    )


def _check_feishu(settings: Settings) -> _CheckItem:
    has_app_bot = bool(
        settings.feishu_app_id and settings.feishu_app_secret and settings.feishu_chat_id
    )
    if has_app_bot:
        return _CheckItem(
            name="飞书通知",
            var="FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_CHAT_ID",
            status="ok",
            detail="应用机器人已配置（支持里程碑卡片更新）",
        )
    if settings.feishu_webhook_url:
        return _CheckItem(
            name="飞书通知",
            var="FEISHU_WEBHOOK_URL",
            status="ok",
            detail="Webhook 已配置（不支持卡片更新，每条通知独立发送）",
        )
    return _CheckItem(
        name="飞书通知",
        var="FEISHU_WEBHOOK_URL",
        status="fail",
        detail="未配置：训练过程中不会有任何飞书推送",
    )


def _check_mlflow(tracker: MLflowTracker | None, tracking_uri: str) -> _CheckItem:
    if not tracking_uri:
        return _CheckItem(
            name="MLflow 实验追踪",
            var="MLFLOW_TRACKING_URI",
            status="skip",
            detail="未配置，yolo_auto_tune 对比和排行榜不可用",
        )
    # sqlite URI: check if db file exists (may be on remote side, warn not fail)
    if tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri[len("sqlite:///"):]
        if not os.path.exists(db_path):
            return _CheckItem(
                name="MLflow 实验追踪",
                var="MLFLOW_TRACKING_URI",
                status="warn",
                detail=f"数据库文件暂不存在：{db_path}（首次训练后由 Ultralytics 自动创建）",
            )
    if tracker is not None:
        reachable, detail = tracker.ping()
        return _CheckItem(
            name="MLflow 实验追踪",
            var="MLFLOW_TRACKING_URI",
            status="ok" if reachable else "warn",
            detail=detail,
        )
    return _CheckItem(
        name="MLflow 实验追踪",
        var="MLFLOW_TRACKING_URI",
        status="ok",
        detail=f"已配置：{tracking_uri}",
    )


def _check_minio() -> _CheckItem:
    alias = os.getenv("YOLO_MINIO_ALIAS", "").strip()
    bucket = os.getenv("YOLO_MINIO_EXPORT_BUCKET", "").strip()
    if alias and bucket:
        return _CheckItem(
            name="MinIO 数据集同步",
            var="YOLO_MINIO_ALIAS / YOLO_MINIO_EXPORT_BUCKET",
            status="ok",
            detail=f"已配置：alias={alias}, bucket={bucket}",
        )
    return _CheckItem(
        name="MinIO 数据集同步",
        var="YOLO_MINIO_ALIAS / YOLO_MINIO_EXPORT_BUCKET",
        status="skip",
        detail="未配置，yolo_sync_dataset 不可用",
    )


def _build_next_steps(
    required: list[_CheckItem],
    optional: list[_CheckItem],
) -> list[str]:
    steps: list[str] = []
    for item in required:
        if item.status == "fail":
            steps.append(f"[必须] 修复 {item.var}：{item.detail}")
    for item in optional:
        if item.status == "warn":
            steps.append(f"[建议] 检查 {item.var}：{item.detail}")
        elif item.status == "skip":
            steps.append(f"[可选] 配置 {item.var} 以启用「{item.name}」")
    return steps


def check_config(
    settings: Settings,
    control_client: HttpControlClient,
    tracker: MLflowTracker | None = None,
) -> dict[str, Any]:
    """检查当前配置是否满足各项功能的运行条件，返回逐项诊断与 nextSteps 建议。"""
    required = [
        _check_control_plane(control_client),
        _check_feishu(settings),
    ]
    optional = [
        _check_mlflow(tracker, settings.mlflow_tracking_uri),
        _check_minio(),
    ]

    all_required_ok = all(item.status == "ok" for item in required)

    capabilities = {
        "canTrain": required[0].status == "ok",
        "canNotify": required[1].status == "ok",
        "canCompareExperiments": optional[0].status == "ok",
        "canSyncDataset": optional[1].status == "ok",
    }

    return ok({
        "allRequiredOk": all_required_ok,
        "required": [i.to_dict() for i in required],
        "optional": [i.to_dict() for i in optional],
        "capabilities": capabilities,
        "paths": {
            "workDir": settings.yolo_work_dir,
            "datasetsDir": settings.yolo_datasets_dir,
            "jobsDir": settings.yolo_jobs_dir,
            "modelsDir": settings.yolo_models_dir,
            "stateFile": settings.yolo_state_file,
        },
        "nextSteps": _build_next_steps(required, optional),
    })
