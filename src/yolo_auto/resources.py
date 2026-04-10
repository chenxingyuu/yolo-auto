from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from yolo_auto.config import Settings
from yolo_auto.models import JobStatus
from yolo_auto.remote_control import HttpControlClient
from yolo_auto.state_store import JobStateStore
from yolo_auto.tracker import MLflowTracker


def register_resources(
    mcp: FastMCP,
    settings: Settings,
    control_client: HttpControlClient,
    state_store: JobStateStore,
    tracker: MLflowTracker,
) -> None:
    """Register MCP Resources for read-only background context."""

    @mcp.resource(
        "yolo://config",
        name="current-config",
        description="当前生效的环境配置概要（已脱敏，不含密钥与 webhook 完整 URL）。",
        mime_type="application/json",
    )
    def resource_config() -> str:
        feishu_mode = (
            "app_bot"
            if (settings.feishu_app_id and settings.feishu_app_secret and settings.feishu_chat_id)
            else "webhook"
        )
        return json.dumps(
            {
                "controlApi": {
                    "baseUrl": settings.yolo_control_base_url,
                    "timeoutSeconds": settings.yolo_control_timeout_seconds,
                },
                "workDir": settings.yolo_work_dir,
                "datasetsDir": settings.yolo_datasets_dir,
                "jobsDir": settings.yolo_jobs_dir,
                "modelsDir": settings.yolo_models_dir,
                "stateFile": settings.yolo_state_file,
                "feishu": {
                    "reportEnable": settings.feishu_report_enable,
                    "reportEveryNEpochs": settings.feishu_report_every_n_epochs,
                    "messageMode": "card",
                    "mode": feishu_mode,
                },
                "primaryMetric": settings.primary_metric_key,
                "watchPollIntervalSeconds": settings.watch_poll_interval_seconds,
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "yolo://jobs/active",
        name="active-jobs",
        description="当前运行中或排队中的训练任务列表。",
        mime_type="application/json",
    )
    def resource_active_jobs() -> str:
        records = state_store.list_all()
        active = [
            r.to_dict()
            for r in records
            if r.status in (JobStatus.RUNNING, JobStatus.QUEUED)
        ]
        return json.dumps(
            {"activeJobs": active, "count": len(active)},
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "yolo://jobs/history",
        name="job-history",
        description="最近 50 条训练任务记录（含已完成/失败/停止），只读本地状态。",
        mime_type="application/json",
    )
    def resource_job_history() -> str:
        records = state_store.list_all()[:50]
        items = [r.to_dict() for r in records]
        return json.dumps(
            {"jobs": items, "count": len(items)},
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "yolo://mlflow/experiments",
        name="mlflow-experiments",
        description="MLflow experiments 列表（只读）。",
        mime_type="application/json",
    )
    def resource_mlflow_experiments() -> str:
        experiments = tracker.list_experiments(max_results=200)
        return json.dumps(
            {
                "experiments": experiments,
                "count": len(experiments),
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "yolo://mlflow/leaderboard",
        name="mlflow-leaderboard",
        description="MLflow 实验中按主指标排序的 Top 10 runs（只读）。",
        mime_type="application/json",
    )
    def resource_mlflow_leaderboard() -> str:
        top_runs = tracker.summarize_top_runs(
            settings.primary_metric_key,
            limit=10,
        )
        return json.dumps(
            {
                "metricKey": settings.primary_metric_key,
                "topRuns": top_runs,
                "count": len(top_runs),
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "yolo://models/registry",
        name="registered-models",
        description="MLflow Model Registry 摘要（只读）。",
        mime_type="application/json",
    )
    def resource_registered_models() -> str:
        models = tracker.list_registered_models(max_results=50)
        enriched: list[dict[str, Any]] = []
        for m in models[:25]:
            name = str(m.get("name", ""))
            if not name:
                continue
            try:
                versions = tracker.list_model_versions(model_name=name, max_results=5)
            except Exception:
                versions = []
            latest = versions[0] if versions else None
            aliases = m.get("aliases") or {}
            enriched.append(
                {
                    **m,
                    "primaryMetricKey": settings.primary_metric_key,
                    "latestVersionDetail": latest,
                    "aliasApproved": aliases.get("approved"),
                    "aliasCandidate": aliases.get("candidate"),
                }
            )
        return json.dumps(
            {
                "enabled": True,
                "primaryMetricKey": settings.primary_metric_key,
                "models": enriched,
                "count": len(enriched),
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "yolo://datasets",
        name="remote-datasets",
        description="训练容器 datasets 目录下的 YAML 数据集配置文件列表。",
        mime_type="application/json",
    )
    def resource_datasets() -> str:
        try:
            payload = control_client.list_datasets(settings.yolo_datasets_dir)
            files = [str(item) for item in (payload.get("files") or [])]
        except Exception:
            files = []
        return json.dumps(
            {
                "datasetsDir": settings.yolo_datasets_dir,
                "files": files,
                "count": len(files),
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "yolo://minio/datasets",
        name="minio-datasets",
        description="MinIO 导出目录中的可用数据集 zip 列表（用于 yolo_sync_dataset）。",
        mime_type="application/json",
    )
    def resource_minio_datasets() -> str:
        alias = os.getenv("YOLO_MINIO_ALIAS", "minio").strip() or "minio"
        bucket = os.getenv("YOLO_MINIO_EXPORT_BUCKET", "cvat-export").strip() or "cvat-export"
        prefix = os.getenv("YOLO_MINIO_EXPORT_PREFIX", "exports").strip() or "exports"
        remote_dir = "/".join(p.strip("/") for p in (alias, bucket, prefix) if p.strip("/"))
        try:
            payload = control_client.list_minio_datasets(remote_dir)
            raw_items = payload.get("files") or []
            items = [dict(item) for item in raw_items if isinstance(item, dict)]
        except Exception:
            items = []
        return json.dumps(
            {
                "source": f"{remote_dir}/",
                "files": items,
                "count": len(items),
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "yolo://models",
        name="remote-models",
        description="训练容器 models 目录下的预训练权重文件（.pt）列表及大小。",
        mime_type="application/json",
    )
    def resource_models() -> str:
        try:
            payload = control_client.list_models(settings.yolo_models_dir)
            raw_models = payload.get("models") or []
            models = [dict(item) for item in raw_models if isinstance(item, dict)]
        except Exception:
            models = []
        return json.dumps(
            {
                "modelsDir": settings.yolo_models_dir,
                "models": models,
                "count": len(models),
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "yolo://env/gpu",
        name="gpu-info",
        description="训练容器 GPU 概况（HTTP-only 模式下需控制面扩展支持）。",
        mime_type="application/json",
    )
    def resource_gpu_info() -> str:
        try:
            payload = control_client.get_gpu_info()
            raw_gpus = payload.get("gpus") or []
            gpus = [dict(item) for item in raw_gpus if isinstance(item, dict)]
        except Exception:
            gpus = []
        return json.dumps(
            {"gpus": gpus, "count": len(gpus)},
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "yolo://env/system",
        name="system-info",
        description="训练容器 CPU、内存、磁盘概况（HTTP-only 模式下需控制面扩展支持）。",
        mime_type="application/json",
    )
    def resource_system_info() -> str:
        try:
            info = control_client.get_system_info()
        except Exception:
            info = {}
        return json.dumps(info, ensure_ascii=False, indent=2)

    @mcp.resource(
        "yolo://jobs/{jobId}/log",
        name="job-log",
        description="读取训练日志（HTTP-only 模式下需控制面扩展日志端点支持）。",
        mime_type="text/plain",
    )
    def resource_job_log(jobId: str) -> str:
        return f"log resource not implemented in HTTP-only mode: {jobId}"

    @mcp.resource(
        "yolo://guide/training-params",
        name="training-params-guide",
        description="Ultralytics YOLO 训练参数速查与推荐值范围，帮助模型给出合理的参数建议。",
        mime_type="text/markdown",
    )
    def resource_training_params_guide() -> str:
        return """\
# YOLO 训练参数速查

## 核心参数
| 参数 | CLI 名 | 类型 | 推荐范围 | 说明 |
|------|--------|------|----------|------|
| model | model | str | yolov8n/s/m/l/x.pt | n=最快, x=最精, 按需选 |
| epochs | epochs | int | 50–300 | 小数据集 100+, 大数据集 50–150 |
| imgSize | imgsz | int | 320–1280 | 640 为平衡点, 精度优先用 1280 |
| batch | batch | int/float | 8–64 / 0.6–0.9 | 显存允许尽量大; 小数为自动显存比 |
| learningRate | lr0 | float 可选 | 0.001–0.02 | 不传不写 lr0，用框架默认；要手动调参时再传 |

## 常用可选参数
| 参数 | CLI 名 | 默认 | 建议 |
|------|--------|------|------|
| optimizer | optimizer | auto | SGD(稳) / AdamW(快收敛) |
| patience | patience | 100 | 20–50 可加速早停 |
| cosLr | cos_lr | False | True 后期精度常更好 |
| amp | amp | True | 保持 True 加速 |
| cache | cache | False | ram(快) / disk(省内存) |
| workers | workers | 8 | 与 CPU 核数匹配 |
| freeze | freeze | None | 迁移学习冻结 backbone: 10 |
| closeMosaic | close_mosaic | 10 | 最后 10–20 epoch 关 mosaic |

## 模型选择指南
容器内预下载权重位于 `/workspace/models/`，涵盖 v5/v8/11/26 全系列 detect，
可直接用绝对路径避免重复下载。

| 场景 | model 参数 | imgSize | batch |
|------|-----------|---------|-------|
| 快速验证 | /workspace/models/yolo26n.pt | 640 | 32 |
| 生产部署(速度优先) | /workspace/models/yolo26s.pt | 640 | 16 |
| 生产部署(精度优先) | /workspace/models/yolo26m.pt 或 l | 1280 | 8 |
| 竞赛/极致精度 | /workspace/models/yolo26x.pt | 1280 | 4–8 |
| 旧版兼容 | /workspace/models/yolov8*.pt 或 yolov5*.pt | 640 | 16 |

可用系列：yolov5n~x / yolov8n~x / yolo11n~x / yolo26n~x（共 20 个 .pt）

## 调参经验
- **欠拟合**: 增大模型(n→s→m)、增加 epochs、提高 lr
- **过拟合**: 增加数据增强、降低 lr、使用早停(patience=30)、增大数据集
- **OOM**: 降低 batch、降低 imgSize、使用 amp=True
- **训练慢**: 使用 cache=ram、增大 batch、用更小模型先验证
"""

    # CVAT 相关资源已迁移到独立服务 cvat-mcp（使用 cvat:// 前缀）。


