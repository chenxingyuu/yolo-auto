from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from yolo_auto.config import Settings
from yolo_auto.cvat_client import CVATClient
from yolo_auto.models import JobStatus
from yolo_auto.ssh_client import SSHClient
from yolo_auto.state_store import JobStateStore
from yolo_auto.tracker import MLflowTracker


def register_resources(
    mcp: FastMCP,
    settings: Settings,
    ssh_clients_by_env: dict[str, SSHClient],
    state_store: JobStateStore,
    tracker: MLflowTracker,
    cvat_client_factory: Callable[[], CVATClient] | None = None,
) -> None:
    """Register MCP Resources for read-only background context."""

    ssh_client_default = ssh_clients_by_env.get("default")
    if ssh_client_default is None:
        # 兜底：至少保证资源可以工作（避免 KeyError）。
        ssh_client_default = next(iter(ssh_clients_by_env.values()))

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
                "ssh": {
                    "host": settings.yolo_ssh_host,
                    "port": settings.yolo_ssh_port,
                    "user": settings.yolo_ssh_user,
                },
                "workDir": settings.yolo_work_dir,
                "datasetsDir": settings.yolo_datasets_dir,
                "jobsDir": settings.yolo_jobs_dir,
                "modelsDir": settings.yolo_models_dir,
                "stateFile": settings.yolo_state_file,
                "mlflow": {
                    "trackingUri": settings.mlflow_tracking_uri,
                    "experimentName": settings.mlflow_experiment_name,
                    "externalUrl": settings.mlflow_external_url,
                    "leaderboardFilter": settings.mlflow_leaderboard_filter,
                },
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
        description="当前运行中或排队中的训练任务列表（从本地状态文件读取，不触发 SSH）。",
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
        description="最近 50 条训练任务记录（含已完成/失败/停止），不触发 SSH。",
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
        "yolo://mlflow/leaderboard",
        name="mlflow-leaderboard",
        description=(
            "MLflow 实验中按主指标排序的 Top 10 runs；"
            "可选环境变量 MLFLOW_LEADERBOARD_FILTER（MLflow search_runs 的 filter_string 语法）缩小范围。"
        ),
        mime_type="application/json",
    )
    def resource_mlflow_leaderboard() -> str:
        top_runs = tracker.summarize_top_runs(
            settings.primary_metric_key,
            limit=10,
            filter_string=settings.mlflow_leaderboard_filter,
        )
        return json.dumps(
            {
                "metricKey": settings.primary_metric_key,
                "filter": settings.mlflow_leaderboard_filter,
                "topRuns": top_runs,
                "count": len(top_runs),
            },
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "yolo://datasets",
        name="remote-datasets",
        description="远程服务器 datasets 目录下的 YAML 数据集配置文件列表。",
        mime_type="application/json",
    )
    def resource_datasets() -> str:
        try:
            stdout, _, code = ssh_client_default.execute(
                f"find {settings.yolo_datasets_dir} -maxdepth 3 -name '*.yaml' -o -name '*.yml'"
                " 2>/dev/null | head -50 | sort",
                timeout=10,
            )
            if code != 0:
                files: list[str] = []
            else:
                files = [line.strip() for line in stdout.splitlines() if line.strip()]
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

        cmd = f"mc ls {remote_dir}/ --json 2>/dev/null"
        try:
            stdout, _, code = ssh_client_default.execute(cmd, timeout=20)
        except Exception:
            stdout, code = "", 1

        items: list[dict[str, Any]] = []
        if code == 0 and stdout.strip():
            for line in stdout.splitlines():
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                key = str(obj.get("key", ""))
                if not key or key.endswith("/"):
                    continue
                items.append(
                    {
                        "filename": key,
                        "size": int(obj.get("size", 0) or 0),
                        "lastModified": str(obj.get("lastModified", "")),
                    }
                )

        items.sort(key=lambda x: x["lastModified"], reverse=True)
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
        description="远程服务器 models 目录下的预训练权重文件（.pt）列表及大小。",
        mime_type="application/json",
    )
    def resource_models() -> str:
        try:
            cmd = (
                f"find {settings.yolo_models_dir} -maxdepth 2 -name '*.pt' "
                "-exec ls -lh {{}} \\; 2>/dev/null | awk '{print $5, $NF}' | sort"
            )
            stdout, _, code = ssh_client_default.execute(cmd, timeout=10)
            if code != 0:
                models: list[dict[str, str]] = []
            else:
                models = []
                for line in stdout.splitlines():
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        models.append({"size": parts[0], "path": parts[1]})
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
        description="远程服务器 GPU 型号、显存、使用率、温度等（nvidia-smi）。",
        mime_type="application/json",
    )
    def resource_gpu_info() -> str:
        try:
            stdout, _, code = ssh_client_default.execute(
                "nvidia-smi --query-gpu=index,name,memory.total,memory.used,"
                "memory.free,utilization.gpu,utilization.memory,temperature.gpu"
                " --format=csv,noheader,nounits",
                timeout=10,
            )
            gpus: list[dict[str, str | int | float]] = []
            if code == 0:
                for line in stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 8:
                        gpus.append(
                            {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "memoryTotalMB": int(parts[2]),
                                "memoryUsedMB": int(parts[3]),
                                "memoryFreeMB": int(parts[4]),
                                "gpuUtil%": int(parts[5]),
                                "memUtil%": int(parts[6]),
                                "tempC": int(parts[7]),
                            }
                        )
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
        description="远程服务器 CPU、内存、磁盘概况。",
        mime_type="application/json",
    )
    def resource_system_info() -> str:
        info: dict[str, Any] = {}
        try:
            cpu_out, _, _ = ssh_client_default.execute(
                "nproc && lscpu | grep 'Model name' | sed 's/.*: *//'",
                timeout=10,
            )
            cpu_lines = cpu_out.strip().splitlines()
            info["cpu"] = {
                "cores": int(cpu_lines[0]) if cpu_lines else 0,
                "model": cpu_lines[1].strip() if len(cpu_lines) > 1 else "",
            }

            mem_out, _, _ = ssh_client_default.execute(
                "free -m | awk '/^Mem:/{print $2,$3,$4}'",
                timeout=10,
            )
            mem_parts = mem_out.strip().split()
            if len(mem_parts) >= 3:
                info["memoryMB"] = {
                    "total": int(mem_parts[0]),
                    "used": int(mem_parts[1]),
                    "free": int(mem_parts[2]),
                }

            disk_out, _, _ = ssh_client_default.execute(
                "df -BG /workspace 2>/dev/null | awk 'NR==2{print $2,$3,$4,$5}'",
                timeout=10,
            )
            disk_parts = disk_out.strip().split()
            if len(disk_parts) >= 4:
                info["diskWorkspace"] = {
                    "totalGB": disk_parts[0].rstrip("G"),
                    "usedGB": disk_parts[1].rstrip("G"),
                    "availGB": disk_parts[2].rstrip("G"),
                    "usePercent": disk_parts[3],
                }
        except Exception:
            pass
        return json.dumps(info, ensure_ascii=False, indent=2)

    @mcp.resource(
        "yolo://jobs/{jobId}/log",
        name="job-log",
        description="读取远程训练进程的 train.log 尾部内容（按 job.envId 选择对应 SSH）。",
        mime_type="text/plain",
    )
    def resource_job_log(jobId: str) -> str:
        record = state_store.get(jobId)
        if not record:
            return f"job not found: {jobId}"

        log_path = record.paths.get("logPath", "")
        if not log_path:
            return f"missing logPath for job: {jobId}"

        ssh_client = ssh_clients_by_env.get(record.env_id, ssh_client_default)
        if not ssh_client:
            return f"missing SSH client for envId={record.env_id}"

        tail, _, code = ssh_client.tail_file(log_path, lines=200)
        if code != 0:
            return f"failed to tail logPath={log_path}\n\n{tail}"
        return tail

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
| learningRate | lr0 | float | 0.001–0.02 | SGD 常用 0.01, AdamW 常用 0.001 |

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

    @mcp.resource(
        "yolo://cvat/projects",
        name="cvat-projects",
        description="CVAT 项目列表（需配置 CVAT_URL/CVAT_TOKEN）。",
        mime_type="application/json",
    )
    def resource_cvat_projects() -> str:
        client = _safe_get_cvat_client(cvat_client_factory)
        if client is None:
            return json.dumps(
                {
                    "ok": False,
                    "error": "CVAT is not configured",
                    "hint": "请设置 CVAT_URL、CVAT_TOKEN",
                },
                ensure_ascii=False,
                indent=2,
            )
        try:
            projects = client.list_projects()
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            )
        return json.dumps(
            {"ok": True, "projects": projects, "count": len(projects)},
            ensure_ascii=False,
            indent=2,
        )

    @mcp.resource(
        "yolo://cvat/tasks",
        name="cvat-tasks",
        description="CVAT 任务列表（需配置 CVAT_URL/CVAT_TOKEN）。",
        mime_type="application/json",
    )
    def resource_cvat_tasks() -> str:
        client = _safe_get_cvat_client(cvat_client_factory)
        if client is None:
            return json.dumps(
                {
                    "ok": False,
                    "error": "CVAT is not configured",
                    "hint": "请设置 CVAT_URL、CVAT_TOKEN",
                },
                ensure_ascii=False,
                indent=2,
            )
        try:
            tasks = client.list_tasks()
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            )
        return json.dumps(
            {"ok": True, "tasks": tasks, "count": len(tasks)},
            ensure_ascii=False,
            indent=2,
        )


def _safe_get_cvat_client(
    cvat_client_factory: Callable[[], CVATClient] | None,
) -> CVATClient | None:
    if cvat_client_factory is None:
        return None
    try:
        client = cvat_client_factory()
    except Exception:
        return None
    if not isinstance(client, CVATClient):
        return None
    return client


