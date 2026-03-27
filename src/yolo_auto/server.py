from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Annotated, Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.responses import PlainTextResponse

from yolo_auto.config import load_settings
from yolo_auto.cvat_client import CVATClient, CVATConfig
from yolo_auto.errors import err
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.prompts import register_prompts
from yolo_auto.resources import register_resources
from yolo_auto.ssh_client import SSHClient, SSHConfig
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.dataset import (
    analyze_cvat_task,
    export_cvat_dataset,
    get_cvat_task_detail,
    list_cvat_formats,
    list_cvat_projects,
    list_cvat_tasks,
)
from yolo_auto.tools.export import run_export
from yolo_auto.tools.jobs import get_job, list_jobs
from yolo_auto.tools.setup_env import setup_env
from yolo_auto.tools.status import get_status
from yolo_auto.tools.training import TrainRequest, start_training, stop_training
from yolo_auto.tools.tuner import TuneRequest, auto_tune
from yolo_auto.tools.validate import run_validation
from yolo_auto.tracker import MLflowTracker, TrackerConfig

_SERVER_INSTRUCTIONS = """\
YOLO Auto — 远程 GPU 容器上的 YOLO 目标检测训练编排 MCP 服务。
通过 SSH 管理 Ultralytics YOLO 训练生命周期，集成 MLflow 追踪与飞书通知。

## 何时使用
当用户提到以下关键词时，优先使用本服务的工具与资源：
训练、YOLO、目标检测、模型训练、调参、GPU、数据集、mAP、验证、导出模型、
训练进度、实验对比、超参搜索、学习率、batch size、epochs。

## 可用资源（只读上下文，建议在操作前先获取）
- `yolo://config` — 当前配置概要（路径、MLflow、飞书等），开始任何操作前建议先读取
- `yolo://datasets` — 远程可用数据集列表，选择数据集时读取
- `yolo://models` — 远程可用预训练权重列表，选择模型时读取
- `yolo://env/gpu` — GPU 状态（显存、使用率），决定 batch/device 时读取
- `yolo://env/system` — CPU/内存/磁盘概况
- `yolo://jobs/active` — 当前运行中的任务，查看进度时读取
- `yolo://jobs/history` — 历史任务记录
- `yolo://mlflow/leaderboard` — 实验排行榜，对比实验时读取
- `yolo://guide/training-params` — 训练参数速查，推荐参数时参考
- `yolo://cvat/projects` — CVAT 项目列表，选择要导出的标注任务时读取
- `yolo://cvat/tasks` — CVAT 任务列表，查看任务规模与状态时读取

## 标准工作流
1. **环境检查** → 调用 `yolo_setup_env` 验证远程环境与数据配置
2. **数据集管理（可选）** → `cvat_list_projects` / `cvat_list_tasks` / `cvat_analyze_dataset`
3. **确认导出格式（可选）** → `cvat_list_formats` 查看 CVAT 当前支持的导入导出格式
4. **数据集导出（可选）** → `cvat_export_dataset` 导出到 CVAT Cloud Storage
5. **启动训练** → 调用 `yolo_start_training`（异步，立即返回 jobId + runId）
6. **监控进度** → 周期调用 `yolo_get_status`（传入 jobId + runId）
7. **任务管理** → `yolo_list_jobs` 列出所有任务，`yolo_get_job` 查询单任务详情
8. **停止训练** → `yolo_stop_training`（传入 jobId + runId）
9. **验证评估** → `yolo_validate`（在验证集上运行已完成任务的 best 权重）
10. **导出模型** → `yolo_export`（导出 ONNX/TensorRT/CoreML 等格式）
11. **自动调参** → `yolo_auto_tune`（网格搜索 lr/batch/imgSize，串行执行多个 trial）

## 关键约定
- 远程路径：数据集在 `/workspace/datasets/`，模型权重在 `/workspace/models/`
- 启动训练前务必用 `yolo_setup_env` 验证环境
- `yolo_get_status` 和 `yolo_stop_training` 都需要 jobId + runId（从 start_training 返回获取）
- 多环境时通过 `envId` 参数选择不同的 SSH 连接
"""

mcp = FastMCP("yolo-auto", instructions=_SERVER_INSTRUCTIONS)
SETTINGS = load_settings()
SSH_BY_ENV: dict[str, SSHClient] = {}
for env_id, ssh_env in SETTINGS.yolo_ssh_envs.items():
    SSH_BY_ENV[env_id] = SSHClient(
        SSHConfig(
            host=ssh_env.host,
            port=ssh_env.port,
            user=ssh_env.user,
            key_path=ssh_env.key_path,
        )
    )
SSH = SSH_BY_ENV["default"]
NOTIFIER = FeishuNotifier(
    webhook_url=SETTINGS.feishu_webhook_url,
    app_id=SETTINGS.feishu_app_id,
    app_secret=SETTINGS.feishu_app_secret,
    chat_id=SETTINGS.feishu_chat_id,
)
TRACKER = MLflowTracker(
    TrackerConfig(
        tracking_uri=SETTINGS.mlflow_tracking_uri,
        experiment_name=SETTINGS.mlflow_experiment_name,
        external_url=SETTINGS.mlflow_external_url,
    )
)
STATE_STORE = JobStateStore(SETTINGS.yolo_state_file)
CVAT: CVATClient | None = None


def _merge_training_cli_extras(
    extra_args: dict[str, Any] | None,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(extra_args or {})
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _get_cvat_client() -> CVATClient:
    global CVAT
    if CVAT is not None:
        return CVAT

    if not SETTINGS.cvat_url or not SETTINGS.cvat_token:
        raise ValueError(
            "CVAT not configured. Please set CVAT_URL and CVAT_TOKEN."
        )

    CVAT = CVATClient(
        CVATConfig(
            url=SETTINGS.cvat_url,
            token=SETTINGS.cvat_token,
            org_slug=SETTINGS.cvat_org_slug,
        )
    )
    return CVAT


def _with_cvat_client(action: Callable[[CVATClient], dict[str, Any]]) -> dict[str, Any]:
    try:
        client = _get_cvat_client()
    except Exception as exc:
        return err(
            error_code="CVAT_NOT_CONFIGURED",
            message=str(exc),
            retryable=False,
            hint="请先设置 CVAT_URL、CVAT_TOKEN",
        )
    try:
        return action(client)
    except Exception as exc:
        return err(
            error_code="CVAT_REQUEST_FAILED",
            message=str(exc),
            retryable=True,
            hint="检查 CVAT 服务可达性、账号权限和 task/project ID",
        )


@mcp.tool(name="yolo_setup_env")
def yolo_setup_env(
    dataConfigPath: Annotated[
        str,
        Field(description="数据集 YAML 路径，用于在远程 YOLO_WORK_DIR 下做存在性等检查。"),
    ],
) -> dict[str, Any]:
    """训练前检查远程环境与数据配置是否就绪（SSH 到配置的主机）。

    建议工作流：先调用本工具，再 yolo_start_training → 周期性 yolo_get_status。
    返回字典含 ok、错误时含 error/hint；与具体字段以 setup_env 实现为准。
    """
    return setup_env(SSH, SETTINGS.yolo_work_dir, dataConfigPath)


@mcp.tool(name="cvat_list_projects")
def cvat_list_projects() -> dict[str, Any]:
    """列出 CVAT 项目。"""
    return _with_cvat_client(list_cvat_projects)


@mcp.tool(name="cvat_list_tasks")
def cvat_list_tasks(
    projectId: Annotated[
        int | None,
        Field(
            default=None,
            description="可选 projectId；提供时仅返回该项目下任务。",
        ),
    ] = None,
) -> dict[str, Any]:
    """列出 CVAT 任务，可按 projectId 过滤。"""
    return _with_cvat_client(lambda client: list_cvat_tasks(client, project_id=projectId))


@mcp.tool(name="cvat_list_formats")
def cvat_list_formats() -> dict[str, Any]:
    """列出 CVAT 当前实例支持的导入/导出标注格式。"""
    return _with_cvat_client(list_cvat_formats)


@mcp.tool(name="cvat_get_task")
def cvat_get_task(
    taskId: Annotated[int, Field(gt=0, description="CVAT 任务 ID。")],
) -> dict[str, Any]:
    """获取单个 CVAT 任务详情。"""
    return _with_cvat_client(lambda client: get_cvat_task_detail(client, taskId))


@mcp.tool(name="cvat_analyze_dataset")
def cvat_analyze_dataset(
    taskId: Annotated[int, Field(gt=0, description="CVAT 任务 ID。")],
) -> dict[str, Any]:
    """分析指定 CVAT 任务的数据集分布。"""
    return _with_cvat_client(lambda client: analyze_cvat_task(client, taskId))


@mcp.tool(name="cvat_export_dataset")
def cvat_export_dataset(
    taskId: Annotated[int, Field(gt=0, description="CVAT 任务 ID。")],
    datasetName: Annotated[
        str,
        Field(description="导出文件名中使用的数据集名称（会自动清洗）。"),
    ],
    formatName: Annotated[
        str,
        Field(
            default="Ultralytics YOLO Detection 1.0",
            description="CVAT 导出格式，默认 Ultralytics YOLO Detection 1.0。",
        ),
    ] = "Ultralytics YOLO Detection 1.0",
    cloudStorageId: Annotated[
        int | None,
        Field(
            default=None,
            description=(
                "可选：CVAT 云存储 ID。为空时默认使用 CVAT_CLOUD_STORAGE_ID；"
                "有值时会先导出到 CVAT Cloud Storage。"
            ),
        ),
    ] = None,
    cloudFilename: Annotated[
        str | None,
        Field(
            default=None,
            description="可选：导出到 CVAT 云存储的文件名（如 exports/my-task.zip）。",
        ),
    ] = None,
    statusCheckPeriod: Annotated[
        int | None,
        Field(
            default=None,
            description="可选：CVAT 导出队列轮询间隔（秒）。",
        ),
    ] = None,
    includeImages: Annotated[
        bool,
        Field(default=False, description="是否在导出包中包含原图，默认 false。"),
    ] = False,
) -> dict[str, Any]:
    """导出 CVAT 数据集到 CVAT Cloud Storage。"""
    return _with_cvat_client(
        lambda client: export_cvat_dataset(
            client,
            task_id=taskId,
            dataset_name=datasetName,
            format_name=formatName,
            include_images=includeImages,
            cloud_storage_id=(
                cloudStorageId
                if cloudStorageId is not None
                else SETTINGS.cvat_cloud_storage_id
            ),
            cloud_filename=cloudFilename,
            status_check_period=statusCheckPeriod,
        )
    )


@mcp.tool(name="yolo_start_training")
def yolo_start_training(
    model: Annotated[
        str,
        Field(
            description=(
                "模型：可为预训练权重路径（.pt）、模型结构 YAML（.yaml），"
                "可配合顶层 pretrained 或 extraArgs 控制是否加载预训练。"
            ),
        ),
    ],
    dataConfigPath: Annotated[
        str,
        Field(description="Ultralytics 数据集配置 YAML 路径（远程工作目录下可读）。"),
    ],
    epochs: Annotated[
        int,
        Field(gt=0, description="训练轮数；写入远程 yolo 命令的 epochs=。"),
    ],
    imgSize: Annotated[
        int,
        Field(gt=0, description="训练输入边长 imgsz=（正方形缩放，与 Ultralytics 一致）。"),
    ],
    batch: Annotated[
        int | float,
        Field(
            gt=0,
            description=(
                "批次：正整数为固定大小；正小数（如 0.7）在支持的 Ultralytics 中"
                "表示显存占用比例。"
            ),
        ),
    ],
    learningRate: Annotated[
        float,
        Field(gt=0, description="初始学习率，对应远程命令中的 lr0=。"),
    ],
    device: Annotated[
        str | int | list[Any] | None,
        Field(
            default=None,
            description=(
                "训练设备：如 0、cpu、mps；多卡可用列表。对应 Ultralytics 的 device=。"
            ),
        ),
    ] = None,
    patience: Annotated[
        int | None,
        Field(
            default=None,
            description="早停耐心轮数（验证无提升），对应 patience=。",
        ),
    ] = None,
    workers: Annotated[
        int | None,
        Field(
            default=None,
            description="DataLoader 线程数，对应 workers=。",
        ),
    ] = None,
    optimizer: Annotated[
        str | None,
        Field(
            default=None,
            description="优化器名（如 SGD、AdamW、auto 等），对应 optimizer=。",
        ),
    ] = None,
    pretrained: Annotated[
        bool | str | None,
        Field(
            default=None,
            description="是否/从何处加载预训练权重，对应 pretrained=（bool 或 .pt 路径）。",
        ),
    ] = None,
    resume: Annotated[
        bool | None,
        Field(
            default=None,
            description="是否从 checkpoint 续训，对应 resume=。",
        ),
    ] = None,
    cache: Annotated[
        bool | str | None,
        Field(
            default=None,
            description="数据缓存：False、True/ram、disk 等，对应 cache=。",
        ),
    ] = None,
    amp: Annotated[
        bool | None,
        Field(
            default=None,
            description="是否混合精度训练，对应 amp=。",
        ),
    ] = None,
    cosLr: Annotated[
        bool | None,
        Field(
            default=None,
            description="是否使用余弦学习率调度，对应 cos_lr=。",
        ),
    ] = None,
    save: Annotated[
        bool | None,
        Field(
            default=None,
            description="是否保存 checkpoint，对应 save=。",
        ),
    ] = None,
    val: Annotated[
        bool | None,
        Field(
            default=None,
            description="训练过程是否做验证，对应 val=。",
        ),
    ] = None,
    plots: Annotated[
        bool | None,
        Field(
            default=None,
            description="是否保存曲线/可视化，对应 plots=。",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        Field(
            default=None,
            description="随机种子，对应 seed=。",
        ),
    ] = None,
    deterministic: Annotated[
        bool | None,
        Field(
            default=None,
            description="是否尽量确定性算法，对应 deterministic=。",
        ),
    ] = None,
    closeMosaic: Annotated[
        int | None,
        Field(
            default=None,
            description="最后若干 epoch 关闭 mosaic，对应 close_mosaic=。",
        ),
    ] = None,
    freeze: Annotated[
        int | list[int] | None,
        Field(
            default=None,
            description="冻结前 n 层或指定层索引列表，对应 freeze=。",
        ),
    ] = None,
    singleCls: Annotated[
        bool | None,
        Field(
            default=None,
            description="多类数据按单类训练，对应 single_cls=。",
        ),
    ] = None,
    fraction: Annotated[
        float | None,
        Field(
            default=None,
            description="使用数据集比例（0–1），对应 fraction=。",
        ),
    ] = None,
    trainTimeHours: Annotated[
        float | None,
        Field(
            default=None,
            description="最长训练时间（小时），设置时会作用于 Ultralytics time=。",
        ),
    ] = None,
    envId: Annotated[
        str,
        Field(
            default="default",
            description="训练运行环境 ID；用于选择对应 SSH（来自 YOLO_SSH_ENVS）。",
        ),
    ] = "default",
    jobId: Annotated[
        str | None,
        Field(
            default=None,
            description="可选任务 ID；不传则自动生成。用于日志目录与 stop 时匹配进程。",
        ),
    ] = None,
    extraArgs: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description=(
                "其余未在顶层暴露的 Ultralytics 参数，键为 CLI 名（如 lrf、warmup_epochs）。"
                "勿传 project、name；与顶层同名键时以顶层为准。"
            ),
        ),
    ] = None,
) -> dict[str, Any]:
    """在远程 GPU 容器后台启动 Ultralytics「yolo detect train」（异步，不阻塞 MCP）。

    会创建 MLflow run、写入本地任务状态、并可能发送飞书「训练已启动」通知。
    成功时返回含 jobId、runId、status、paths（日志与权重路径提示）等；
    重复启动同 job 且仍在队列/运行中会直接返回已有记录。
    """
    resolved_job_id = jobId or f"job-{int(time.time())}-{uuid4().hex[:8]}"
    merged_extras = _merge_training_cli_extras(
        extraArgs,
        {
            "device": device,
            "patience": patience,
            "workers": workers,
            "optimizer": optimizer,
            "pretrained": pretrained,
            "resume": resume,
            "cache": cache,
            "amp": amp,
            "cos_lr": cosLr,
            "save": save,
            "val": val,
            "plots": plots,
            "seed": seed,
            "deterministic": deterministic,
            "close_mosaic": closeMosaic,
            "freeze": freeze,
            "single_cls": singleCls,
            "fraction": fraction,
            "time": trainTimeHours,
        },
    )
    req = TrainRequest(
        job_id=resolved_job_id,
        model=model,
        data_config_path=dataConfigPath,
        epochs=epochs,
        img_size=imgSize,
        batch=batch,
        learning_rate=learningRate,
        work_dir=SETTINGS.yolo_work_dir,
        jobs_dir=SETTINGS.yolo_jobs_dir,
        env_id=envId,
        extra_args=merged_extras,
    )
    ssh_client = SSH_BY_ENV.get(envId, SSH)
    return start_training(req, ssh_client, NOTIFIER, TRACKER, STATE_STORE)


@mcp.tool(name="yolo_get_status")
def yolo_get_status(
    jobId: Annotated[str, Field(description="训练任务 ID。")],
    runId: Annotated[
        str,
        Field(description="MLflow runId；与 jobId 一起用于更新指标与 MLflow 状态。"),
    ],
) -> dict[str, Any]:
    """拉取指定任务的训练进度（SSH 读 results.csv）、回写 MLflow 指标，并按配置推送飞书里程碑/终态。

    训练进行中可反复调用；返回含 ok、status、metrics 等。需使用启动时返回的 jobId 与 runId。
    """
    record = STATE_STORE.get(jobId)
    ssh_client = SSH_BY_ENV.get(record.env_id, SSH) if record else SSH
    return get_status(
        jobId,
        runId,
        STATE_STORE,
        ssh_client,
        TRACKER,
        NOTIFIER,
        feishu_report_enable=SETTINGS.feishu_report_enable,
        feishu_report_every_n_epochs=SETTINGS.feishu_report_every_n_epochs,
        primary_metric_key=SETTINGS.primary_metric_key,
    )


@mcp.tool(name="yolo_stop_training")
def yolo_stop_training(
    jobId: Annotated[str, Field(description="要停止的训练任务 ID（与启动时一致）。")],
    runId: Annotated[
        str,
        Field(description="该任务对应的 MLflow runId（yolo_start_training 返回的 runId）。"),
    ],
) -> dict[str, Any]:
    """请求停止远程训练进程，并结束对应 MLflow run（kill/标记以 stop_training 实现为准）。

    需提供 jobId 与 runId；成功返回更新后的任务状态或停止确认字段。
    """
    record = STATE_STORE.get(jobId)
    ssh_client = SSH_BY_ENV.get(record.env_id, SSH) if record else SSH
    return stop_training(jobId, runId, ssh_client, NOTIFIER, TRACKER, STATE_STORE)


@mcp.tool(name="yolo_validate")
def yolo_validate(
    jobId: Annotated[str, Field(description="要在验证集上运行验证的任务 ID。")],
    dataConfigPath: Annotated[
        str | None,
        Field(
            description="可选：验证数据集 YAML（远程）。不传则用 job record 的 dataConfigPath。"
        ),
    ] = None,
    imgSize: Annotated[
        int | None,
        Field(description="可选：验证 imgsz=。不传则使用 Ultralytics 默认。"),
    ] = None,
    batch: Annotated[
        int | None,
        Field(description="可选：验证 batch=。不传则使用 Ultralytics 默认。"),
    ] = None,
    device: Annotated[
        str | None,
        Field(description="可选：验证 device=（如 0、cpu）。"),
    ] = None,
    extraArgs: Annotated[
        dict[str, Any] | None,
        Field(description="可选：透传 Ultralytics CLI 参数，转为 key=value 形式。"),
    ] = None,
) -> dict[str, Any]:
    """基于训练完成后的最佳权重在验证集上跑 yolo detect val。"""
    record = STATE_STORE.get(jobId)
    ssh_client = SSH_BY_ENV.get(record.env_id, SSH) if record else SSH
    return run_validation(
        jobId,
        STATE_STORE,
        ssh_client,
        SETTINGS.yolo_jobs_dir,
        SETTINGS.yolo_work_dir,
        data_config_path=dataConfigPath,
        img_size=imgSize,
        batch=batch,
        device=device,
        extra_args=extraArgs,
    )


@mcp.tool(name="yolo_export")
def yolo_export(
    jobId: Annotated[str, Field(description="要导出的训练任务 ID。")],
    formats: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="导出格式列表（如 ['onnx','engine','coreml']）。不传则使用默认三种。",
        ),
    ] = None,
    imgSize: Annotated[
        int | None,
        Field(description="可选：export imgsz。"),
    ] = None,
    half: Annotated[
        bool | None,
        Field(description="可选：export half=True（支持的格式上生效）。"),
    ] = None,
    int8: Annotated[
        bool | None,
        Field(description="可选：export int8=True（支持的格式上生效）。"),
    ] = None,
    device: Annotated[
        str | None,
        Field(description="可选：export device=，如 0、cpu、mps。"),
    ] = None,
    extraArgs: Annotated[
        dict[str, Any] | None,
        Field(description="可选：透传 Ultralytics export CLI 参数（key=value）。"),
    ] = None,
) -> dict[str, Any]:
    """对已完成训练的 best 权重做 yolo export，并返回 job_dir 下找到的导出产物列表。"""
    record = STATE_STORE.get(jobId)
    ssh_client = SSH_BY_ENV.get(record.env_id, SSH) if record else SSH
    return run_export(
        job_id=jobId,
        state_store=STATE_STORE,
        ssh_client=ssh_client,
        jobs_dir=SETTINGS.yolo_jobs_dir,
        work_dir=SETTINGS.yolo_work_dir,
        formats=formats,
        img_size=imgSize,
        half=half,
        int8=int8,
        device=device,
        extra_args=extraArgs,
    )


@mcp.tool(name="yolo_auto_tune")
def yolo_auto_tune(
    envId: Annotated[
        str,
        Field(description="环境/实验标识字符串，会写入返回 JSON，便于区分调参批次。"),
    ],
    baseJobId: Annotated[
        str,
        Field(description="trial 任务 ID 前缀；实际 jobId 为 {baseJobId}-t1、t2 …"),
    ],
    model: Annotated[str, Field(description="与 yolo_start_training 的 model 含义相同。")],
    dataConfigPath: Annotated[
        str,
        Field(description="数据集 YAML 路径，同 yolo_start_training。"),
    ],
    epochs: Annotated[int, Field(gt=0, description="每个 trial 的训练轮数。")],
    maxTrials: Annotated[
        int,
        Field(gt=0, description="最多运行多少个 trial（网格前列出顺序截断）。"),
    ],
    searchSpace: Annotated[
        dict[str, list[float | int]],
        Field(
            description=(
                "搜索空间：键名仅支持 learningRate、batch、imgSize（三者笛卡尔积为 trial 网格）。"
                "每项为候选值列表；未提供的键使用默认单值 [0.01]、[16]、[640]。"
            ),
        ),
    ],
) -> dict[str, Any]:
    """按 searchSpace 对 learningRate、batch、imgSize 做网格遍历
    （最多 maxTrials 个 trial），串行启动训练并轮询 get_status。

    返回最佳 trial、MLflow 顶部 run 摘要及 trials 列表；
    全部失败时返回 ok=false。耗时随 epochs 与 trial 数增长。
    """
    req = TuneRequest(
        env_id=envId,
        base_job_id=baseJobId,
        model=model,
        data_config_path=dataConfigPath,
        epochs=epochs,
        search_space=searchSpace,
        max_trials=maxTrials,
        work_dir=SETTINGS.yolo_work_dir,
        jobs_dir=SETTINGS.yolo_jobs_dir,
        primary_metric_key=SETTINGS.primary_metric_key,
        feishu_report_enable=SETTINGS.feishu_report_enable,
        feishu_report_every_n_epochs=SETTINGS.feishu_report_every_n_epochs,
    )
    ssh_client = SSH_BY_ENV.get(envId, SSH)
    return auto_tune(req, ssh_client, NOTIFIER, TRACKER, STATE_STORE)


@mcp.tool(name="yolo_list_jobs")
def yolo_list_jobs(
    limit: Annotated[
        int,
        Field(default=20, ge=1, le=100, description="返回最近任务条数上限（1–100）。"),
    ] = 20,
) -> dict[str, Any]:
    """列出本地状态存储中的最近训练任务摘要（含状态、epoch 提示等，具体字段见 list_jobs 实现）。

    用于在对话中快速查看有哪些 jobId 可继续 get_status 或 stop。
    """
    return list_jobs(STATE_STORE, SSH_BY_ENV, limit=limit)


@mcp.tool(name="yolo_get_job")
def yolo_get_job(
    jobId: Annotated[str, Field(description="任务 ID。")],
    refresh: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "为 true 时在返回前尝试拉取远程 results.csv 并刷新状态"
                "（等同串联 get_status 逻辑）。"
            ),
        ),
    ] = False,
) -> dict[str, Any]:
    """按 jobId 查询单条任务记录；refresh=true 时会顺带刷新远程指标与状态（类似 get_status）。

    返回存储中的 Job 详情及可选刷新后的指标。
    """
    return get_job(
        jobId,
        STATE_STORE,
        SSH_BY_ENV,
        TRACKER,
        NOTIFIER,
        refresh=refresh,
        feishu_report_enable=SETTINGS.feishu_report_enable,
        feishu_report_every_n_epochs=SETTINGS.feishu_report_every_n_epochs,
        primary_metric_key=SETTINGS.primary_metric_key,
    )


register_resources(
    mcp,
    SETTINGS,
    SSH_BY_ENV,
    STATE_STORE,
    TRACKER,
    cvat_client_factory=_get_cvat_client,
)
register_prompts(mcp)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(_request: Any) -> PlainTextResponse:
    return PlainTextResponse("OK")


def main() -> None:
    raw_transport = os.getenv("MCP_TRANSPORT", "streamable-http").strip().lower()
    if raw_transport in ("http", "streamable_http", "streamable-http"):
        transport = "streamable-http"
    elif raw_transport in ("sse", "stdio"):
        transport = raw_transport
    else:
        raise ValueError(f"Unsupported MCP_TRANSPORT: {raw_transport}")

    mount_path = os.getenv("MCP_MOUNT_PATH", "").strip() or None

    host = os.getenv("MCP_HOST", "").strip()
    port_raw = os.getenv("MCP_PORT", "").strip()
    if host:
        mcp.settings.host = host
    if port_raw:
        mcp.settings.port = int(port_raw)

    if host and host not in ("127.0.0.1", "localhost", "::1"):
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        )

    streamable_path = os.getenv("MCP_PATH", "").strip()
    if streamable_path:
        mcp.settings.streamable_http_path = streamable_path

    mcp.run(transport=transport, mount_path=mount_path)


if __name__ == "__main__":
    main()
