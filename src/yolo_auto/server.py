from __future__ import annotations

import json
import time
from typing import Annotated, Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from yolo_auto.config import load_settings
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.models import JobStatus
from yolo_auto.ssh_client import SSHClient, SSHConfig
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.jobs import get_job, list_jobs
from yolo_auto.tools.setup_env import setup_env
from yolo_auto.tools.status import get_status
from yolo_auto.tools.training import TrainRequest, start_training, stop_training
from yolo_auto.tools.tuner import TuneRequest, auto_tune
from yolo_auto.tracker import MLflowTracker, TrackerConfig

mcp = FastMCP("yolo-auto")
SETTINGS = load_settings()
SSH = SSHClient(
    SSHConfig(
        host=SETTINGS.yolo_ssh_host,
        port=SETTINGS.yolo_ssh_port,
        user=SETTINGS.yolo_ssh_user,
        key_path=SETTINGS.yolo_ssh_key_path,
    )
)
NOTIFIER = FeishuNotifier(SETTINGS.feishu_webhook_url, SETTINGS.feishu_message_mode)
TRACKER = MLflowTracker(
    TrackerConfig(
        tracking_uri=SETTINGS.mlflow_tracking_uri,
        experiment_name=SETTINGS.mlflow_experiment_name,
    )
)
STATE_STORE = JobStateStore(SETTINGS.yolo_state_file)


def _merge_training_cli_extras(
    extra_args: dict[str, Any] | None,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(extra_args or {})
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


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
        extra_args=merged_extras,
    )
    return start_training(req, SSH, NOTIFIER, TRACKER, STATE_STORE)


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
    return get_status(
        jobId,
        runId,
        STATE_STORE,
        SSH,
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
    return stop_training(jobId, runId, SSH, NOTIFIER, TRACKER, STATE_STORE)


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
    return auto_tune(req, SSH, NOTIFIER, TRACKER, STATE_STORE)


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
    return list_jobs(STATE_STORE, SSH, limit=limit)


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
        SSH,
        TRACKER,
        NOTIFIER,
        refresh=refresh,
        feishu_report_enable=SETTINGS.feishu_report_enable,
        feishu_report_every_n_epochs=SETTINGS.feishu_report_every_n_epochs,
        primary_metric_key=SETTINGS.primary_metric_key,
    )


# ---------------------------------------------------------------------------
# MCP Resources — 只读上下文，供模型按需获取
# ---------------------------------------------------------------------------


@mcp.resource(
    "yolo://config",
    name="current-config",
    description="当前生效的环境配置概要（已脱敏，不含密钥与 webhook 完整 URL）。",
    mime_type="application/json",
)
def resource_config() -> str:
    return json.dumps(
        {
            "ssh": {
                "host": SETTINGS.yolo_ssh_host,
                "port": SETTINGS.yolo_ssh_port,
                "user": SETTINGS.yolo_ssh_user,
            },
            "workDir": SETTINGS.yolo_work_dir,
            "datasetsDir": SETTINGS.yolo_datasets_dir,
            "jobsDir": SETTINGS.yolo_jobs_dir,
            "modelsDir": SETTINGS.yolo_models_dir,
            "stateFile": SETTINGS.yolo_state_file,
            "mlflow": {
                "trackingUri": SETTINGS.mlflow_tracking_uri,
                "experimentName": SETTINGS.mlflow_experiment_name,
            },
            "feishu": {
                "reportEnable": SETTINGS.feishu_report_enable,
                "reportEveryNEpochs": SETTINGS.feishu_report_every_n_epochs,
                "messageMode": SETTINGS.feishu_message_mode,
            },
            "primaryMetric": SETTINGS.primary_metric_key,
            "watchPollIntervalSeconds": SETTINGS.watch_poll_interval_seconds,
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
    records = STATE_STORE.list_all()
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
    records = STATE_STORE.list_all()[:50]
    items = [r.to_dict() for r in records]
    return json.dumps(
        {"jobs": items, "count": len(items)},
        ensure_ascii=False,
        indent=2,
    )


@mcp.resource(
    "yolo://mlflow/leaderboard",
    name="mlflow-leaderboard",
    description="MLflow 实验中按主指标排序的 Top 10 runs 排行榜。",
    mime_type="application/json",
)
def resource_mlflow_leaderboard() -> str:
    top_runs = TRACKER.summarize_top_runs(SETTINGS.primary_metric_key, limit=10)
    return json.dumps(
        {
            "metricKey": SETTINGS.primary_metric_key,
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
        stdout, _, code = SSH.execute(
            f"find {SETTINGS.yolo_datasets_dir} -maxdepth 3 -name '*.yaml' -o -name '*.yml'"
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
            "datasetsDir": SETTINGS.yolo_datasets_dir,
            "files": files,
            "count": len(files),
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
        stdout, _, code = SSH.execute(
            f"find {SETTINGS.yolo_models_dir} -maxdepth 2 -name '*.pt' -exec"
            " ls -lh {} \\; 2>/dev/null | awk '{print $5, $NF}' | sort",
            timeout=10,
        )
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
            "modelsDir": SETTINGS.yolo_models_dir,
            "models": models,
            "count": len(models),
        },
        ensure_ascii=False,
        indent=2,
    )


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


# ---------------------------------------------------------------------------
# MCP Prompts — CEO 视角的高层工作流模板
# ---------------------------------------------------------------------------


@mcp.prompt(name="quick-train")
def quick_train_prompt(
    dataset: str,
    model: str = "yolov8n.pt",
    epochs: str = "100",
) -> str:
    """一句话启动训练：环境检查 → 启动 → 首次状态确认。"""
    return (
        f"请帮我完成以下 YOLO 训练流程（按顺序执行，每步失败立即告知原因与修复建议）：\n"
        f"\n"
        f"1. 调用 yolo_setup_env（dataConfigPath=\"{dataset}\"）确认远程环境就绪\n"
        f"2. 环境就绪后调用 yolo_start_training：\n"
        f"   - model = \"{model}\"\n"
        f"   - dataConfigPath = \"{dataset}\"\n"
        f"   - epochs = {epochs}\n"
        f"   - imgSize = 640, batch = 16, learningRate = 0.01\n"
        f"3. 启动成功后等待约 30 秒，调用 yolo_get_status 确认训练已开始运行\n"
        f"4. 用一段简洁摘要回复：任务 ID、MLflow runId、关键参数、飞书是否已通知\n"
    )


@mcp.prompt(name="dashboard")
def dashboard_prompt() -> str:
    """全局状态看板：一眼掌握所有训练任务进展。"""
    return (
        "请帮我生成当前训练状态看板：\n"
        "\n"
        "1. 调用 yolo_list_jobs 获取最近所有任务\n"
        "2. 对状态为 running 的任务逐个调用 yolo_get_job（refresh=true）刷新指标\n"
        "3. 用表格汇总：任务 ID | 状态 | 模型 | 当前 epoch/总 epoch | 主指标 | 启动时间\n"
        "4. 运行中的任务给出预估剩余时间\n"
        "5. 失败的任务简述原因\n"
        "6. 最后用一句话总结全局情况（如「2 个运行中、1 个已完成、最佳 mAP 72.3%」）\n"
    )


@mcp.prompt(name="compare-experiments")
def compare_experiments_prompt(job_ids: str) -> str:
    """对比多个实验并给出最佳推荐。job_ids 逗号分隔。"""
    ids = [jid.strip() for jid in job_ids.split(",")]
    job_list = "\n".join(f"   - {jid}" for jid in ids)
    return (
        f"请对比以下训练实验并给出推荐：\n"
        f"\n"
        f"1. 逐个调用 yolo_get_job（refresh=true）获取详情：\n"
        f"{job_list}\n"
        f"2. 输出对比表格：模型 | epochs | batch | lr | 主指标(mAP) | 训练时长 | 状态\n"
        f"3. 分析：\n"
        f"   - 哪个指标最好？领先幅度？\n"
        f"   - 参数差异对结果的关键影响\n"
        f"   - 是否有过拟合/欠拟合迹象\n"
        f"4. 给出明确结论：推荐哪个方案，以及下一步行动建议（继续调参/增加 epoch/换模型）\n"
    )


@mcp.prompt(name="smart-tune")
def smart_tune_prompt(
    dataset: str,
    model: str = "yolov8n.pt",
    goal: str = "精度与速度兼顾",
) -> str:
    """根据业务目标智能推荐调参策略并一键执行。"""
    return (
        f"请帮我制定并执行智能调参方案：\n"
        f"\n"
        f"目标：{goal}\n"
        f"模型：{model}\n"
        f"数据集：{dataset}\n"
        f"\n"
        f"1. 调用 yolo_setup_env 确认环境\n"
        f"2. 调用 yolo_list_jobs 查看历史实验，避免重复参数\n"
        f"3. 根据目标「{goal}」设计搜索空间：\n"
        f"   - 追求精度 → imgSize [640,1280], batch [8,16], lr [0.001,0.01]\n"
        f"   - 追求速度 → imgSize [320,640], batch [32,64], lr [0.01,0.02]\n"
        f"   - 兼顾平衡 → imgSize [640], batch [16,32], lr [0.005,0.01,0.02]\n"
        f"4. 调用 yolo_auto_tune，epochs 先用 30 做快速筛选，maxTrials 不超过 6\n"
        f"5. 输出报告：最佳参数、各 trial 对比、是否建议用最佳参数跑完整 epoch\n"
    )


@mcp.prompt(name="diagnose")
def diagnose_prompt(job_id: str) -> str:
    """诊断训练任务异常：检查 → 定位 → 给出修复方案。"""
    return (
        f"任务 {job_id} 可能出了问题，请帮我诊断：\n"
        f"\n"
        f"1. 调用 yolo_get_job（jobId=\"{job_id}\", refresh=true）获取最新状态与指标\n"
        f"2. 根据状态判断：\n"
        f"   - running 但长时间无新 epoch → 是否卡住（GPU 挂起/OOM 后静默失败）\n"
        f"   - failed → 分析错误（OOM、数据路径、SSH 断连、磁盘满）\n"
        f"   - completed 但指标差 → 欠拟合（epoch 不够/lr 太小）或过拟合（train↑ val↓）\n"
        f"3. 给出诊断结论和修复方案\n"
        f"4. 如果需要重跑，直接给出调整后的推荐参数\n"
    )


@mcp.prompt(name="report")
def report_prompt(period: str = "今天") -> str:
    """生成可直接转发给团队/上级的训练进展报告。"""
    return (
        f"请帮我生成「{period}」的训练进展报告（适合发飞书群/汇报）：\n"
        f"\n"
        f"1. 调用 yolo_list_jobs 获取相关任务\n"
        f"2. 对运行中/已完成的任务调用 yolo_get_job（refresh=true）刷新数据\n"
        f"3. 按以下结构输出报告：\n"
        f"\n"
        f"   【训练进展摘要】\n"
        f"   - 本轮共运行 X 个实验\n"
        f"   - 当前最佳：[模型] mAP=XX.X%（任务 ID）\n"
        f"   - 相比上次提升/下降 X 个百分点\n"
        f"\n"
        f"   【实验明细】\n"
        f"   （表格：任务 ID | 模型 | 关键参数 | 主指标 | 状态）\n"
        f"\n"
        f"   【下一步计划】\n"
        f"   - 基于当前结果的优化方向\n"
        f"   - 需要的资源或配置调整\n"
        f"\n"
        f"报告要简洁专业、数据准确、结论清晰。\n"
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
