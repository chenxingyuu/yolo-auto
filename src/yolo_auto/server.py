from __future__ import annotations

import time
from typing import Annotated, Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from yolo_auto.config import load_settings
from yolo_auto.feishu import FeishuNotifier
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


class StartTrainingInput(BaseModel):
    model: str = Field(
        description=(
            "模型：可为预训练权重路径（.pt）、模型结构 YAML（.yaml），"
            "或按需配合 extraArgs.pretrained 控制是否加载预训练。"
        ),
    )
    dataConfigPath: str = Field(
        description="Ultralytics 数据集配置 YAML 路径（远程工作目录下可读）。",
    )
    epochs: int = Field(
        gt=0,
        description="训练轮数；写入远程 yolo 命令的 epochs=。",
    )
    imgSize: int = Field(
        gt=0,
        description="训练输入边长 imgsz=（正方形缩放，与 Ultralytics 一致）。",
    )
    batch: int | float = Field(
        gt=0,
        description=(
            "批次：正整数为固定大小；正小数（如 0.7）在支持的 Ultralytics 中"
            "表示显存占用比例。"
        ),
    )
    learningRate: float = Field(
        gt=0,
        description="初始学习率，对应远程命令中的 lr0=。",
    )
    jobId: str | None = Field(
        default=None,
        description="可选任务 ID；不传则自动生成。用于日志目录与 stop 时匹配进程。",
    )
    extraArgs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "额外 Ultralytics 训练参数，键值对会拼成 key=value 追加到 yolo detect train 命令。"
            "建议勿传入 project、name，以免覆盖本系统为任务分配的目录；其余如 device、patience、"
            "optimizer、resume、pretrained 等可按需填写。"
        ),
    )


class StopTrainingInput(BaseModel):
    jobId: str = Field(description="要停止的训练任务 ID（与启动时一致）。")
    runId: str = Field(
        description="该任务对应的 MLflow runId（yolo_start_training 返回的 runId）。",
    )


class StatusInput(BaseModel):
    jobId: str = Field(description="训练任务 ID。")
    runId: str = Field(
        description="MLflow runId；与 jobId 一起用于更新指标与 MLflow 状态。",
    )


class AutoTuneInput(BaseModel):
    envId: str = Field(
        description="环境/实验标识字符串，会写入返回 JSON，便于区分调参批次。",
    )
    baseJobId: str = Field(
        description="trial 任务 ID 前缀；实际 jobId 为 {baseJobId}-t1、t2 …",
    )
    model: str = Field(description="与 yolo_start_training 的 model 含义相同。")
    dataConfigPath: str = Field(description="数据集 YAML 路径，同 yolo_start_training。")
    epochs: int = Field(gt=0, description="每个 trial 的训练轮数。")
    maxTrials: int = Field(gt=0, description="最多运行多少个 trial（网格前列出顺序截断）。")
    searchSpace: dict[str, list[float | int]] = Field(
        description=(
            "搜索空间：键名仅支持 learningRate、batch、imgSize（三者笛卡尔积为 trial 网格）。"
            "每项为候选值列表；未提供的键使用默认单值 [0.01]、[16]、[640]。"
        ),
    )


class ListJobsInput(BaseModel):
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="返回最近任务条数上限（1–100）。",
    )


class GetJobInput(BaseModel):
    jobId: str = Field(description="任务 ID。")
    refresh: bool = Field(
        default=False,
        description=(
            "为 true 时在返回前尝试拉取远程 results.csv 并刷新状态"
            "（等同串联 get_status 逻辑）。"
        ),
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


@mcp.tool(name="yolo_start_training")
def yolo_start_training(payload: StartTrainingInput) -> dict[str, Any]:
    """在远程 GPU 容器后台启动 Ultralytics「yolo detect train」（异步，不阻塞 MCP）。

    会创建 MLflow run、写入本地任务状态、并可能发送飞书「训练已启动」通知。
    成功时返回含 jobId、runId、status、paths（日志与权重路径提示）等；
    重复启动同 job 且仍在队列/运行中会直接返回已有记录。
    """
    job_id = payload.jobId or f"job-{int(time.time())}-{uuid4().hex[:8]}"
    req = TrainRequest(
        job_id=job_id,
        model=payload.model,
        data_config_path=payload.dataConfigPath,
        epochs=payload.epochs,
        img_size=payload.imgSize,
        batch=payload.batch,
        learning_rate=payload.learningRate,
        work_dir=SETTINGS.yolo_work_dir,
        jobs_dir=SETTINGS.yolo_jobs_dir,
        extra_args=payload.extraArgs,
    )
    return start_training(req, SSH, NOTIFIER, TRACKER, STATE_STORE)


@mcp.tool(name="yolo_get_status")
def yolo_get_status(payload: StatusInput) -> dict[str, Any]:
    """拉取指定任务的训练进度（SSH 读 results.csv）、回写 MLflow 指标，并按配置推送飞书里程碑/终态。

    训练进行中可反复调用；返回含 ok、status、metrics 等。需使用启动时返回的 jobId 与 runId。
    """
    return get_status(
        payload.jobId,
        payload.runId,
        STATE_STORE,
        SSH,
        TRACKER,
        NOTIFIER,
        feishu_report_enable=SETTINGS.feishu_report_enable,
        feishu_report_every_n_epochs=SETTINGS.feishu_report_every_n_epochs,
        primary_metric_key=SETTINGS.primary_metric_key,
    )


@mcp.tool(name="yolo_stop_training")
def yolo_stop_training(payload: StopTrainingInput) -> dict[str, Any]:
    """请求停止远程训练进程，并结束对应 MLflow run（kill/标记以 stop_training 实现为准）。

    需提供 jobId 与 runId；成功返回更新后的任务状态或停止确认字段。
    """
    return stop_training(payload.jobId, payload.runId, SSH, NOTIFIER, TRACKER, STATE_STORE)


@mcp.tool(name="yolo_auto_tune")
def yolo_auto_tune(payload: AutoTuneInput) -> dict[str, Any]:
    """按 searchSpace 对 learningRate、batch、imgSize 做网格遍历
    （最多 maxTrials 个 trial），串行启动训练并轮询 get_status。

    返回最佳 trial、MLflow 顶部 run 摘要及 trials 列表；
    全部失败时返回 ok=false。耗时随 epochs 与 trial 数增长。
    """
    req = TuneRequest(
        env_id=payload.envId,
        base_job_id=payload.baseJobId,
        model=payload.model,
        data_config_path=payload.dataConfigPath,
        epochs=payload.epochs,
        search_space=payload.searchSpace,
        max_trials=payload.maxTrials,
        work_dir=SETTINGS.yolo_work_dir,
        jobs_dir=SETTINGS.yolo_jobs_dir,
        primary_metric_key=SETTINGS.primary_metric_key,
        feishu_report_enable=SETTINGS.feishu_report_enable,
        feishu_report_every_n_epochs=SETTINGS.feishu_report_every_n_epochs,
    )
    return auto_tune(req, SSH, NOTIFIER, TRACKER, STATE_STORE)


@mcp.tool(name="yolo_list_jobs")
def yolo_list_jobs(payload: ListJobsInput) -> dict[str, Any]:
    """列出本地状态存储中的最近训练任务摘要（含状态、epoch 提示等，具体字段见 list_jobs 实现）。

    用于在对话中快速查看有哪些 jobId 可继续 get_status 或 stop。
    """
    return list_jobs(STATE_STORE, SSH, limit=payload.limit)


@mcp.tool(name="yolo_get_job")
def yolo_get_job(payload: GetJobInput) -> dict[str, Any]:
    """按 jobId 查询单条任务记录；refresh=true 时会顺带刷新远程指标与状态（类似 get_status）。

    返回存储中的 Job 详情及可选刷新后的指标。
    """
    return get_job(
        payload.jobId,
        STATE_STORE,
        SSH,
        TRACKER,
        NOTIFIER,
        refresh=payload.refresh,
        feishu_report_enable=SETTINGS.feishu_report_enable,
        feishu_report_every_n_epochs=SETTINGS.feishu_report_every_n_epochs,
        primary_metric_key=SETTINGS.primary_metric_key,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
