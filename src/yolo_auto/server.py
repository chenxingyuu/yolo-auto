from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from yolo_auto.config import load_settings
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.ssh_client import SSHClient, SSHConfig
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
NOTIFIER = FeishuNotifier(SETTINGS.feishu_webhook_url)
TRACKER = MLflowTracker(
    TrackerConfig(
        tracking_uri=SETTINGS.mlflow_tracking_uri,
        experiment_name=SETTINGS.mlflow_experiment_name,
    )
)


class StartTrainingInput(BaseModel):
    model: str
    dataConfigPath: str
    epochs: int = Field(gt=0)
    imgSize: int = Field(gt=0)
    batch: int = Field(gt=0)
    learningRate: float = Field(gt=0)
    jobId: str | None = None


class StopTrainingInput(BaseModel):
    jobId: str
    runId: str


class StatusInput(BaseModel):
    jobId: str
    runId: str


class AutoTuneInput(BaseModel):
    envId: str
    baseJobId: str
    model: str
    dataConfigPath: str
    epochs: int = Field(gt=0)
    maxTrials: int = Field(gt=0)
    searchSpace: dict[str, list[float | int]]


@mcp.tool(name="yolo.setup_env")
def yolo_setup_env(dataConfigPath: str) -> dict[str, Any]:
    return setup_env(SSH, SETTINGS.yolo_work_dir, dataConfigPath)


@mcp.tool(name="yolo.start_training")
def yolo_start_training(payload: StartTrainingInput) -> dict[str, Any]:
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
    )
    return start_training(req, SSH, NOTIFIER, TRACKER)


@mcp.tool(name="yolo.get_status")
def yolo_get_status(payload: StatusInput) -> dict[str, Any]:
    return get_status(payload.jobId, payload.runId, SETTINGS.yolo_jobs_dir, SSH, TRACKER)


@mcp.tool(name="yolo.stop_training")
def yolo_stop_training(payload: StopTrainingInput) -> dict[str, Any]:
    return stop_training(payload.jobId, payload.runId, SSH, NOTIFIER, TRACKER)


@mcp.tool(name="yolo.auto_tune")
def yolo_auto_tune(payload: AutoTuneInput) -> dict[str, Any]:
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
    )
    return auto_tune(req, SSH, NOTIFIER, TRACKER)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

