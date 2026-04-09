from __future__ import annotations

import csv
import os
import re
import shlex
import signal
import subprocess
import time
from io import StringIO
from threading import Lock
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"missing env: {name}")
    return value


def _token_guard(authorization: str | None = Header(default=None)) -> None:
    expected = (os.getenv("YOLO_CONTROL_BEARER_TOKEN") or "").strip()
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    provided = authorization[7:].strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="invalid bearer token")


def _format_yolo_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    return shlex.quote(str(value))


def _build_extra_cli_args(extra_args: dict[str, Any] | None) -> str:
    if not extra_args:
        return ""
    parts: list[str] = []
    for key, raw_value in extra_args.items():
        if raw_value is None:
            continue
        parts.append(f"{key}={_format_yolo_value(raw_value)}")
    return " ".join(parts)


class TrainStartRequest(BaseModel):
    jobId: str
    runName: str | None = None
    modelPath: str
    dataPath: str
    project: str
    name: str
    device: str | int | None = None
    epochs: int = Field(gt=0)
    imgsz: int = Field(gt=0)
    batch: int | float = Field(gt=0)
    extraArgs: dict[str, Any] | None = None
    workDir: str
    mlflowTrackingUri: str
    mlflowExperimentName: str


class StopRequest(BaseModel):
    jobId: str
    executionId: str | None = None


class ValidateRequest(BaseModel):
    jobId: str
    bestPath: str
    dataPath: str
    jobsDir: str
    workDir: str
    imgsz: int | None = None
    batch: int | None = None
    device: str | None = None
    extraArgs: dict[str, Any] | None = None
    skipPerImageQc: bool = False


_ALL_LINE_RE = re.compile(
    r"^\s*all\s+\d+\s+\d+\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s*$",
    re.MULTILINE,
)


def _parse_val_stdout(stdout: str) -> dict[str, float] | None:
    match = _ALL_LINE_RE.search(stdout)
    if not match:
        return None
    return {
        "precision": float(match.group(1)),
        "recall": float(match.group(2)),
        "map50": float(match.group(3)),
        "map5095": float(match.group(4)),
    }


app = FastAPI(title="yolo-control-api", version="1.0.0")
_job_pid: dict[str, int] = {}
_lock = Lock()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/train/start", dependencies=[Depends(_token_guard)])
def train_start(req: TrainStartRequest) -> dict[str, Any]:
    if not os.path.isfile(req.modelPath):
        raise HTTPException(status_code=400, detail=f"model not found: {req.modelPath}")
    os.makedirs(os.path.join(req.project, req.name), exist_ok=True)
    job_dir = os.path.join(req.project, req.name)
    log_path = os.path.join(job_dir, "train.log")
    extra = dict(req.extraArgs or {})
    if req.device is not None:
        extra["device"] = req.device
    extra_cli = _build_extra_cli_args(extra)
    cmd = (
        f"cd {shlex.quote(req.workDir)} && "
        f"export MLFLOW_TRACKING_URI={shlex.quote(req.mlflowTrackingUri)} && "
        f"export MLFLOW_EXPERIMENT_NAME={shlex.quote(req.mlflowExperimentName)} && "
        f"export MLFLOW_RUN_NAME={shlex.quote(req.runName or req.jobId)} && "
        f"yolo detect train model={shlex.quote(req.modelPath)} data={shlex.quote(req.dataPath)} "
        f"epochs={req.epochs} imgsz={req.imgsz} batch={req.batch} "
        f"project={shlex.quote(req.project)} "
        f"name={shlex.quote(req.name)} exist_ok=True {extra_cli} "
        f"> {shlex.quote(log_path)} 2>&1"
    )
    proc = subprocess.Popen(  # noqa: S603
        ["bash", "-lc", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    with _lock:
        _job_pid[req.jobId] = int(proc.pid)
    return {
        "jobId": req.jobId,
        "executionId": str(proc.pid),
        "pid": str(proc.pid),
        "status": "running",
        "paths": {
            "jobDir": job_dir,
            "logPath": log_path,
            "metricsPath": os.path.join(job_dir, "results.csv"),
            "bestPath": os.path.join(job_dir, "weights", "best.pt"),
            "lastPath": os.path.join(job_dir, "weights", "last.pt"),
            "modelPath": req.modelPath,
            "dataConfigPath": req.dataPath,
        },
    }


@app.post("/api/v1/train/stop", dependencies=[Depends(_token_guard)])
def train_stop(req: StopRequest) -> dict[str, Any]:
    with _lock:
        pid = _job_pid.get(req.jobId)
    if req.executionId and req.executionId.isdigit():
        pid = int(req.executionId)
    if not pid:
        return {"jobId": req.jobId, "status": "already_stopped"}
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"jobId": req.jobId, "status": "already_stopped"}
    return {"jobId": req.jobId, "status": "stopped", "executionId": str(pid)}


@app.get("/api/v1/train/status", dependencies=[Depends(_token_guard)])
def train_status(
    jobId: str = Query(...),
    pid: str | None = Query(default=None),
    metricsPath: str | None = Query(default=None),
    logPath: str | None = Query(default=None),
    totalEpochs: int | None = Query(default=None),
    createdAt: int | None = Query(default=None),
) -> dict[str, Any]:
    with _lock:
        known_pid = _job_pid.get(jobId)
    active_pid = int(pid) if pid and pid.isdigit() else known_pid
    process_alive = False
    if active_pid:
        try:
            os.kill(active_pid, 0)
            process_alive = True
        except OSError:
            process_alive = False

    metrics: dict[str, Any] = {}
    progress = 0.0
    if metricsPath and os.path.isfile(metricsPath):
        text = open(metricsPath, encoding="utf-8").read()
        rows = list(csv.DictReader(StringIO(text)))
        if rows:
            last = rows[-1]
            epoch = int(float(last.get("epoch", 0)))
            metrics = {
                "epoch": epoch,
                "loss": float(last.get("train/box_loss", 0) or 0)
                + float(last.get("train/cls_loss", 0) or 0)
                + float(last.get("train/dfl_loss", 0) or 0),
                "map50": float(last.get("metrics/mAP50(B)", 0) or 0),
                "map5095": float(last.get("metrics/mAP50-95(B)", 0) or 0),
                "precision": float(last.get("metrics/precision(B)", 0) or 0),
                "recall": float(last.get("metrics/recall(B)", 0) or 0),
            }
            if totalEpochs and totalEpochs > 0:
                progress = round(min(1.0, epoch / totalEpochs), 4)

    log_tail = ""
    if logPath and os.path.isfile(logPath):
        lines = open(logPath, encoding="utf-8", errors="replace").read().splitlines()
        log_tail = "\n".join(lines[-80:])

    status = "running" if process_alive else "completed"
    if not process_alive and ("traceback" in log_tail.lower() or "error" in log_tail.lower()):
        status = "failed"
    elapsed = 0
    if createdAt:
        elapsed = max(int(time.time()) - int(createdAt), 0)
    return {
        "jobId": jobId,
        "status": status,
        "processAlive": process_alive,
        "progress": progress,
        "metrics": metrics,
        "elapsedSeconds": elapsed,
        "logTail": log_tail,
    }


@app.post("/api/v1/validate/run", dependencies=[Depends(_token_guard)])
def validate_run(req: ValidateRequest) -> dict[str, Any]:
    if not os.path.isfile(req.bestPath):
        raise HTTPException(status_code=400, detail=f"best model not found: {req.bestPath}")
    cmd = [
        f"cd {shlex.quote(req.workDir)}",
        "&&",
        "yolo detect val",
        f"model={shlex.quote(req.bestPath)}",
        f"data={shlex.quote(req.dataPath)}",
        f"project={shlex.quote(req.jobsDir)}",
        f"name={shlex.quote(f'val-{req.jobId}')}",
    ]
    if req.imgsz is not None:
        cmd.append(f"imgsz={req.imgsz}")
    if req.batch is not None:
        cmd.append(f"batch={req.batch}")
    if req.device is not None:
        cmd.append(f"device={shlex.quote(req.device)}")
    extra_cli = _build_extra_cli_args(req.extraArgs)
    if extra_cli:
        cmd.append(extra_cli)
    full = " ".join(cmd)
    proc = subprocess.run(  # noqa: S603
        ["bash", "-lc", full],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
    )
    if proc.returncode != 0:
        raise HTTPException(status_code=400, detail=proc.stderr.strip() or proc.stdout.strip())
    metrics = _parse_val_stdout(proc.stdout)
    if not metrics:
        raise HTTPException(status_code=400, detail="unable to parse yolo val stdout")
    qc_path = os.path.join(req.jobsDir.rstrip("/"), f"val-{req.jobId}", "per_image_qc.jsonl")
    return {
        "jobId": req.jobId,
        "modelPath": req.bestPath,
        "metrics": metrics,
        "rawOutput": proc.stdout,
        "qcArtifactPath": "" if req.skipPerImageQc else qc_path,
        "qcPreviewLines": [],
        "qcLineCount": None,
        "qcSkipped": req.skipPerImageQc,
    }


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(_token_guard)])
def get_job(job_id: str) -> dict[str, Any]:
    with _lock:
        pid = _job_pid.get(job_id)
    return {"jobId": job_id, "executionId": str(pid) if pid else None}


@app.get("/api/v1/jobs", dependencies=[Depends(_token_guard)])
def list_jobs(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    with _lock:
        items = list(_job_pid.items())[:limit]
    return {
        "jobs": [{"jobId": job_id, "executionId": str(pid)} for job_id, pid in items],
        "count": len(items),
    }


def main() -> None:
    host = os.getenv("YOLO_CONTROL_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = int((os.getenv("YOLO_CONTROL_PORT", "18080") or "18080").strip())
    uvicorn.run("yolo_auto.control_api:app", host=host, port=port, reload=False)

