from __future__ import annotations

import csv
import itertools
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
import yaml
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


class SetupEnvRequest(BaseModel):
    model: str
    dataConfigPath: str
    workDir: str = "/workspace"
    modelsDir: str = "/workspace/models"


class DatasetCheckRequest(BaseModel):
    dataConfigPath: str
    workDir: str = "/workspace"


class DatasetFixRequest(BaseModel):
    dataConfigPath: str
    dryRun: bool = True
    apply: bool = False


class DatasetSyncRequest(BaseModel):
    filename: str
    datasetName: str
    minioAlias: str = "minio"
    minioBucket: str = "cvat-export"
    minioPrefix: str = "exports"
    datasetsDir: str = "/workspace/datasets"


class ExportRequest(BaseModel):
    bestPath: str
    jobsDir: str
    workDir: str
    jobId: str
    formats: list[str] | None = None
    imgSize: int | None = None
    half: bool | None = None
    int8: bool | None = None
    device: str | None = None
    extraArgs: dict[str, Any] | None = None


class SahiSliceRequest(BaseModel):
    dataConfigPath: str
    outputDatasetName: str
    outputDatasetsDir: str = "/workspace/datasets"
    sliceHeight: int = Field(default=640, gt=0)
    sliceWidth: int = Field(default=640, gt=0)
    overlapHeightRatio: float = Field(default=0.2, ge=0.0, lt=1.0)
    overlapWidthRatio: float = Field(default=0.2, ge=0.0, lt=1.0)
    minAreaRatio: float = Field(default=0.1, ge=0.0, le=1.0)


class AutoTuneRequest(BaseModel):
    baseJobId: str
    model: str
    dataConfigPath: str
    epochs: int
    maxTrials: int
    searchSpace: dict[str, list[float | int]]
    workDir: str
    jobsDir: str
    trialTimeoutSeconds: int = 1800
    pollIntervalSeconds: int = 10


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


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


app = FastAPI(title="yolo-control-api", version="1.0.0")
_job_pid: dict[str, int] = {}
_lock = Lock()


def _run_bash(cmd: str, *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["bash", "-lc", cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


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
    training_rows: list[dict[str, Any]] = []
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
            # 采样历史行，严格上限 100 条，避免响应过大；末尾始终是最新一行
            _max_rows = 100
            if len(rows) <= _max_rows:
                training_rows = [dict(r) for r in rows]
            else:
                step = max(1, len(rows) // _max_rows)
                sampled = [dict(rows[i]) for i in range(0, len(rows), step)][:_max_rows]
                last_dict = dict(rows[-1])
                if sampled[-1] != last_dict:
                    sampled[-1] = last_dict  # 替换末尾，而非追加
                training_rows = sampled

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
        "trainingRows": training_rows,
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


@app.post("/api/v1/env/setup", dependencies=[Depends(_token_guard)])
def env_setup(req: SetupEnvRequest) -> dict[str, Any]:
    model_path = req.model if req.model.startswith("/") else os.path.join(req.workDir, req.model)
    data_path = (
        req.dataConfigPath
        if req.dataConfigPath.startswith("/")
        else os.path.join(req.workDir, req.dataConfigPath)
    )
    return {
        "workDirExists": os.path.isdir(req.workDir),
        "modelsDirExists": os.path.isdir(req.modelsDir),
        "modelExists": os.path.isfile(model_path),
        "dataConfigExists": os.path.isfile(data_path),
        "resolved": {"modelPath": model_path, "dataConfigPath": data_path},
    }


@app.post("/api/v1/dataset/check", dependencies=[Depends(_token_guard)])
def dataset_check(req: DatasetCheckRequest) -> dict[str, Any]:
    data_path = (
        req.dataConfigPath
        if req.dataConfigPath.startswith("/")
        else os.path.join(req.workDir, req.dataConfigPath)
    )
    if not os.path.isfile(data_path):
        raise HTTPException(status_code=400, detail=f"dataset yaml not found: {data_path}")
    with open(data_path, encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    base = cfg.get("path", os.path.dirname(data_path))
    if not str(base).startswith("/"):
        base = os.path.normpath(os.path.join(os.path.dirname(data_path), str(base)))
    train_rel = str(cfg.get("train", "")).strip()
    val_rel = str(cfg.get("val", "")).strip()
    train_dir = train_rel if train_rel.startswith("/") else os.path.join(base, train_rel)
    val_dir = val_rel if val_rel.startswith("/") else os.path.join(base, val_rel)
    return {
        "ok": bool(train_rel and val_rel and os.path.exists(train_dir) and os.path.exists(val_dir)),
        "datasetRoot": base,
        "trainPath": train_dir,
        "valPath": val_dir,
        "trainExists": os.path.exists(train_dir),
        "valExists": os.path.exists(val_dir),
    }


@app.post("/api/v1/dataset/fix", dependencies=[Depends(_token_guard)])
def dataset_fix(req: DatasetFixRequest) -> dict[str, Any]:
    if req.apply and req.dryRun:
        raise HTTPException(status_code=400, detail="apply=true requires dryRun=false")
    return {
        "ok": True,
        "dryRun": req.dryRun,
        "apply": req.apply,
        "message": "HTTP 控制面当前提供保守 no-op 修复；建议先人工核验后再训练",
        "plannedChanges": [],
    }


@app.post("/api/v1/dataset/sync", dependencies=[Depends(_token_guard)])
def dataset_sync(req: DatasetSyncRequest) -> dict[str, Any]:
    src = "/".join(
        s.strip("/") for s in (req.minioAlias, req.minioBucket, req.minioPrefix, req.filename) if s
    )
    target_dir = os.path.join(req.datasetsDir, req.datasetName)
    os.makedirs(target_dir, exist_ok=True)
    tmp_zip = os.path.join("/tmp", f"{req.datasetName}.zip")
    cp_cmd = f"mc cp {shlex.quote(src)} {shlex.quote(tmp_zip)}"
    cp_res = _run_bash(cp_cmd, timeout=600)
    if cp_res.returncode != 0:
        raise HTTPException(status_code=400, detail=cp_res.stderr.strip() or cp_res.stdout.strip())
    unzip_cmd = f"unzip -o {shlex.quote(tmp_zip)} -d {shlex.quote(target_dir)}"
    unzip_res = _run_bash(unzip_cmd, timeout=600)
    if unzip_res.returncode != 0:
        raise HTTPException(
            status_code=400, detail=unzip_res.stderr.strip() or unzip_res.stdout.strip()
        )
    data_yaml = ""
    for root, _dirs, files in os.walk(target_dir):
        for f in files:
            if f.endswith(".yaml") or f.endswith(".yml"):
                data_yaml = os.path.join(root, f)
                break
        if data_yaml:
            break
    return {
        "ok": True,
        "datasetDir": target_dir,
        "dataConfigPath": data_yaml,
        "provenance": {"objectName": req.filename},
    }


@app.post("/api/v1/export/run", dependencies=[Depends(_token_guard)])
def export_run(req: ExportRequest) -> dict[str, Any]:
    if not os.path.isfile(req.bestPath):
        raise HTTPException(status_code=400, detail=f"best model not found: {req.bestPath}")
    formats = req.formats or ["onnx", "engine", "coreml"]
    artifacts: list[dict[str, str]] = []
    for fmt in formats:
        parts = [
            f"cd {shlex.quote(req.workDir)}",
            "&&",
            "yolo export",
            f"model={shlex.quote(req.bestPath)}",
            f"format={shlex.quote(fmt)}",
        ]
        if req.imgSize is not None:
            parts.append(f"imgsz={req.imgSize}")
        if req.half is not None:
            parts.append(f"half={_format_yolo_value(req.half)}")
        if req.int8 is not None:
            parts.append(f"int8={_format_yolo_value(req.int8)}")
        if req.device is not None:
            parts.append(f"device={shlex.quote(req.device)}")
        extra_cli = _build_extra_cli_args(req.extraArgs)
        if extra_cli:
            parts.append(extra_cli)
        proc = _run_bash(" ".join(parts), timeout=1800)
        if proc.returncode != 0:
            raise HTTPException(status_code=400, detail=proc.stderr.strip() or proc.stdout.strip())
    job_dir = os.path.join(req.jobsDir, req.jobId)
    for root, _dirs, files in os.walk(job_dir):
        for f in files:
            if any(f.endswith(ext) for ext in (".onnx", ".engine", ".mlmodel", ".pt")):
                artifacts.append({"path": os.path.join(root, f)})
    return {"ok": True, "jobId": req.jobId, "artifacts": artifacts}


def _slice_windows(
    img_h: int, img_w: int, sh: int, sw: int, oh: float, ow: float
) -> list[tuple[int, int, int, int]]:
    """Generate (x1, y1, x2, y2) sliding windows; edge tiles are shifted back
    to maintain full sh×sw size (SAHI-aligned behaviour)."""
    step_h = max(1, int(sh * (1 - oh)))
    step_w = max(1, int(sw * (1 - ow)))
    windows: list[tuple[int, int, int, int]] = []
    y = 0
    while True:
        y1 = min(y, max(0, img_h - sh))   # 回退，确保 y1+sh <= img_h
        y2 = y1 + sh
        x = 0
        while True:
            x1 = min(x, max(0, img_w - sw))  # 回退，确保 x1+sw <= img_w
            x2 = x1 + sw
            windows.append((x1, y1, x2, y2))
            if x + sw >= img_w:
                break
            x += step_w
        if y + sh >= img_h:
            break
        y += step_h
    return windows


def _remap_yolo_bboxes(
    labels: list[str],
    img_w: int,
    img_h: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    min_area_ratio: float,
) -> list[str]:
    """Clip and remap YOLO-format labels from a full image into a slice."""
    slice_w = x2 - x1
    slice_h = y2 - y1
    result: list[str] = []
    for line in labels:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = parts[0]
        cx_n, cy_n, w_n, h_n = map(float, parts[1:])
        cx = cx_n * img_w
        cy = cy_n * img_h
        bw = w_n * img_w
        bh = h_n * img_h
        bx1, by1 = cx - bw / 2, cy - bh / 2
        bx2, by2 = cx + bw / 2, cy + bh / 2
        ix1 = max(bx1, x1)
        iy1 = max(by1, y1)
        ix2 = min(bx2, x2)
        iy2 = min(by2, y2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        orig_area = bw * bh
        if orig_area <= 0:
            continue
        if (ix2 - ix1) * (iy2 - iy1) / orig_area < min_area_ratio:
            continue
        new_cx = max(0.0, min(1.0, ((ix1 + ix2) / 2 - x1) / slice_w))
        new_cy = max(0.0, min(1.0, ((iy1 + iy2) / 2 - y1) / slice_h))
        new_w = max(0.0, min(1.0, (ix2 - ix1) / slice_w))
        new_h = max(0.0, min(1.0, (iy2 - iy1) / slice_h))
        result.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")
    return result


def _find_label_path(img_path: str) -> str:
    if "/images/" in img_path:
        label_p = img_path.replace("/images/", "/labels/", 1)
    else:
        label_p = img_path
    base, _ = os.path.splitext(label_p)
    return base + ".txt"


def _split_ref_anchor(split_ref: str, dataset_root: str) -> str:
    """Directory or list file path (same anchor as split image collection)."""
    return split_ref if split_ref.startswith("/") else os.path.join(dataset_root, split_ref)


def _sahi_output_subdir_and_stem(
    img_path: str,
    *,
    split: str,
    split_ref: str,
    dataset_root: str,
) -> tuple[str, str]:
    """Relative subdirectory under images/<split> (posix) and file stem; mirrors source layout."""
    p = _split_ref_anchor(split_ref, dataset_root)
    img_norm = os.path.normpath(img_path)
    ds_norm = os.path.normpath(dataset_root)

    if os.path.isdir(p):
        anchor = os.path.normpath(p)
        try:
            rel = os.path.relpath(img_norm, anchor)
        except ValueError:
            rel = os.path.basename(img_norm)
        if rel.startswith(".."):
            rel = os.path.basename(img_norm)
        rel_dir, name = os.path.split(rel)
        stem = os.path.splitext(name)[0]
        return rel_dir.replace("\\", "/"), stem

    try:
        rel = os.path.relpath(img_norm, ds_norm)
    except ValueError:
        return "", os.path.splitext(os.path.basename(img_norm))[0]
    if rel.startswith(".."):
        return "", os.path.splitext(os.path.basename(img_norm))[0]
    norm_rel = rel.replace("\\", "/")
    for pref in (f"images/{split}/", f"./images/{split}/"):
        if norm_rel.startswith(pref):
            tail = norm_rel[len(pref) :]
            rel_dir, name = os.path.split(tail)
            return rel_dir, os.path.splitext(name)[0]
    name = os.path.basename(norm_rel)
    return os.path.dirname(norm_rel).replace("\\", "/"), os.path.splitext(name)[0]


def _collect_split_images(split_ref: str, dataset_root: str) -> list[str]:
    images: list[str] = []
    if not split_ref:
        return images
    p = _split_ref_anchor(split_ref, dataset_root)
    img_exts = {"jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"}
    if os.path.isdir(p):
        for root, _, files in os.walk(p):
            for f in files:
                if f.lower().rsplit(".", 1)[-1] in img_exts:
                    images.append(os.path.join(root, f))
    elif os.path.isfile(p) and p.lower().endswith(".txt"):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    img = line if line.startswith("/") else os.path.join(dataset_root, line)
                    images.append(img)
    return sorted(images)


@app.post("/api/v1/dataset/sahi-slice", dependencies=[Depends(_token_guard)])
def dataset_sahi_slice(req: SahiSliceRequest) -> dict[str, Any]:
    from PIL import Image  # pillow is bundled with ultralytics

    if not os.path.isfile(req.dataConfigPath):
        raise HTTPException(status_code=400, detail=f"data config not found: {req.dataConfigPath}")

    with open(req.dataConfigPath, encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}

    yaml_dir = os.path.dirname(req.dataConfigPath)
    dataset_root = str(cfg.get("path", yaml_dir))
    if not dataset_root.startswith("/"):
        dataset_root = os.path.normpath(os.path.join(yaml_dir, dataset_root))

    out_dir = os.path.join(req.outputDatasetsDir, req.outputDatasetName)
    total_source = 0
    total_slices = 0
    new_cfg: dict[str, Any] = {
        "path": out_dir,
        "nc": cfg.get("nc"),
        "names": cfg.get("names"),
    }

    for split in ("train", "val", "test"):
        split_ref = str(cfg.get(split, "")).strip()
        if not split_ref:
            continue
        images = _collect_split_images(split_ref, dataset_root)
        if not images:
            new_cfg[split] = f"images/{split}"
            continue

        out_img_dir = os.path.join(out_dir, "images", split)
        out_lbl_dir = os.path.join(out_dir, "labels", split)

        for img_path in images:
            if not os.path.isfile(img_path):
                continue
            total_source += 1
            rel_subdir, stem = _sahi_output_subdir_and_stem(
                img_path,
                split=split,
                split_ref=split_ref,
                dataset_root=dataset_root,
            )
            slice_img_base = os.path.join(out_img_dir, rel_subdir) if rel_subdir else out_img_dir
            slice_lbl_base = os.path.join(out_lbl_dir, rel_subdir) if rel_subdir else out_lbl_dir
            os.makedirs(slice_img_base, exist_ok=True)
            os.makedirs(slice_lbl_base, exist_ok=True)

            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size

            label_path = _find_label_path(img_path)
            labels: list[str] = []
            if os.path.isfile(label_path):
                with open(label_path, encoding="utf-8") as lf:
                    labels = [ln for ln in lf.read().splitlines() if ln.strip()]

            windows = _slice_windows(
                img_h, img_w,
                req.sliceHeight, req.sliceWidth,
                req.overlapHeightRatio, req.overlapWidthRatio,
            )
            for idx, (wx1, wy1, wx2, wy2) in enumerate(windows):
                slice_img = img.crop((wx1, wy1, wx2, wy2))
                out_stem = f"{stem}_{idx:04d}"
                slice_img.save(
                    os.path.join(slice_img_base, f"{out_stem}.jpg"), "JPEG", quality=95
                )
                remapped = _remap_yolo_bboxes(
                    labels, img_w, img_h, wx1, wy1, wx2, wy2, req.minAreaRatio
                )
                label_out_path = os.path.join(slice_lbl_base, f"{out_stem}.txt")
                with open(label_out_path, "w", encoding="utf-8") as lf:
                    lf.write("\n".join(remapped) + ("\n" if remapped else ""))
                total_slices += 1

        new_cfg[split] = f"images/{split}"

    out_yaml = os.path.join(out_dir, "data.yaml")
    with open(out_yaml, "w", encoding="utf-8") as fp:
        yaml.dump(new_cfg, fp, allow_unicode=True, default_flow_style=False)

    avg = round(total_slices / total_source, 2) if total_source > 0 else 0.0
    return {
        "ok": True,
        "dataConfigPath": out_yaml,
        "stats": {
            "sourceImages": total_source,
            "totalSlices": total_slices,
            "avgSlicesPerImage": avg,
        },
    }


@app.post("/api/v1/tune/auto", dependencies=[Depends(_token_guard)])
def tune_auto(req: AutoTuneRequest) -> dict[str, Any]:
    lrs = req.searchSpace.get("learningRate", [0.01])
    batches = req.searchSpace.get("batch", [16])
    imgs = req.searchSpace.get("imgSize", [640])
    grid = list(itertools.product(lrs, batches, imgs))[: req.maxTrials]
    trials: list[dict[str, Any]] = []
    for idx, (lr, bs, img) in enumerate(grid, start=1):
        job_id = f"{req.baseJobId}-t{idx}"
        start_resp = train_start(
            TrainStartRequest(
                jobId=job_id,
                runName=job_id,
                modelPath=req.model,
                dataPath=req.dataConfigPath,
                project=req.jobsDir,
                name=job_id,
                device=0,
                epochs=req.epochs,
                imgsz=int(img),
                batch=float(bs),
                extraArgs={"lr0": float(lr)},
                workDir=req.workDir,
                mlflowTrackingUri=os.getenv("MLFLOW_TRACKING_URI", ""),
                mlflowExperimentName=os.getenv("MLFLOW_EXPERIMENT_NAME", "yolo-auto"),
            )
        )
        pid = str(start_resp.get("pid", ""))
        deadline = time.time() + req.trialTimeoutSeconds
        latest = {"status": "running", "metrics": {}}
        while time.time() <= deadline:
            latest = train_status(
                jobId=job_id,
                pid=pid,
                metricsPath=start_resp["paths"]["metricsPath"],
                logPath=start_resp["paths"]["logPath"],
                totalEpochs=req.epochs,
                createdAt=int(time.time()),
            )
            if latest["status"] in {"completed", "failed", "stopped"}:
                break
            time.sleep(req.pollIntervalSeconds)
        metric = float((latest.get("metrics") or {}).get("map5095") or 0.0)
        trials.append(
            {
                "jobId": job_id,
                "runId": job_id,
                "params": {"learningRate": lr, "batch": bs, "imgSize": img},
                "metric": metric,
                "status": latest.get("status", "failed"),
                "error": None,
            }
        )
    succeeded = [t for t in trials if t["status"] == "completed"]
    if not succeeded:
        raise HTTPException(status_code=400, detail="all tuning trials failed")
    best = max(succeeded, key=lambda x: float(x["metric"]))
    return {
        "envId": "default",
        "baseJobId": req.baseJobId,
        "bestJobId": best["jobId"],
        "bestMetrics": {"map5095": best["metric"]},
        "bestParams": best["params"],
        "trials": trials,
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


@app.get("/api/v1/datasets", dependencies=[Depends(_token_guard)])
def list_datasets(datasetsDir: str = Query(default="/workspace/datasets")) -> dict[str, Any]:
    files: list[str] = []
    root = datasetsDir
    if os.path.isdir(root):
        for current_root, _dirs, names in os.walk(root):
            for name in names:
                lower = name.lower()
                if lower.endswith(".yaml") or lower.endswith(".yml"):
                    abs_path = os.path.join(current_root, name)
                    files.append(os.path.relpath(abs_path, root))
    files.sort()
    return {"datasetsDir": root, "files": files, "count": len(files)}


@app.get("/api/v1/models", dependencies=[Depends(_token_guard)])
def list_models(modelsDir: str = Query(default="/workspace/models")) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    root = modelsDir
    if os.path.isdir(root):
        for current_root, _dirs, names in os.walk(root):
            for name in names:
                if not name.lower().endswith(".pt"):
                    continue
                abs_path = os.path.join(current_root, name)
                try:
                    size = os.path.getsize(abs_path)
                except OSError:
                    size = None
                items.append(
                    {
                        "path": abs_path,
                        "relativePath": os.path.relpath(abs_path, root),
                        "name": name,
                        "sizeBytes": size,
                    }
                )
    items.sort(key=lambda x: str(x.get("relativePath", "")))
    return {"modelsDir": root, "models": items, "count": len(items)}


@app.get("/api/v1/minio/datasets", dependencies=[Depends(_token_guard)])
def list_minio_datasets(source: str = Query(..., min_length=1)) -> dict[str, Any]:
    cmd = f"mc ls --recursive {shlex.quote(source)}"
    proc = _run_bash(cmd, timeout=120)
    if proc.returncode != 0:
        raise HTTPException(status_code=400, detail=proc.stderr.strip() or proc.stdout.strip())
    items: list[dict[str, Any]] = []
    for line in (proc.stdout or "").splitlines():
        row = line.strip()
        if not row:
            continue
        parts = row.split(maxsplit=3)
        if len(parts) < 4:
            continue
        path = parts[3]
        if not path.lower().endswith(".zip"):
            continue
        items.append(
            {
                "path": path,
                "size": parts[2],
                "modifiedAt": f"{parts[0]} {parts[1]}",
            }
        )
    return {"source": source, "files": items, "count": len(items)}


@app.get("/api/v1/env/gpu", dependencies=[Depends(_token_guard)])
def env_gpu() -> dict[str, Any]:
    cmd = (
        "nvidia-smi --query-gpu="
        "name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu "
        "--format=csv,noheader,nounits"
    )
    proc = _run_bash(cmd, timeout=30)
    if proc.returncode != 0:
        return {"gpus": [], "count": 0, "warning": proc.stderr.strip() or proc.stdout.strip()}
    gpus: list[dict[str, Any]] = []
    for idx, line in enumerate((proc.stdout or "").splitlines()):
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 6:
            continue
        gpus.append(
            {
                "index": idx,
                "name": cols[0],
                "memoryTotalMb": _safe_int(cols[1]),
                "memoryUsedMb": _safe_int(cols[2]),
                "memoryFreeMb": _safe_int(cols[3]),
                "utilizationGpuPercent": _safe_float(cols[4]),
                "temperatureC": _safe_float(cols[5]),
            }
        )
    return {"gpus": gpus, "count": len(gpus)}


@app.get("/api/v1/env/system", dependencies=[Depends(_token_guard)])
def env_system() -> dict[str, Any]:
    cpu_proc = _run_bash("nproc", timeout=10)
    mem_proc = _run_bash("free -m", timeout=10)
    disk_proc = _run_bash("df -h /workspace", timeout=10)
    info: dict[str, Any] = {}
    if cpu_proc.returncode == 0:
        info["cpuCores"] = _safe_int((cpu_proc.stdout or "").strip())
    if mem_proc.returncode == 0:
        lines = (mem_proc.stdout or "").splitlines()
        if len(lines) >= 2:
            cols = lines[1].split()
            if len(cols) >= 4:
                info["memoryMb"] = {
                    "total": _safe_int(cols[1]),
                    "used": _safe_int(cols[2]),
                    "free": _safe_int(cols[3]),
                }
    if disk_proc.returncode == 0:
        lines = (disk_proc.stdout or "").splitlines()
        if len(lines) >= 2:
            cols = lines[1].split()
            if len(cols) >= 6:
                info["workspaceDisk"] = {
                    "filesystem": cols[0],
                    "size": cols[1],
                    "used": cols[2],
                    "available": cols[3],
                    "usePercent": cols[4],
                    "mountpoint": cols[5],
                }
    return info


def main() -> None:
    host = os.getenv("YOLO_CONTROL_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = int((os.getenv("YOLO_CONTROL_PORT", "18080") or "18080").strip())
    uvicorn.run("yolo_auto.control_api:app", host=host, port=port, reload=False)

