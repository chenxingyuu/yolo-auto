from __future__ import annotations

import base64
import re
import shlex
from typing import Any

from yolo_auto.errors import err, ok
from yolo_auto.models import JobStatus
from yolo_auto.remote_control import HttpControlClient, RemoteControlError
from yolo_auto.state_store import JobStateStore

_ALL_LINE_RE = re.compile(
    r"^\s*all\s+\d+\s+\d+\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s*$",
    re.MULTILINE,
)


def _format_yolo_value(value: Any) -> str:
    # Ultralytics CLI 参数使用 key=value 形式，字符串需要 quote，数字/布尔可直接输出。
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        list_repr = "[" + ",".join(_format_yolo_value(item) for item in value) + "]"
        return shlex.quote(list_repr)
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


def _parse_val_stdout(stdout: str) -> dict[str, float] | None:
    match = _ALL_LINE_RE.search(stdout)
    if not match:
        return None
    precision = float(match.group(1))
    recall = float(match.group(2))
    map50 = float(match.group(3))
    map5095 = float(match.group(4))
    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map5095": map5095,
    }


# 远程执行：按验证集逐图 predict，写出 jsonl（依赖远端已安装 ultralytics & PyYAML）。
_REMOTE_PER_IMAGE_QC_SCRIPT = r"""import json
import os
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def _count_gt_boxes(label_path: Path) -> int:
    if not label_path.is_file():
        return 0
    text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return 0
    return len([ln for ln in text.splitlines() if ln.strip()])


def _label_dir_for_val(root: Path, val_rel: str) -> Path:
    val_rel_path = Path(val_rel)
    parts = val_rel_path.parts
    if parts and parts[0] == "images" and len(parts) >= 2:
        return root.joinpath("labels", *parts[1:])
    if "images" in parts:
        idx = parts.index("images")
        suffix = parts[idx + 1 :]
        return root.joinpath("labels", *suffix)
    return root / "labels" / val_rel_path.name


def _label_path_for_image(img_path: Path) -> Path:
    posix = img_path.resolve().as_posix()
    if "/images/" in posix:
        return Path(posix.replace("/images/", "/labels/", 1)).with_suffix(".txt")
    p = img_path.resolve()
    return p.parent / f"{p.stem}.txt"


def _paths_from_val_txt(list_file: Path, root: Path, yaml_dir: Path) -> list[Path]:
    images: list[Path] = []
    for line in list_file.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cand = Path(s)
        if cand.is_absolute():
            ip = cand.resolve()
        else:
            ip = (root / s).resolve()
            if not ip.is_file():
                ip = (yaml_dir / s).resolve()
            if not ip.is_file():
                ip = (list_file.parent / s).resolve()
        if not ip.is_file():
            print(f"missing image (val list): {s!r} -> {ip}", file=sys.stderr)
            sys.exit(5)
        images.append(ip)
    return images


def main() -> None:
    best = sys.argv[1]
    data_yaml = Path(sys.argv[2]).resolve()
    out_path = Path(sys.argv[3]).resolve()
    yaml_dir = data_yaml.parent

    imgsz_raw = os.environ.get("YOLO_QC_IMGSZ", "").strip()
    imgsz = int(imgsz_raw) if imgsz_raw else None
    device = os.environ.get("YOLO_QC_DEVICE", "").strip() or None

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (yaml_dir / root).resolve()

    val_rel = cfg.get("val")
    if not val_rel:
        print("dataset yaml missing 'val' key", file=sys.stderr)
        sys.exit(2)

    val_ref = Path(str(val_rel))
    val_path = val_ref.resolve() if val_ref.is_absolute() else (root / val_ref).resolve()

    predict_kw: dict[str, object] = {"verbose": False, "save": False}
    if imgsz is not None:
        predict_kw["imgsz"] = imgsz
    if device is not None:
        predict_kw["device"] = device

    model = YOLO(best)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    val_dir_for_rel: Path | None = None

    if val_path.is_file() and val_path.suffix.lower() == ".txt":
        images = _paths_from_val_txt(val_path, root, yaml_dir)
    elif val_path.is_dir():
        val_dir_for_rel = val_path
        images = sorted(p for p in val_path.rglob("*") if p.suffix.lower() in exts)
    else:
        print(f"val must be a directory or .txt image list, got: {val_path}", file=sys.stderr)
        sys.exit(3)

    if not images:
        print(f"no images resolved for val: {val_path}", file=sys.stderr)
        sys.exit(4)

    root_res = root.resolve()
    label_dir_fallback = _label_dir_for_val(root, str(val_rel))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_iter = model.predict(source=[str(p) for p in images], stream=True, **predict_kw)

    with out_path.open("w", encoding="utf-8") as fh:
        for img_path, res in zip(images, results_iter, strict=True):
            ip = img_path.resolve()
            if val_dir_for_rel is not None:
                try:
                    rel_img = str(ip.relative_to(val_dir_for_rel.resolve()))
                except ValueError:
                    rel_img = ip.name
            else:
                try:
                    rel_img = str(ip.relative_to(root_res))
                except ValueError:
                    rel_img = ip.name
            stem = ip.stem
            label_path = _label_path_for_image(ip)
            if not label_path.is_file():
                label_path = (label_dir_fallback / f"{stem}.txt").resolve()
            gt_boxes = _count_gt_boxes(label_path)

            boxes = res.boxes
            n_pred = 0
            max_conf = 0.0
            mean_conf = 0.0
            if boxes is not None and len(boxes) > 0:
                n_pred = int(len(boxes))
                confs = boxes.conf.cpu().tolist()
                max_conf = float(max(confs))
                mean_conf = float(sum(confs) / len(confs))

            row = {
                "image": str(ip),
                "relImage": rel_img,
                "stem": stem,
                "gtBoxes": gt_boxes,
                "predBoxes": n_pred,
                "maxConf": round(max_conf, 6),
                "meanConf": round(mean_conf, 6),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
"""


def _run_remote_per_image_qc(
    legacy_client: object,
    *,
    best_path: str,
    data_config_path: str,
    qc_out_path: str,
    img_size: int | None,
    device: str | None,
) -> tuple[str | None, str | None]:
    """Returns (stderr_or_none, error_message_or_none)."""
    b64 = base64.b64encode(_REMOTE_PER_IMAGE_QC_SCRIPT.encode("utf-8")).decode("ascii")
    script_invocation = (
        f"echo {b64} | base64 -d | python3 - "
        f"{shlex.quote(best_path)} {shlex.quote(data_config_path)} {shlex.quote(qc_out_path)}"
    )
    exports: list[str] = []
    if img_size is not None:
        exports.append(f"export YOLO_QC_IMGSZ={int(img_size)}")
    if device:
        exports.append(f"export YOLO_QC_DEVICE={shlex.quote(device)}")
    prefix = " && ".join(exports) + " && " if exports else ""
    bash_script = prefix + script_invocation
    full_cmd = "bash -lc " + shlex.quote(bash_script)
    _stdout, stderr_text, exit_code = legacy_client.execute(full_cmd, timeout=3600)
    if exit_code != 0:
        return stderr_text.strip() or _stdout.strip(), "per_image_qc script failed on remote host"
    return None, None


def _qc_preview_and_line_count(
    legacy_client: object, qc_path: str
) -> tuple[list[str], int | None]:
    _ = legacy_client
    _ = qc_path
    return [], None


def run_validation(
    job_id: str,
    state_store: JobStateStore,
    legacy_client: object | None,
    jobs_dir: str,
    work_dir: str,
    *,
    data_config_path: str | None = None,
    img_size: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    extra_args: dict[str, Any] | None = None,
    skip_per_image_qc: bool = False,
    control_client: HttpControlClient | None = None,
) -> dict[str, object]:
    effective_job_id = (job_id or "").strip()
    if not effective_job_id:
        return err(
            error_code="MISSING_JOB_ID",
            message="jobId 不能为空",
            retryable=False,
            hint="请传入训练任务 jobId",
            payload={},
        )

    record = state_store.get(effective_job_id)
    if not record:
        return err(
            error_code="JOB_NOT_FOUND",
            message=f"job not found: {effective_job_id}",
            retryable=False,
            hint="请先启动训练或检查 jobId",
            payload={"jobId": effective_job_id},
        )

    if record.status != JobStatus.COMPLETED:
        return err(
            error_code="JOB_NOT_COMPLETED",
            message=f"job not completed: {effective_job_id}",
            retryable=False,
            hint="请先等待训练完成后再调用 yolo_validate",
            payload=record.to_dict(),
        )

    best_path = record.paths.get("bestPath", "")
    if not best_path:
        return err(
            error_code="MISSING_BEST_MODEL",
            message="bestPath missing from job record",
            retryable=False,
            hint="请确认训练过程中 best 权重已生成，或检查 JobStateStore 的 paths 字段",
            payload=record.to_dict(),
        )

    _ = legacy_client

    effective_data_config = data_config_path or record.paths.get("dataConfigPath", "")
    if not effective_data_config:
        return err(
            error_code="MISSING_DATA_CONFIG",
            message="dataConfigPath missing for validation",
            retryable=False,
            hint="请传入 dataConfigPath 参数，或在训练后确保 job record 中写入 dataConfigPath",
            payload={"jobId": effective_job_id},
        )

    if control_client is not None:
        try:
            remote = control_client.run_validation(
                {
                    "jobId": effective_job_id,
                    "bestPath": best_path,
                    "dataPath": effective_data_config,
                    "jobsDir": jobs_dir,
                    "workDir": work_dir,
                    "imgsz": img_size,
                    "batch": batch,
                    "device": device,
                    "extraArgs": extra_args,
                    "skipPerImageQc": skip_per_image_qc,
                }
            )
        except RemoteControlError as exc:
            return err(
                error_code=exc.error_code,
                message=str(exc),
                retryable=exc.retryable,
                hint=exc.hint,
                payload={"jobId": effective_job_id, **exc.payload},
            )
        return ok(dict(remote))

    return err(
        error_code="REMOTE_CLIENT_MISSING",
        message="control client is required in HTTP-only mode",
        retryable=False,
        hint="请检查 YOLO_CONTROL_BASE_URL 配置",
        payload={"jobId": effective_job_id},
    )
