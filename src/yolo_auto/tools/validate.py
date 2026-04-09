from __future__ import annotations

import base64
import re
import shlex
from typing import Any

from yolo_auto.errors import err, ok
from yolo_auto.models import JobStatus
from yolo_auto.ssh_client import SSHClient
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
    ssh_client: SSHClient,
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
    _stdout, stderr_text, exit_code = ssh_client.execute(full_cmd, timeout=3600)
    if exit_code != 0:
        return stderr_text.strip() or _stdout.strip(), "per_image_qc script failed on remote host"
    return None, None


def _qc_preview_and_line_count(
    ssh_client: SSHClient, qc_path: str
) -> tuple[list[str], int | None]:
    head_inner = f"head -n 5 {shlex.quote(qc_path)}"
    head_cmd = f"bash -lc {shlex.quote(head_inner)}"
    head_out, _, head_code = ssh_client.execute(head_cmd, timeout=60)
    lines = [ln for ln in head_out.splitlines() if ln.strip()] if head_code == 0 else []
    wc_inner = f"wc -l < {shlex.quote(qc_path)}"
    wc_cmd = f"bash -lc {shlex.quote(wc_inner)}"
    wc_out, _, wc_code = ssh_client.execute(wc_cmd, timeout=60)
    line_count: int | None = None
    if wc_code == 0 and wc_out.strip().isdigit():
        line_count = int(wc_out.strip())
    return lines, line_count


def run_validation(
    job_id: str,
    state_store: JobStateStore,
    ssh_client: SSHClient,
    jobs_dir: str,
    work_dir: str,
    *,
    data_config_path: str | None = None,
    img_size: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    extra_args: dict[str, Any] | None = None,
    skip_per_image_qc: bool = False,
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

    if not ssh_client.file_exists(best_path):
        return err(
            error_code="BEST_MODEL_NOT_FOUND",
            message="best model file not found on remote host",
            retryable=False,
            hint="请确认远程权重路径可访问，并且权限允许读取",
            payload={"jobId": effective_job_id, "modelPath": best_path},
        )

    effective_data_config = data_config_path or record.paths.get("dataConfigPath", "")
    if not effective_data_config:
        return err(
            error_code="MISSING_DATA_CONFIG",
            message="dataConfigPath missing for validation",
            retryable=False,
            hint="请传入 dataConfigPath 参数，或在训练后确保 job record 中写入 dataConfigPath",
            payload={"jobId": effective_job_id},
        )

    extra_cli_args = _build_extra_cli_args(extra_args)
    cmd = [
        f"cd {shlex.quote(work_dir)}",
        "&&",
        "yolo detect val",
        f"model={shlex.quote(best_path)}",
        f"data={shlex.quote(effective_data_config)}",
        f"project={shlex.quote(jobs_dir)}",
        f"name={shlex.quote(f'val-{effective_job_id}')}",
    ]
    if img_size is not None:
        cmd.append(f"imgsz={img_size}")
    if batch is not None:
        cmd.append(f"batch={batch}")
    if device is not None:
        cmd.append(f"device={shlex.quote(device)}")
    if extra_cli_args:
        cmd.append(extra_cli_args)

    val_cmd = " ".join(cmd)
    stdout_text, stderr_text, exit_code = ssh_client.execute(val_cmd, timeout=300)
    if exit_code != 0:
        return err(
            error_code="VALIDATION_FAILED",
            message=stderr_text.strip() or stdout_text.strip() or "yolo val failed",
            retryable=False,
            hint="检查远程路径、数据集 YAML 与权重文件是否可用",
            payload={"jobId": effective_job_id},
        )

    metrics = _parse_val_stdout(stdout_text)
    if not metrics:
        return err(
            error_code="VALIDATION_PARSE_FAILED",
            message="unable to parse yolo val stdout",
            retryable=False,
            hint="检查远程输出格式是否与 Ultralytics 版本一致（all 行可能不同）",
            payload={"jobId": effective_job_id},
        )

    qc_path = f"{jobs_dir.rstrip('/')}/val-{effective_job_id}/per_image_qc.jsonl"
    qc_preview_lines: list[str] = []
    qc_line_count: int | None = None
    qc_skipped = skip_per_image_qc

    if not skip_per_image_qc:
        qcerr, qmsg = _run_remote_per_image_qc(
            ssh_client,
            best_path=best_path,
            data_config_path=effective_data_config,
            qc_out_path=qc_path,
            img_size=img_size,
            device=device,
        )
        if qmsg is not None:
            return err(
                error_code="PER_IMAGE_QC_FAILED",
                message=qmsg,
                retryable=True,
                hint=(
                    "确认远端 python3、ultralytics、PyYAML 可用；"
                    "或 skipPerImageQc=true 仅跑聚合 val"
                ),
                payload={
                    "jobId": effective_job_id,
                    "metrics": metrics,
                    "rawOutput": stdout_text,
                    "qcArtifactPath": qc_path,
                    "qcStderr": qcerr or "",
                },
            )
        qc_preview_lines, qc_line_count = _qc_preview_and_line_count(ssh_client, qc_path)

    out: dict[str, object] = {
        "jobId": effective_job_id,
        "modelPath": best_path,
        "metrics": metrics,
        "rawOutput": stdout_text,
        "qcArtifactPath": "" if skip_per_image_qc else qc_path,
        "qcPreviewLines": qc_preview_lines,
        "qcLineCount": qc_line_count,
        "qcSkipped": qc_skipped,
    }
    return ok(out)
