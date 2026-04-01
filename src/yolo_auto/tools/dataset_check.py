from __future__ import annotations

import json
import shlex
from typing import Any

from yolo_auto.errors import err, ok
from yolo_auto.ssh_client import SSHClient


def _resolve_remote_path(path: str, work_dir: str) -> str:
    if path.startswith("/"):
        return path
    return f"{work_dir.rstrip('/')}/{path.lstrip('./')}"


def check_dataset(
    ssh_client: SSHClient,
    *,
    work_dir: str,
    data_config_path: str,
    max_errors: int = 200,
) -> dict[str, Any]:
    data_config_abs_path = _resolve_remote_path(data_config_path, work_dir)
    py_snippet = f"""
import json
from pathlib import Path
import yaml

MAX_ERRORS = {int(max_errors)}
cfg_path = Path({data_config_abs_path!r})

data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
if not isinstance(data, dict):
    raise ValueError("dataset yaml root must be a mapping")

base_dir = cfg_path.parent
root = Path(str(data.get("path", ".")).strip() or ".")
if not root.is_absolute():
    root = (base_dir / root).resolve()

names = data.get("names")
nc = data.get("nc")
if isinstance(names, dict):
    class_count = len(names)
elif isinstance(names, list):
    class_count = len(names)
elif isinstance(nc, int):
    class_count = nc
else:
    class_count = None

errors = []
stats = {{
    "totalSamples": 0,
    "missingImages": 0,
    "missingLabels": 0,
    "invalidLabelRows": 0,
    "classOutOfRange": 0,
}}
splits = {{}}

def add_error(code, message, split=None, imagePath=None, labelPath=None, lineNo=None):
    record = {{"code": code, "message": message}}
    if split is not None:
        record["split"] = split
    if imagePath is not None:
        record["imagePath"] = str(imagePath)
    if labelPath is not None:
        record["labelPath"] = str(labelPath)
    if lineNo is not None:
        record["lineNo"] = int(lineNo)
    if len(errors) < MAX_ERRORS:
        errors.append(record)

def resolve_entry_path(raw):
    text = str(raw).strip()
    if not text:
        return None
    p = Path(text)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()

def split_entries(value):
    p = resolve_entry_path(value)
    if p is None:
        return None, []
    if p.is_file():
        lines = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            pp = Path(line)
            if not pp.is_absolute():
                pp = (root / pp).resolve()
            lines.append(pp)
        return str(p), lines
    if p.is_dir():
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            imgs.extend(sorted(p.rglob(ext)))
        return str(p), imgs
    return str(p), []

def image_to_label(image_path):
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")

for split in ("train", "val", "test"):
    raw = data.get(split)
    if split in ("train", "val") and (raw is None or str(raw).strip() == ""):
        add_error("MISSING_SPLIT", f"dataset yaml missing required split: {{split}}", split=split)
        splits[split] = {{"source": None, "count": 0}}
        continue
    if raw is None or str(raw).strip() == "":
        splits[split] = {{"source": None, "count": 0}}
        continue
    source, images = split_entries(raw)
    splits[split] = {{"source": source, "count": len(images)}}
    if split in ("train", "val") and len(images) == 0:
        add_error("EMPTY_SPLIT", f"required split has no samples: {{split}}", split=split)
    for image_path in images:
        stats["totalSamples"] += 1
        if not image_path.exists():
            stats["missingImages"] += 1
            add_error("MISSING_IMAGE", "image not found", split=split, imagePath=image_path)
            continue
        label_path = image_to_label(image_path)
        if not label_path.exists():
            stats["missingLabels"] += 1
            add_error(
                "MISSING_LABEL",
                "label file not found for image",
                split=split,
                imagePath=image_path,
                labelPath=label_path,
            )
            continue
        for ln, row in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            row = row.strip()
            if not row:
                continue
            cols = row.split()
            if len(cols) != 5:
                stats["invalidLabelRows"] += 1
                add_error(
                    "INVALID_LABEL_ROW",
                    "label row must have 5 columns: class x y w h",
                    split=split,
                    imagePath=image_path,
                    labelPath=label_path,
                    lineNo=ln,
                )
                continue
            try:
                class_id = int(float(cols[0]))
                float(cols[1]); float(cols[2]); float(cols[3]); float(cols[4])
            except Exception:
                stats["invalidLabelRows"] += 1
                add_error(
                    "INVALID_LABEL_ROW",
                    "label row contains non-numeric values",
                    split=split,
                    imagePath=image_path,
                    labelPath=label_path,
                    lineNo=ln,
                )
                continue
            if class_count is not None and (class_id < 0 or class_id >= class_count):
                stats["classOutOfRange"] += 1
                add_error(
                    "CLASS_OUT_OF_RANGE",
                    f"class id out of range: {{class_id}} not in [0, {{class_count-1}}]",
                    split=split,
                    imagePath=image_path,
                    labelPath=label_path,
                    lineNo=ln,
                )

result = {{
    "dataConfigPath": str(cfg_path),
    "datasetRoot": str(root),
    "classCount": class_count,
    "stats": stats,
    "splits": splits,
    "errors": errors,
    "hasErrors": len(errors) > 0,
}}
print(json.dumps(result, ensure_ascii=True))
"""

    cmd = f"python -c {shlex.quote(py_snippet)}"
    stdout_text, stderr_text, exit_code = ssh_client.execute(cmd, timeout=1800)
    if exit_code != 0:
        return err(
            error_code="DATASET_CHECK_EXEC_FAILED",
            message=stderr_text.strip() or "dataset check execution failed",
            retryable=True,
            hint="检查远程 Python 环境、PyYAML 依赖和数据集路径",
            payload={"dataConfigPath": data_config_abs_path},
        )
    try:
        payload = json.loads(stdout_text.strip())
    except Exception:
        return err(
            error_code="DATASET_CHECK_PARSE_FAILED",
            message="invalid dataset check output",
            retryable=False,
            hint="请重试，或检查远程输出是否被额外日志污染",
            payload={"raw": stdout_text[:500]},
        )

    if payload.get("hasErrors"):
        return err(
            error_code="DATASET_CHECK_FAILED",
            message="dataset check failed with strict policy",
            retryable=False,
            hint="修复 errors 列表中的问题后再启动训练",
            payload=payload,
        )
    return ok(payload)

