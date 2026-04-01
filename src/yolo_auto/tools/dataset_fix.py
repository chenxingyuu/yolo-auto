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


def fix_dataset(
    ssh_client: SSHClient,
    *,
    work_dir: str,
    data_config_path: str,
    dry_run: bool = True,
    apply: bool = False,
    val_ratio: float = 0.2,
    max_fix_items: int = 5000,
) -> dict[str, Any]:
    if dry_run and apply:
        return err(
            error_code="INVALID_FIX_MODE",
            message="dryRun and apply cannot both be true",
            retryable=False,
            hint="请二选一：dryRun=true 仅预览；apply=true 执行修复",
        )

    data_config_abs_path = _resolve_remote_path(data_config_path, work_dir)
    py_snippet = f"""
import json
import random
import time
from pathlib import Path
import yaml

cfg_path = Path({data_config_abs_path!r}).resolve()
dry_run = {bool(dry_run)}
apply_mode = {bool(apply)}
val_ratio = float({float(val_ratio)})
max_fix_items = int({int(max_fix_items)})

if val_ratio <= 0 or val_ratio >= 1:
    raise ValueError("valRatio must be between 0 and 1")

data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
if not isinstance(data, dict):
    raise ValueError("dataset yaml root must be a mapping")

base_dir = cfg_path.parent
root = Path(str(data.get("path", ".")).strip() or ".")
if not root.is_absolute():
    root = (base_dir / root).resolve()

planned_changes = []
applied_changes = []
risk_items = []
backup_paths = []

def add_change(kind, file_path, detail):
    item = {{"kind": kind, "file": str(file_path), "detail": detail}}
    planned_changes.append(item)
    return item

def write_with_backup(path: Path, text: str):
    if not apply_mode:
        return
    ts = int(time.time())
    if path.exists():
        bak = path.with_name(path.name + f".bak.{{ts}}")
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        backup_paths.append(str(bak))
    path.write_text(text, encoding="utf-8")

def normalize_split_file(split_value):
    if split_value is None or str(split_value).strip() == "":
        return None, [], 0
    p = Path(str(split_value).strip())
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    if not p.exists():
        return p, [], 0
    lines = []
    fixed_prefix = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        ip = Path(line)
        if not ip.is_absolute():
            ip = (root / ip).resolve()
            if (not ip.exists()) and line.startswith("data/"):
                trimmed = line[len("data/") :]
                candidate = (root / Path(trimmed)).resolve()
                if candidate.exists():
                    ip = candidate
                    fixed_prefix += 1
        lines.append(str(ip))
    uniq = list(dict.fromkeys(lines))
    removed = len(lines) - len(uniq)
    return p, uniq, removed, fixed_prefix

split_results = {{}}
for split in ("train", "val", "test"):
    raw = data.get(split)
    if raw is None or str(raw).strip() == "":
        split_results[split] = {{"path": None, "lines": [], "removedDuplicates": 0}}
        continue
    p, lines, removed, fixed_prefix = normalize_split_file(raw)
    split_results[split] = {{"path": p, "lines": lines, "removedDuplicates": removed}}
    if removed > 0:
        add_change("split_deduplicate", p, f"removed duplicates: {{removed}}")
    if fixed_prefix > 0:
        add_change(
            "normalize_split_data_prefix",
            p,
            f"fixed data/ prefix paths: {{fixed_prefix}}",
        )

raw_path = str(data.get("path", ".")).strip() if data.get("path") is not None else "."
if raw_path in ("", ".", "./"):
    normalized_root = str(cfg_path.parent.resolve())
    if data.get("path") != normalized_root:
        data["path"] = normalized_root
        add_change(
            "normalize_dataset_root_path",
            cfg_path,
            "set path to absolute dataset root",
        )

if not data.get("val"):
    train_lines = split_results["train"]["lines"]
    if train_lines:
        random.seed(42)
        shuffled = list(train_lines)
        random.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * val_ratio))
        val_lines = shuffled[:val_count]
        train_new = shuffled[val_count:]
        val_path = cfg_path.parent / "val.txt"
        split_results["val"]["path"] = val_path
        split_results["val"]["lines"] = val_lines
        split_results["train"]["lines"] = train_new
        data["val"] = "val.txt"
        add_change("generate_val_split", val_path, f"generated {{val_count}} samples from train")
    else:
        risk_items.append("missing val and train split is empty; cannot generate val.txt")

names = data.get("names")
nc = data.get("nc")
if isinstance(names, dict):
    names_count = len(names)
elif isinstance(names, list):
    names_count = len(names)
else:
    names_count = None

if isinstance(names_count, int):
    if nc != names_count:
        data["nc"] = names_count
        add_change("sync_nc", cfg_path, "set nc=" + str(names_count) + " from names")
else:
    risk_items.append("names is missing or invalid; cannot infer class count for safe auto-fix")

fixed_rows = 0
skipped_risky_rows = 0
class_count = names_count if isinstance(names_count, int) else (nc if isinstance(nc, int) else None)
all_images = (
    split_results["train"]["lines"]
    + split_results["val"]["lines"]
    + split_results["test"]["lines"]
)
all_images = list(dict.fromkeys(all_images))[:max_fix_items]

def image_to_label(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")

label_file_cache = {{}}
for img in all_images:
    ip = Path(img)
    lp = image_to_label(ip)
    if not lp.exists():
        continue
    lines = lp.read_text(encoding="utf-8").splitlines()
    changed = False
    new_lines = []
    for row in lines:
        s = row.strip()
        if not s:
            new_lines.append("")
            continue
        cols = s.split()
        if len(cols) < 5:
            new_lines.append(row)
            skipped_risky_rows += 1
            continue
        if len(cols) > 5:
            cols = cols[:5]
            changed = True
            fixed_rows += 1
        try:
            class_id = int(float(cols[0]))
            x = float(cols[1]); y = float(cols[2]); w = float(cols[3]); h = float(cols[4])
        except Exception:
            new_lines.append(row)
            skipped_risky_rows += 1
            continue
        if class_count is not None and (class_id < 0 or class_id >= class_count):
            new_lines.append(row)
            skipped_risky_rows += 1
            continue
        normalized = f"{{class_id}} {{x:.6f}} {{y:.6f}} {{w:.6f}} {{h:.6f}}"
        if normalized != row:
            changed = True
            fixed_rows += 1
        new_lines.append(normalized)
    if changed:
        label_file_cache[lp] = "\\n".join(new_lines) + "\\n"
        add_change("normalize_label_rows", lp, "normalized label rows")

if len(planned_changes) > max_fix_items:
    planned_changes = planned_changes[:max_fix_items]
    risk_items.append("planned changes truncated by maxFixItems")

if apply_mode:
    for split in ("train", "val", "test"):
        split_path = split_results[split]["path"]
        if split_path is None:
            continue
        lines = split_results[split]["lines"]
        rel_lines = []
        for line in lines:
            p = Path(line)
            try:
                rel = p.relative_to(root)
                rel_text = str(rel)
                if rel_text in ("", "."):
                    rel_lines.append("./")
                elif rel_text.startswith("./"):
                    rel_lines.append(rel_text)
                else:
                    rel_lines.append(f"./{{rel_text}}")
            except Exception:
                rel_lines.append(str(p))
        text = "\\n".join(rel_lines) + ("\\n" if rel_lines else "")
        write_with_backup(split_path, text)
        applied_changes.append(
            {{
                "kind": "write_split",
                "file": str(split_path),
                "count": len(rel_lines),
            }}
        )

    yaml_text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    write_with_backup(cfg_path, yaml_text)
    applied_changes.append({{"kind": "write_yaml", "file": str(cfg_path)}})

    for p, text in label_file_cache.items():
        write_with_backup(p, text)
        applied_changes.append({{"kind": "write_label", "file": str(p)}})

result = {{
    "dataConfigPath": str(cfg_path),
    "datasetRoot": str(root),
    "dryRun": dry_run,
    "apply": apply_mode,
    "plannedChanges": planned_changes,
    "appliedChanges": applied_changes,
    "backupPaths": backup_paths,
    "riskItems": risk_items,
    "estimatedImpact": {{
        "plannedChangeCount": len(planned_changes),
        "fixedLabelRows": fixed_rows,
        "skippedRiskyRows": skipped_risky_rows,
    }},
}}
print(json.dumps(result, ensure_ascii=True))
"""
    cmd = f"python -c {shlex.quote(py_snippet)}"
    stdout_text, stderr_text, exit_code = ssh_client.execute(cmd, timeout=1800)
    if exit_code != 0:
        return err(
            error_code="DATASET_FIX_EXEC_FAILED",
            message=stderr_text.strip() or "dataset fix execution failed",
            retryable=True,
            hint="检查远程 Python/PyYAML 环境、数据集路径与写入权限",
            payload={"dataConfigPath": data_config_abs_path},
        )
    try:
        payload = json.loads(stdout_text.strip())
    except Exception:
        return err(
            error_code="DATASET_FIX_PARSE_FAILED",
            message="invalid dataset fix output",
            retryable=False,
            hint="请重试，或检查远程输出是否被额外日志污染",
            payload={"raw": stdout_text[:500]},
        )
    return ok(payload)

