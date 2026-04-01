from __future__ import annotations

import json
import shlex
from pathlib import PurePosixPath
from typing import Any

from yolo_auto.errors import err, ok
from yolo_auto.ssh_client import SSHClient


def _resolve_remote_path(path: str, work_dir: str) -> str:
    if path.startswith("/"):
        return path
    return f"{work_dir.rstrip('/')}/{path.lstrip('./')}"


def _parse_dataset_summary(
    ssh_client: SSHClient, *, data_config_abs_path: str, work_dir: str
) -> tuple[dict[str, Any] | None, str]:
    py_snippet = (
        "import json\n"
        "from pathlib import Path\n"
        "import yaml\n"
        "\n"
        f"cfg_path = Path({data_config_abs_path!r})\n"
        "data = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))\n"
        "if not isinstance(data, dict):\n"
        "    raise ValueError('dataset yaml root must be a mapping')\n"
        "\n"
        "base_dir = cfg_path.parent\n"
        f"work_dir = Path({work_dir!r})\n"
        "\n"
        "def resolve_path(raw):\n"
        "    if raw is None:\n"
        "        return None\n"
        "    text = str(raw).strip()\n"
        "    if not text:\n"
        "        return None\n"
        "    p = Path(text)\n"
        "    if p.is_absolute():\n"
        "        return p\n"
        "    return (base_dir / p).resolve()\n"
        "\n"
        "def path_exists(raw):\n"
        "    p = resolve_path(raw)\n"
        "    if p is None:\n"
        "        return {'exists': False, 'path': None}\n"
        "    return {'exists': p.exists(), 'path': str(p)}\n"
        "\n"
        "names = data.get('names')\n"
        "nc = data.get('nc')\n"
        "if isinstance(names, dict):\n"
        "    names_count = len(names)\n"
        "elif isinstance(names, list):\n"
        "    names_count = len(names)\n"
        "else:\n"
        "    names_count = None\n"
        "\n"
        "out = {\n"
        "    'train': path_exists(data.get('train')),\n"
        "    'val': path_exists(data.get('val')),\n"
        "    'test': path_exists(data.get('test')),\n"
        "    'hasTrain': bool(data.get('train')),\n"
        "    'hasVal': bool(data.get('val')),\n"
        "    'hasTest': bool(data.get('test')),\n"
        "    'hasNames': isinstance(names, (dict, list)),\n"
        "    'namesCount': names_count,\n"
        "    'nc': nc,\n"
        "    'yamlPath': str(cfg_path.resolve()),\n"
        "    'workDir': str(work_dir),\n"
        "}\n"
        "print(json.dumps(out, ensure_ascii=True))\n"
    )
    cmd = f"python -c {shlex.quote(py_snippet)}"
    stdout_text, stderr_text, exit_code = ssh_client.execute(cmd)
    if exit_code != 0:
        return None, stderr_text.strip() or "failed to parse dataset yaml"
    try:
        return json.loads(stdout_text.strip()), ""
    except Exception:
        return None, "invalid dataset summary output"


def _resolve_model_path(
    ssh_client: SSHClient,
    *,
    model: str,
    work_dir: str,
    models_dir: str,
) -> tuple[str | None, bool, list[str], str | None]:
    model_abs_path = _resolve_remote_path(model, work_dir)
    model_q = shlex.quote(model_abs_path)
    _, _, exists_code = ssh_client.execute(f"test -f {model_q}")
    if exists_code == 0:
        return model_abs_path, False, [model_abs_path], None

    basename = PurePosixPath(model).name
    if not basename:
        return None, False, [], "MODEL_NOT_FOUND"
    search_cmd = (
        f"find {shlex.quote(models_dir)} -type f -name {shlex.quote(basename)} -print | head -100"
    )
    out, _, search_code = ssh_client.execute(search_cmd)
    if search_code != 0:
        return None, False, [], "MODEL_NOT_FOUND"
    candidates = [line.strip() for line in out.splitlines() if line.strip()]
    if len(candidates) == 1:
        return candidates[0], True, candidates, None
    if len(candidates) > 1:
        return None, False, candidates, "MODEL_AMBIGUOUS"
    return None, False, [], "MODEL_NOT_FOUND"


def setup_env(
    ssh_client: SSHClient,
    work_dir: str,
    data_config_path: str,
    model: str,
    models_dir: str = "/workspace/models",
) -> dict[str, object]:
    version_cmd = "python -c 'import ultralytics; print(ultralytics.__version__)'"
    stdout_text, stderr_text, exit_code = ssh_client.execute(version_cmd)
    if exit_code != 0:
        return err(
            error_code="ENV_UNREACHABLE",
            message=stderr_text.strip() or "ultralytics not available",
            retryable=True,
            hint="检查 SSH、Python 环境与 ultralytics 安装",
            payload={"reachable": False},
        )

    work_dir_q = shlex.quote(work_dir)
    data_config_abs_path = _resolve_remote_path(data_config_path, work_dir)
    model_abs_path, model_auto_fixed, model_candidates, model_err = _resolve_model_path(
        ssh_client,
        model=model,
        work_dir=work_dir,
        models_dir=models_dir,
    )
    if model_err == "MODEL_AMBIGUOUS":
        return err(
            error_code="MODEL_AMBIGUOUS",
            message="multiple model candidates found in models directory",
            retryable=False,
            hint="请传入更精确的模型路径，或在候选中选择一个",
            payload={
                "reachable": True,
                "workDir": work_dir,
                "yoloVersion": stdout_text.strip(),
                "validModel": False,
                "modelPath": _resolve_remote_path(model, work_dir),
                "searchedDir": models_dir,
                "candidates": model_candidates,
            },
        )
    if model_abs_path is None:
        return err(
            error_code="MODEL_NOT_FOUND",
            message=f"model not found: {_resolve_remote_path(model, work_dir)}",
            retryable=False,
            hint="确认 model 路径存在，或将同名模型放到 YOLO_MODELS_DIR",
            payload={
                "reachable": True,
                "workDir": work_dir,
                "yoloVersion": stdout_text.strip(),
                "validModel": False,
                "modelPath": _resolve_remote_path(model, work_dir),
                "searchedDir": models_dir,
                "candidates": model_candidates,
            },
        )
    data_q = shlex.quote(data_config_abs_path)
    model_q = shlex.quote(model_abs_path)
    check_cmd = f"test -d {work_dir_q} && test -f {data_q} && test -f {model_q}"
    _, check_err, check_code = ssh_client.execute(check_cmd)
    if check_code != 0:
        return err(
            error_code="DATA_CONFIG_INVALID",
            message=check_err.strip() or "workDir or dataConfigPath missing",
            retryable=False,
            hint="确认 dataConfigPath 与工作目录在远程容器内存在",
            payload={
                "reachable": True,
                "workDir": work_dir,
                "yoloVersion": stdout_text.strip(),
                "validData": False,
                "validModel": True,
                "dataConfigPath": data_config_abs_path,
                "modelPath": model_abs_path,
                "resolvedModelPath": model_abs_path,
                "modelAutoFixed": model_auto_fixed,
            },
        )

    dataset_summary, parse_err = _parse_dataset_summary(
        ssh_client,
        data_config_abs_path=data_config_abs_path,
        work_dir=work_dir,
    )
    if dataset_summary is None:
        return err(
            error_code="DATASET_YAML_INVALID",
            message=parse_err,
            retryable=False,
            hint="确认 dataConfigPath 为合法 YAML，且远程 Python 环境可导入 yaml",
            payload={
                "reachable": True,
                "workDir": work_dir,
                "yoloVersion": stdout_text.strip(),
                "validModel": True,
                "modelPath": model_abs_path,
                "resolvedModelPath": model_abs_path,
                "modelAutoFixed": model_auto_fixed,
                "validData": False,
                "dataConfigPath": data_config_abs_path,
            },
        )

    if not dataset_summary["hasTrain"] or not dataset_summary["hasVal"]:
        return err(
            error_code="DATASET_SPLIT_INVALID",
            message="dataset yaml must include non-empty train and val",
            retryable=False,
            hint="请在 YAML 中配置 train 与 val，test 可选",
            payload=dataset_summary,
        )
    if not dataset_summary["train"]["exists"] or not dataset_summary["val"]["exists"]:
        return err(
            error_code="DATASET_PATH_NOT_FOUND",
            message="train/val path does not exist",
            retryable=False,
            hint="请确认 train/val 路径存在；相对路径会按 YAML 所在目录解析",
            payload=dataset_summary,
        )
    if not dataset_summary["hasNames"]:
        return err(
            error_code="DATASET_CLASSES_INVALID",
            message="dataset yaml missing names",
            retryable=False,
            hint="请提供 names（list 或 dict）以声明类别名称",
            payload=dataset_summary,
        )
    names_count = dataset_summary["namesCount"]
    nc_value = dataset_summary["nc"]
    if (
        isinstance(names_count, int)
        and isinstance(nc_value, int)
        and names_count != nc_value
    ):
        return err(
            error_code="DATASET_CLASSES_MISMATCH",
            message=f"nc ({nc_value}) does not match names count ({names_count})",
            retryable=False,
            hint="请保持 nc 与 names 数量一致，或仅保留 names",
            payload=dataset_summary,
        )

    warnings: list[str] = []
    if not dataset_summary["hasTest"]:
        warnings.append("dataset yaml 未配置 test（允许为空）")
    elif not dataset_summary["test"]["exists"]:
        warnings.append("dataset yaml 配置了 test，但路径不存在")

    return ok(
        {
            "reachable": True,
            "workDir": work_dir,
            "yoloVersion": stdout_text.strip(),
            "validData": True,
            "validModel": True,
            "modelPath": model_abs_path,
            "resolvedModelPath": model_abs_path,
            "modelAutoFixed": model_auto_fixed,
            "modelCandidates": model_candidates,
            "dataConfigPath": data_config_abs_path,
            "datasetChecks": dataset_summary,
            "warnings": warnings,
        }
    )

