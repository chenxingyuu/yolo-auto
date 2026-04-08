from __future__ import annotations

import os
from typing import Any


def mlflow_tags_for_dataset_provenance(
    provenance: dict[str, Any] | None,
) -> dict[str, str]:
    """将 JobRecord.dataset_provenance 转为 MLflow tags（截断过长值）。"""
    if not provenance:
        return {}
    tags: dict[str, str] = {}
    max_len = 500
    z = provenance.get("minioExportZip")
    if z is not None and str(z).strip():
        tags["yolo_minio_export_zip"] = str(z).strip()[:max_len]
    slug = provenance.get("datasetSlug")
    if slug is not None and str(slug).strip():
        tags["yolo_dataset_slug"] = str(slug).strip()[:max_len]
    note = provenance.get("datasetVersionNote")
    if note is not None and str(note).strip():
        tags["yolo_dataset_version_note"] = str(note).strip()[:max_len]
    return tags


def path_stem(path: str) -> str:
    """数据集 YAML 或权重路径的文件名去掉扩展名，用于跨 run 分组。"""
    cleaned = path.strip().replace("\\", "/")
    base = os.path.basename(cleaned) or cleaned
    stem, _ext = os.path.splitext(base)
    return stem or base


def dataset_scope_key(path: str) -> str:
    """为数据集分组生成更稳定的键。

    若文件名是通用名（如 data.yaml / dataset.yml），优先使用父目录名，
    避免不同数据集都被归到同一个 `data` 分组。
    """
    cleaned = path.strip().replace("\\", "/")
    stem = path_stem(cleaned).strip().lower()
    if stem not in {"data", "dataset"}:
        return path_stem(cleaned)

    parent = os.path.basename(os.path.dirname(cleaned)).strip()
    if parent:
        return parent
    return path_stem(cleaned)


def build_training_group_tags(
    *,
    job_id: str,
    env_id: str,
    model_path: str,
    data_config_path: str,
) -> dict[str, str]:
    """MLflow run tags：单实验内按数据/模型/环境等维度筛选。"""
    return {
        "yolo_job_id": job_id,
        "yolo_env_id": str(env_id).strip() or "default",
        "yolo_data_stem": dataset_scope_key(data_config_path),
        "yolo_model_stem": path_stem(model_path),
        "yolo_source": "yolo_auto.training",
    }


def _escape_filter_literal(value: str) -> str:
    return value.replace("'", "''")


def mlflow_filter_same_training_scope(
    *,
    env_id: str,
    model_path: str,
    data_config_path: str,
) -> str:
    """与当前训练/调参会话同环境、同数据与模型分组的 runs（用于 search_runs）。"""
    e = _escape_filter_literal(str(env_id).strip() or "default")
    ds = _escape_filter_literal(dataset_scope_key(data_config_path))
    ms = _escape_filter_literal(path_stem(model_path))
    return (
        f"tags.yolo_env_id = '{e}' and "
        f"tags.yolo_data_stem = '{ds}' and tags.yolo_model_stem = '{ms}'"
    )
