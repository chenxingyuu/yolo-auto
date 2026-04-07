from __future__ import annotations

from yolo_auto.tools.mlflow_grouping import (
    build_training_group_tags,
    dataset_scope_key,
    mlflow_filter_same_training_scope,
    path_stem,
)


def test_path_stem_basename_and_relative() -> None:
    assert path_stem("/data/foo/coco.yaml") == "coco"
    assert path_stem("datasets/bar.yml") == "bar"
    assert path_stem("/workspace/models/yolo11n.pt") == "yolo11n"


def test_build_training_group_tags() -> None:
    tags = build_training_group_tags(
        job_id="j1",
        env_id="gpu-a",
        model_path="/m/yolov8n.pt",
        data_config_path="/d/x.yaml",
    )
    assert tags["yolo_job_id"] == "j1"
    assert tags["yolo_env_id"] == "gpu-a"
    assert tags["yolo_model_stem"] == "yolov8n"
    assert tags["yolo_data_stem"] == "x"
    assert tags["yolo_source"] == "yolo_auto.training"


def test_dataset_scope_key_prefers_parent_for_generic_data_yaml() -> None:
    assert dataset_scope_key("/workspace/datasets/helmet/data.yaml") == "helmet"
    assert dataset_scope_key("/workspace/datasets/task_7/dataset.yml") == "task_7"
    assert dataset_scope_key("/workspace/datasets/coco128.yaml") == "coco128"


def test_mlflow_filter_escapes_quotes() -> None:
    flt = mlflow_filter_same_training_scope(
        env_id="e'1",
        model_path="a.pt",
        data_config_path="/datasets/b/data.yaml",
    )
    assert "tags.yolo_env_id = 'e''1'" in flt
    assert "tags.yolo_data_stem = 'b'" in flt
    assert "tags.yolo_model_stem = 'a'" in flt
