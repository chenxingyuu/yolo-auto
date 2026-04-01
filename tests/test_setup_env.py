from __future__ import annotations

import json
from unittest.mock import MagicMock

from yolo_auto.tools.setup_env import setup_env


def test_setup_env_success(mock_ssh: MagicMock) -> None:
    summary = {
        "train": {"exists": True, "path": "/data/images/train"},
        "val": {"exists": True, "path": "/data/images/val"},
        "test": {"exists": False, "path": "/data/images/test"},
        "hasTrain": True,
        "hasVal": True,
        "hasTest": False,
        "hasNames": True,
        "namesCount": 2,
        "nc": 2,
        "yamlPath": "/data/dataset.yaml",
        "workDir": "/workspace/yolo-auto",
    }
    mock_ssh.execute.side_effect = [
        ("8.1.0", "", 0),
        ("", "", 0),
        ("", "", 0),
        (json.dumps(summary), "", 0),
    ]

    result = setup_env(
        mock_ssh,
        "/workspace/yolo-auto",
        "/data/dataset.yaml",
        "/workspace/models/yolo.pt",
    )
    assert result["ok"] is True
    assert result["reachable"] is True
    assert result["validData"] is True
    assert result["validModel"] is True
    assert result["modelAutoFixed"] is False
    assert result["yoloVersion"] == "8.1.0"
    assert result["warnings"] == ["dataset yaml 未配置 test（允许为空）"]


def test_setup_env_unreachable(mock_ssh: MagicMock) -> None:
    mock_ssh.execute.side_effect = [
        ("", "ultralytics not available", 1),
    ]

    result = setup_env(
        mock_ssh,
        "/workspace/yolo-auto",
        "/data/dataset.yaml",
        "/workspace/models/yolo.pt",
    )
    assert result["ok"] is False
    assert result["errorCode"] == "ENV_UNREACHABLE"


def test_setup_env_invalid_data(mock_ssh: MagicMock) -> None:
    mock_ssh.execute.side_effect = [
        ("8.1.0", "", 0),
        ("", "", 0),
        ("", "workDir or dataConfigPath missing", 1),
    ]

    result = setup_env(
        mock_ssh,
        "/workspace/yolo-auto",
        "/data/dataset.yaml",
        "/workspace/models/yolo.pt",
    )
    assert result["ok"] is False
    assert result["errorCode"] == "DATA_CONFIG_INVALID"
    assert result["reachable"] is True
    assert result["validData"] is False


def test_setup_env_model_not_found(mock_ssh: MagicMock) -> None:
    mock_ssh.execute.side_effect = [
        ("8.1.0", "", 0),
        ("", "missing model", 1),
        ("", "", 0),
    ]
    result = setup_env(
        mock_ssh,
        "/workspace/yolo-auto",
        "/data/dataset.yaml",
        "/workspace/models/missing.pt",
    )
    assert result["ok"] is False
    assert result["errorCode"] == "MODEL_NOT_FOUND"


def test_setup_env_model_auto_fixed_by_models_dir(mock_ssh: MagicMock) -> None:
    summary = {
        "train": {"exists": True, "path": "/data/images/train"},
        "val": {"exists": True, "path": "/data/images/val"},
        "test": {"exists": False, "path": "/data/images/test"},
        "hasTrain": True,
        "hasVal": True,
        "hasTest": False,
        "hasNames": True,
        "namesCount": 2,
        "nc": 2,
        "yamlPath": "/data/dataset.yaml",
        "workDir": "/workspace/yolo-auto",
    }
    mock_ssh.execute.side_effect = [
        ("8.1.0", "", 0),  # version
        ("", "", 1),  # original model missing
        ("/workspace/models/yolo.pt\n", "", 0),  # find exact basename
        ("", "", 0),  # check cmd
        (json.dumps(summary), "", 0),  # dataset summary
    ]
    result = setup_env(
        mock_ssh,
        "/workspace/yolo-auto",
        "/data/dataset.yaml",
        "/workspace/custom/yolo.pt",
        "/workspace/models",
    )
    assert result["ok"] is True
    assert result["modelAutoFixed"] is True
    assert result["resolvedModelPath"] == "/workspace/models/yolo.pt"


def test_setup_env_model_ambiguous(mock_ssh: MagicMock) -> None:
    mock_ssh.execute.side_effect = [
        ("8.1.0", "", 0),
        ("", "", 1),
        ("/workspace/models/a/yolo.pt\n/workspace/models/b/yolo.pt\n", "", 0),
    ]
    result = setup_env(
        mock_ssh,
        "/workspace/yolo-auto",
        "/data/dataset.yaml",
        "yolo.pt",
        "/workspace/models",
    )
    assert result["ok"] is False
    assert result["errorCode"] == "MODEL_AMBIGUOUS"
    assert len(result["candidates"]) == 2


def test_setup_env_invalid_split(mock_ssh: MagicMock) -> None:
    summary = {
        "train": {"exists": True, "path": "/data/images/train"},
        "val": {"exists": False, "path": None},
        "test": {"exists": False, "path": None},
        "hasTrain": True,
        "hasVal": False,
        "hasTest": False,
        "hasNames": True,
        "namesCount": 2,
        "nc": 2,
        "yamlPath": "/data/dataset.yaml",
        "workDir": "/workspace/yolo-auto",
    }
    mock_ssh.execute.side_effect = [
        ("8.1.0", "", 0),
        ("", "", 0),
        ("", "", 0),
        (json.dumps(summary), "", 0),
    ]
    result = setup_env(
        mock_ssh,
        "/workspace/yolo-auto",
        "/data/dataset.yaml",
        "/workspace/models/yolo.pt",
    )
    assert result["ok"] is False
    assert result["errorCode"] == "DATASET_SPLIT_INVALID"


def test_setup_env_invalid_class_count(mock_ssh: MagicMock) -> None:
    summary = {
        "train": {"exists": True, "path": "/data/images/train"},
        "val": {"exists": True, "path": "/data/images/val"},
        "test": {"exists": True, "path": "/data/images/test"},
        "hasTrain": True,
        "hasVal": True,
        "hasTest": True,
        "hasNames": True,
        "namesCount": 3,
        "nc": 2,
        "yamlPath": "/data/dataset.yaml",
        "workDir": "/workspace/yolo-auto",
    }
    mock_ssh.execute.side_effect = [
        ("8.1.0", "", 0),
        ("", "", 0),
        ("", "", 0),
        (json.dumps(summary), "", 0),
    ]
    result = setup_env(
        mock_ssh,
        "/workspace/yolo-auto",
        "/data/dataset.yaml",
        "/workspace/models/yolo.pt",
    )
    assert result["ok"] is False
    assert result["errorCode"] == "DATASET_CLASSES_MISMATCH"


def test_setup_env_missing_train_val_path(mock_ssh: MagicMock) -> None:
    summary = {
        "train": {"exists": False, "path": "/data/images/train"},
        "val": {"exists": True, "path": "/data/images/val"},
        "test": {"exists": False, "path": None},
        "hasTrain": True,
        "hasVal": True,
        "hasTest": False,
        "hasNames": True,
        "namesCount": 2,
        "nc": 2,
        "yamlPath": "/data/dataset.yaml",
        "workDir": "/workspace/yolo-auto",
    }
    mock_ssh.execute.side_effect = [
        ("8.1.0", "", 0),
        ("", "", 0),
        ("", "", 0),
        (json.dumps(summary), "", 0),
    ]
    result = setup_env(
        mock_ssh,
        "/workspace/yolo-auto",
        "/data/dataset.yaml",
        "/workspace/models/yolo.pt",
    )
    assert result["ok"] is False
    assert result["errorCode"] == "DATASET_PATH_NOT_FOUND"

