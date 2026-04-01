from __future__ import annotations

import json
from unittest.mock import MagicMock

from yolo_auto.tools.dataset_check import check_dataset


def test_check_dataset_success(mock_ssh: MagicMock) -> None:
    payload = {
        "dataConfigPath": "/workspace/datasets/a/data.yaml",
        "datasetRoot": "/workspace/datasets/a",
        "classCount": 2,
        "stats": {
            "totalSamples": 10,
            "missingImages": 0,
            "missingLabels": 0,
            "invalidLabelRows": 0,
            "classOutOfRange": 0,
        },
        "splits": {
            "train": {"source": "/workspace/datasets/a/train.txt", "count": 8},
            "val": {"source": "/workspace/datasets/a/val.txt", "count": 2},
            "test": {"source": None, "count": 0},
        },
        "errors": [],
        "hasErrors": False,
    }
    mock_ssh.execute.return_value = (json.dumps(payload), "", 0)
    result = check_dataset(
        mock_ssh,
        work_dir="/workspace/yolo-auto",
        data_config_path="/workspace/datasets/a/data.yaml",
    )
    assert result["ok"] is True
    assert result["classCount"] == 2
    assert result["stats"]["missingImages"] == 0


def test_check_dataset_failed_strict(mock_ssh: MagicMock) -> None:
    payload = {
        "dataConfigPath": "/workspace/datasets/a/data.yaml",
        "datasetRoot": "/workspace/datasets/a",
        "classCount": 2,
        "stats": {
            "totalSamples": 10,
            "missingImages": 1,
            "missingLabels": 1,
            "invalidLabelRows": 0,
            "classOutOfRange": 0,
        },
        "splits": {
            "train": {"source": "/workspace/datasets/a/train.txt", "count": 8},
            "val": {"source": "/workspace/datasets/a/val.txt", "count": 2},
            "test": {"source": None, "count": 0},
        },
        "errors": [{"code": "MISSING_IMAGE", "message": "image not found"}],
        "hasErrors": True,
    }
    mock_ssh.execute.return_value = (json.dumps(payload), "", 0)
    result = check_dataset(
        mock_ssh,
        work_dir="/workspace/yolo-auto",
        data_config_path="/workspace/datasets/a/data.yaml",
    )
    assert result["ok"] is False
    assert result["errorCode"] == "DATASET_CHECK_FAILED"
    assert result["stats"]["missingImages"] == 1


def test_check_dataset_exec_failed(mock_ssh: MagicMock) -> None:
    mock_ssh.execute.return_value = ("", "python error", 1)
    result = check_dataset(
        mock_ssh,
        work_dir="/workspace/yolo-auto",
        data_config_path="/workspace/datasets/a/data.yaml",
    )
    assert result["ok"] is False
    assert result["errorCode"] == "DATASET_CHECK_EXEC_FAILED"

