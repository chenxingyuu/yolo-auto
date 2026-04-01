from __future__ import annotations

import json
from unittest.mock import MagicMock

from yolo_auto.tools.dataset_fix import fix_dataset


def test_fix_dataset_dry_run_success(mock_ssh: MagicMock) -> None:
    payload = {
        "dataConfigPath": "/workspace/datasets/a/data.yaml",
        "datasetRoot": "/workspace/datasets/a",
        "dryRun": True,
        "apply": False,
        "plannedChanges": [{"kind": "sync_nc", "file": "/workspace/datasets/a/data.yaml"}],
        "appliedChanges": [],
        "backupPaths": [],
        "riskItems": [],
        "estimatedImpact": {
            "plannedChangeCount": 1,
            "fixedLabelRows": 3,
            "skippedRiskyRows": 1,
        },
    }
    mock_ssh.execute.return_value = (json.dumps(payload), "", 0)
    result = fix_dataset(
        mock_ssh,
        work_dir="/workspace/yolo-auto",
        data_config_path="/workspace/datasets/a/data.yaml",
        dry_run=True,
        apply=False,
    )
    assert result["ok"] is True
    assert result["dryRun"] is True
    assert result["apply"] is False
    assert result["estimatedImpact"]["plannedChangeCount"] == 1


def test_fix_dataset_apply_success(mock_ssh: MagicMock) -> None:
    payload = {
        "dataConfigPath": "/workspace/datasets/a/data.yaml",
        "datasetRoot": "/workspace/datasets/a",
        "dryRun": False,
        "apply": True,
        "plannedChanges": [{"kind": "generate_val_split", "file": "/workspace/datasets/a/val.txt"}],
        "appliedChanges": [{"kind": "write_yaml", "file": "/workspace/datasets/a/data.yaml"}],
        "backupPaths": ["/workspace/datasets/a/data.yaml.bak.1710000000"],
        "riskItems": [],
        "estimatedImpact": {
            "plannedChangeCount": 1,
            "fixedLabelRows": 0,
            "skippedRiskyRows": 0,
        },
    }
    mock_ssh.execute.return_value = (json.dumps(payload), "", 0)
    result = fix_dataset(
        mock_ssh,
        work_dir="/workspace/yolo-auto",
        data_config_path="/workspace/datasets/a/data.yaml",
        dry_run=False,
        apply=True,
    )
    assert result["ok"] is True
    assert result["apply"] is True
    assert result["backupPaths"]


def test_fix_dataset_invalid_mode() -> None:
    ssh = MagicMock()
    result = fix_dataset(
        ssh,
        work_dir="/workspace/yolo-auto",
        data_config_path="/workspace/datasets/a/data.yaml",
        dry_run=True,
        apply=True,
    )
    assert result["ok"] is False
    assert result["errorCode"] == "INVALID_FIX_MODE"


def test_fix_dataset_exec_failed(mock_ssh: MagicMock) -> None:
    mock_ssh.execute.return_value = ("", "exec failed", 1)
    result = fix_dataset(
        mock_ssh,
        work_dir="/workspace/yolo-auto",
        data_config_path="/workspace/datasets/a/data.yaml",
    )
    assert result["ok"] is False
    assert result["errorCode"] == "DATASET_FIX_EXEC_FAILED"

