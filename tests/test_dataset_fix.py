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


def test_fix_dataset_path_dot_dry_run_plans_normalization(mock_ssh: MagicMock) -> None:
    payload = {
        "dataConfigPath": "/workspace/datasets/a/data.yaml",
        "datasetRoot": "/workspace/datasets/a",
        "dryRun": True,
        "apply": False,
        "plannedChanges": [
            {
                "kind": "normalize_dataset_root_path",
                "file": "/workspace/datasets/a/data.yaml",
                "detail": "set path to absolute dataset root",
            }
        ],
        "appliedChanges": [],
        "backupPaths": [],
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
        dry_run=True,
        apply=False,
    )
    assert result["ok"] is True
    assert result["plannedChanges"][0]["kind"] == "normalize_dataset_root_path"


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


def test_fix_dataset_path_dot_apply_writes_yaml_with_backup(mock_ssh: MagicMock) -> None:
    payload = {
        "dataConfigPath": "/workspace/datasets/a/data.yaml",
        "datasetRoot": "/workspace/datasets/a",
        "dryRun": False,
        "apply": True,
        "plannedChanges": [
            {
                "kind": "normalize_dataset_root_path",
                "file": "/workspace/datasets/a/data.yaml",
                "detail": "set path to absolute dataset root",
            }
        ],
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
    assert any(item["kind"] == "write_yaml" for item in result["appliedChanges"])
    assert result["backupPaths"]


def test_fix_dataset_absolute_path_no_redundant_normalization(mock_ssh: MagicMock) -> None:
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
            "fixedLabelRows": 0,
            "skippedRiskyRows": 0,
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
    assert all(
        item["kind"] != "normalize_dataset_root_path" for item in result["plannedChanges"]
    )


def test_fix_dataset_plan_contains_data_prefix_normalization(mock_ssh: MagicMock) -> None:
    payload = {
        "dataConfigPath": "/workspace/datasets/a/data.yaml",
        "datasetRoot": "/workspace/datasets/a",
        "dryRun": True,
        "apply": False,
        "plannedChanges": [
            {
                "kind": "normalize_split_data_prefix",
                "file": "/workspace/datasets/a/val.txt",
                "detail": "fixed data/ prefix paths: 12",
            }
        ],
        "appliedChanges": [],
        "backupPaths": [],
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
        dry_run=True,
        apply=False,
    )
    assert result["ok"] is True
    assert any(
        item["kind"] == "normalize_split_data_prefix" for item in result["plannedChanges"]
    )


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


def test_fix_dataset_split_write_uses_dot_slash_prefix(mock_ssh: MagicMock) -> None:
    payload = {
        "dataConfigPath": "/workspace/datasets/a/data.yaml",
        "datasetRoot": "/workspace/datasets/a",
        "dryRun": False,
        "apply": True,
        "plannedChanges": [],
        "appliedChanges": [],
        "backupPaths": [],
        "riskItems": [],
        "estimatedImpact": {
            "plannedChangeCount": 0,
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
    executed_cmd = mock_ssh.execute.call_args[0][0]
    assert 'rel_lines.append(f"./{rel_text}")' in executed_cmd


def test_fix_dataset_normalize_split_file_returns_consistent_arity(
    mock_ssh: MagicMock,
) -> None:
    payload = {
        "dataConfigPath": "/workspace/datasets/a/data.yaml",
        "datasetRoot": "/workspace/datasets/a",
        "dryRun": True,
        "apply": False,
        "plannedChanges": [],
        "appliedChanges": [],
        "backupPaths": [],
        "riskItems": [],
        "estimatedImpact": {
            "plannedChangeCount": 0,
            "fixedLabelRows": 0,
            "skippedRiskyRows": 0,
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
    executed_cmd = mock_ssh.execute.call_args[0][0]
    assert "return None, [], 0, 0" in executed_cmd
    assert "return p, [], 0, 0" in executed_cmd
    assert "p, lines, removed, fixed_prefix = normalize_split_file(raw)" in executed_cmd

