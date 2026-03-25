from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.tools.setup_env import setup_env


def test_setup_env_success(mock_ssh: MagicMock) -> None:
    # 1) ultralytics 可 import
    # 2) work_dir + data_config_path 都存在
    mock_ssh.execute.side_effect = [
        ("8.1.0", "", 0),
        ("", "", 0),
    ]

    result = setup_env(mock_ssh, "/workspace/yolo-auto", "/data/dataset.yaml")
    assert result["ok"] is True
    assert result["reachable"] is True
    assert result["validData"] is True
    assert result["yoloVersion"] == "8.1.0"


def test_setup_env_unreachable(mock_ssh: MagicMock) -> None:
    mock_ssh.execute.side_effect = [
        ("", "ultralytics not available", 1),
    ]

    result = setup_env(mock_ssh, "/workspace/yolo-auto", "/data/dataset.yaml")
    assert result["ok"] is False
    assert result["errorCode"] == "ENV_UNREACHABLE"


def test_setup_env_invalid_data(mock_ssh: MagicMock) -> None:
    mock_ssh.execute.side_effect = [
        ("8.1.0", "", 0),
        ("", "workDir or dataConfigPath missing", 1),
    ]

    result = setup_env(mock_ssh, "/workspace/yolo-auto", "/data/dataset.yaml")
    assert result["ok"] is False
    assert result["errorCode"] == "DATA_CONFIG_INVALID"
    assert result["reachable"] is True
    assert result["validData"] is False

