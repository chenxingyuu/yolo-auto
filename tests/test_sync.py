from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.tools.dataset import export_and_sync_cvat_dataset
from yolo_auto.tools.sync import sync_dataset


def test_sync_dataset_success(mock_ssh: MagicMock) -> None:
    mock_ssh.execute.side_effect = [
        ("", "", 0),  # command -v mc
        ("", "", 0),  # command -v unzip
        ("", "", 0),  # mc stat
        ("", "", 0),  # sync command
        ("nested/data.yaml\n", "", 0),  # detect data yaml
        ("./nested/data.yaml\n./nested/images/train/1.jpg\n", "", 0),  # list files
    ]

    result = sync_dataset(
        mock_ssh,
        minio_alias="minio",
        minio_bucket="cvat-export",
        minio_prefix="exports",
        filename="task8-20260327075925.zip",
        dataset_name="task8",
        datasets_dir="/workspace/datasets",
    )

    assert result["ok"] is True
    assert result["dataConfigPath"] == "/workspace/datasets/task8/nested/data.yaml"
    assert result["source"] == "minio/cvat-export/exports/task8-20260327075925.zip"
    assert result["datasetName"] == "task8"
    assert result["extractedDir"] == "/workspace/datasets/task8"
    assert len(result["files"]) >= 1


def test_sync_dataset_fail_when_object_missing(mock_ssh: MagicMock) -> None:
    mock_ssh.execute.side_effect = [
        ("", "", 0),  # command -v mc
        ("", "", 0),  # command -v unzip
        ("", "", 1),  # mc stat
    ]

    result = sync_dataset(
        mock_ssh,
        minio_alias="minio",
        minio_bucket="cvat-export",
        minio_prefix="exports",
        filename="missing.zip",
        dataset_name="task9",
        datasets_dir="/workspace/datasets",
    )

    assert result["ok"] is False
    assert result["errorCode"] == "MINIO_OBJECT_NOT_FOUND"


def test_sync_dataset_fail_when_data_yaml_missing(mock_ssh: MagicMock) -> None:
    mock_ssh.execute.side_effect = [
        ("", "", 0),  # command -v mc
        ("", "", 0),  # command -v unzip
        ("", "", 0),  # mc stat
        ("", "", 0),  # sync command
        ("", "", 0),  # detect data yaml
    ]

    result = sync_dataset(
        mock_ssh,
        minio_alias="minio",
        minio_bucket="cvat-export",
        minio_prefix="exports",
        filename="task9.zip",
        dataset_name="task9",
        datasets_dir="/workspace/datasets",
    )

    assert result["ok"] is False
    assert result["errorCode"] == "DATA_YAML_NOT_FOUND"


def test_sync_dataset_accepts_exports_prefixed_filename(mock_ssh: MagicMock) -> None:
    mock_ssh.execute.side_effect = [
        ("", "", 0),  # command -v mc
        ("", "", 0),  # command -v unzip
        ("", "", 0),  # mc stat
        ("", "", 0),  # sync command
        ("data.yaml\n", "", 0),  # detect data yaml
        ("./data.yaml\n", "", 0),  # list files
    ]

    result = sync_dataset(
        mock_ssh,
        minio_alias="minio",
        minio_bucket="cvat-export",
        minio_prefix="exports",
        filename="exports/task8-20260327075925.zip",
        dataset_name="task8",
        datasets_dir="/workspace/datasets",
    )

    assert result["ok"] is True
    assert result["source"] == "minio/cvat-export/exports/task8-20260327075925.zip"


def test_export_and_sync_success(mock_ssh: MagicMock) -> None:
    cvat_client = MagicMock()
    cvat_client.export_task_dataset_to_cloud.return_value = None
    mock_ssh.execute.side_effect = [
        ("", "", 0),  # command -v mc
        ("", "", 0),  # command -v unzip
        ("", "", 0),  # mc stat
        ("", "", 0),  # sync command
        ("data.yaml\n", "", 0),  # detect data yaml
        ("./data.yaml\n", "", 0),  # list files
    ]

    result = export_and_sync_cvat_dataset(
        cvat_client,
        mock_ssh,
        task_id=8,
        dataset_name="task8_sync",
        include_images=True,
        cloud_storage_id=1,
        minio_alias="minio",
        minio_bucket="cvat-export",
        minio_prefix="exports",
        datasets_dir="/workspace/datasets",
    )

    assert result["ok"] is True
    assert result["sync"]["dataConfigPath"] == "/workspace/datasets/task8_sync/data.yaml"
