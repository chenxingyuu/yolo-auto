from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.tools.dataset import (
    export_and_sync_cvat_project_dataset,
    export_cvat_project_dataset,
)


def test_export_cvat_project_dataset_success() -> None:
    client = MagicMock()
    client.export_project_dataset_to_cloud.return_value = "rq-proj-1"
    result = export_cvat_project_dataset(
        client,
        project_id=3,
        dataset_name="myproj",
        include_images=True,
        cloud_storage_id=99,
    )
    assert result["ok"] is True
    assert result["projectId"] == 3
    assert result["datasetName"] == "myproj"
    assert result["rqId"] == "rq-proj-1"
    fname = result["cloudExport"]["filename"]
    assert fname.startswith("exports/myproj-proj3-")
    assert fname.endswith(".zip")
    client.export_project_dataset_to_cloud.assert_called_once()


def test_export_cvat_project_dataset_requires_storage() -> None:
    client = MagicMock()
    result = export_cvat_project_dataset(
        client,
        project_id=3,
        dataset_name="myproj",
        cloud_storage_id=None,
    )
    assert result["ok"] is False
    assert result["errorCode"] == "CVAT_CLOUD_STORAGE_REQUIRED"
    client.export_project_dataset_to_cloud.assert_not_called()


def test_export_and_sync_project_success(mock_ssh: MagicMock) -> None:
    cvat_client = MagicMock()
    cvat_client.export_project_dataset_to_cloud.return_value = "rq-p"
    cvat_client.get_request.side_effect = [
        {"status": {"value": "started"}, "message": ""},
        {"status": {"value": "finished"}, "message": ""},
    ]
    mock_ssh.execute.side_effect = [
        ("", "", 0),
        ("", "", 0),
        ("", "", 0),
        ("", "", 0),
        ("data.yaml\n", "", 0),
        ("./data.yaml\n", "", 0),
    ]

    result = export_and_sync_cvat_project_dataset(
        cvat_client,
        mock_ssh,
        project_id=5,
        dataset_name="proj5",
        include_images=True,
        cloud_storage_id=1,
        poll_seconds=0.01,
        max_wait_seconds=30.0,
        minio_alias="minio",
        minio_bucket="cvat-export",
        minio_prefix="exports",
        datasets_dir="/workspace/datasets",
    )

    assert result["ok"] is True
    assert result["projectId"] == 5
    assert result["sync"]["dataConfigPath"] == "/workspace/datasets/proj5/data.yaml"
