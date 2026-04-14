from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yolo_auto.remote_control import RemoteControlError
from yolo_auto.tools.sahi_slice import sahi_slice_dataset


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


def test_sahi_slice_success(mock_client: MagicMock) -> None:
    mock_client.sahi_slice_dataset.return_value = {
        "ok": True,
        "dataConfigPath": "/workspace/datasets/pano-sliced/data.yaml",
        "stats": {
            "sourceImages": 100,
            "totalSlices": 1200,
            "avgSlicesPerImage": 12.0,
        },
    }

    result = sahi_slice_dataset(
        mock_client,
        data_config_path="/workspace/datasets/panoramic/data.yaml",
        output_dataset_name="pano-sliced",
        output_datasets_dir="/workspace/datasets",
    )

    assert result["ok"] is True
    assert result["dataConfigPath"] == "/workspace/datasets/pano-sliced/data.yaml"
    assert result["stats"]["sourceImages"] == 100
    assert result["stats"]["totalSlices"] == 1200


def test_sahi_slice_remote_error(mock_client: MagicMock) -> None:
    mock_client.sahi_slice_dataset.side_effect = RemoteControlError(
        "REMOTE_BAD_REQUEST",
        "invalid dataConfigPath",
        retryable=False,
        hint="请检查数据集路径",
        payload={"detail": "file not found"},
    )

    result = sahi_slice_dataset(
        mock_client,
        data_config_path="/workspace/datasets/missing/data.yaml",
        output_dataset_name="out",
        output_datasets_dir="/workspace/datasets",
    )

    assert result["ok"] is False
    assert result["errorCode"] == "REMOTE_BAD_REQUEST"
    assert result["dataConfigPath"] == "/workspace/datasets/missing/data.yaml"


def test_sahi_slice_default_params(mock_client: MagicMock) -> None:
    mock_client.sahi_slice_dataset.return_value = {"ok": True, "dataConfigPath": "/x/data.yaml"}

    sahi_slice_dataset(
        mock_client,
        data_config_path="/workspace/datasets/pano/data.yaml",
        output_dataset_name="pano-out",
        output_datasets_dir="/workspace/datasets",
    )

    called_payload = mock_client.sahi_slice_dataset.call_args[0][0]
    assert called_payload["sliceHeight"] == 640
    assert called_payload["sliceWidth"] == 640
    assert called_payload["overlapHeightRatio"] == 0.2
    assert called_payload["overlapWidthRatio"] == 0.2
    assert called_payload["minAreaRatio"] == 0.1
    assert called_payload["outputDatasetName"] == "pano-out"
    assert called_payload["outputDatasetsDir"] == "/workspace/datasets"
