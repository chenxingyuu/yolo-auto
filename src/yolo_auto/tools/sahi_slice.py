from __future__ import annotations

from typing import Any

from yolo_auto.errors import err
from yolo_auto.remote_control import HttpControlClient, RemoteControlError


def sahi_slice_dataset(
    control_client: HttpControlClient,
    *,
    data_config_path: str,
    output_dataset_name: str,
    output_datasets_dir: str,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    min_area_ratio: float = 0.1,
) -> dict[str, Any]:
    try:
        return control_client.sahi_slice_dataset({
            "dataConfigPath": data_config_path,
            "outputDatasetName": output_dataset_name,
            "outputDatasetsDir": output_datasets_dir,
            "sliceHeight": slice_height,
            "sliceWidth": slice_width,
            "overlapHeightRatio": overlap_height_ratio,
            "overlapWidthRatio": overlap_width_ratio,
            "minAreaRatio": min_area_ratio,
        })
    except RemoteControlError as exc:
        return err(
            error_code=exc.error_code,
            message=str(exc),
            retryable=exc.retryable,
            hint=exc.hint,
            payload={"dataConfigPath": data_config_path, **exc.payload},
        )
    except Exception as exc:
        return err(
            error_code="SAHI_SLICE_FAILED",
            message=str(exc),
            retryable=True,
            hint="请检查控制面日志与数据集路径",
            payload={"dataConfigPath": data_config_path},
        )
