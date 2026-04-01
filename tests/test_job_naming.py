from __future__ import annotations

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.tools.job_naming import resolve_job_id


def _upsert_job(state_store, job_id: str) -> None:
    state_store.upsert(
        JobRecord(
            job_id=job_id,
            run_id="run-1",
            status=JobStatus.RUNNING,
            pid="1",
            paths={},
            created_at=1,
            updated_at=1,
        )
    )


def test_resolve_job_id_uses_new_default_format(state_store) -> None:
    job_id = resolve_job_id(
        None,
        model="yolov8n.pt",
        data_config_path="/workspace/datasets/coco128.yaml",
        state_store=state_store,
        now_ts=1711956645,  # 2024-04-01 15:30:45 localtime
    )
    assert job_id.startswith("train-yolov8n-coco128-")
    assert job_id.count("-") >= 4


def test_resolve_job_id_conflict_adds_numeric_suffix(state_store) -> None:
    first = resolve_job_id(
        None,
        model="yolov8n.pt",
        data_config_path="/workspace/datasets/coco128.yaml",
        state_store=state_store,
        now_ts=1711956645,
    )
    _upsert_job(state_store, first)

    second = resolve_job_id(
        None,
        model="yolov8n.pt",
        data_config_path="/workspace/datasets/coco128.yaml",
        state_store=state_store,
        now_ts=1711956645,
    )
    assert second == f"{first}-2"


def test_resolve_job_id_keeps_explicit_value(state_store) -> None:
    assert (
        resolve_job_id(
            "custom-job-id",
            model="yolov8n.pt",
            data_config_path="/workspace/datasets/coco128.yaml",
            state_store=state_store,
            now_ts=1711956645,
        )
        == "custom-job-id"
    )
