from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.tools.model_ref import parse_model_ref, resolve_model_ref_to_record


def test_parse_model_ref_basic() -> None:
    assert parse_model_ref("my-model:approved") == ("my-model", "approved")
    assert parse_model_ref("  a:b  ") == ("a", "b")


def test_parse_model_ref_registry_prefix() -> None:
    assert parse_model_ref("registry:foo-bar:candidate") == ("foo-bar", "candidate")


def test_parse_model_ref_invalid() -> None:
    assert parse_model_ref("") is None
    assert parse_model_ref("nocolon") is None
    assert parse_model_ref(":onlyalias") is None


def test_resolve_model_ref_to_record_uses_tracker_and_store() -> None:
    tracker = MagicMock()
    tracker._model_registry_enable = True
    mv = MagicMock()
    mv.run_id = "run-xyz"
    tracker.get_model_version_by_alias.return_value = mv

    store = MagicMock()
    rec = JobRecord(
        job_id="j1",
        run_id="run-xyz",
        status=JobStatus.COMPLETED,
        pid="0",
        paths={},
        created_at=1,
        updated_at=1,
    )
    store.find_by_run_id.return_value = rec

    out = resolve_model_ref_to_record(
        tracker, store, model_name="m", alias="approved"
    )
    assert out is rec
    tracker.get_model_version_by_alias.assert_called_once_with("m", "approved")
    store.find_by_run_id.assert_called_once_with("run-xyz")


def test_resolve_model_ref_disabled_tracker() -> None:
    tracker = MagicMock()
    tracker._model_registry_enable = False
    store = MagicMock()
    assert (
        resolve_model_ref_to_record(tracker, store, model_name="m", alias="a")
        is None
    )
    store.find_by_run_id.assert_not_called()
