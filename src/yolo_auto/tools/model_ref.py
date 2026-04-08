from __future__ import annotations

from typing import TYPE_CHECKING

from yolo_auto.models import JobRecord

if TYPE_CHECKING:
    from yolo_auto.state_store import JobStateStore
    from yolo_auto.tracker import MLflowTracker


def parse_model_ref(raw: str) -> tuple[str, str] | None:
    """解析 `modelName:alias` 或 `registry:modelName:alias`。"""
    s = raw.strip()
    if not s:
        return None
    if s.lower().startswith("registry:"):
        s = s[9:].strip()
    if ":" not in s:
        return None
    model_name, alias = s.rsplit(":", 1)
    model_name, alias = model_name.strip(), alias.strip()
    if not model_name or not alias:
        return None
    return model_name, alias


def resolve_model_ref_to_record(
    tracker: MLflowTracker,
    state_store: JobStateStore,
    *,
    model_name: str,
    alias: str,
) -> JobRecord | None:
    """通过 Registry alias 解析到本地 JobRecord（依赖 runId 与状态库中的任务）。"""
    if not getattr(tracker, "_model_registry_enable", False):
        return None
    try:
        mv = tracker.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None
    run_id = mv.run_id
    if not run_id:
        return None
    return state_store.find_by_run_id(run_id)
