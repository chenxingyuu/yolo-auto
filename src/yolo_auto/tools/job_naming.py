from __future__ import annotations

import os
import re
import time

from yolo_auto.state_store import JobStateStore

_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


def _extract_stem(raw: str, *, fallback: str) -> str:
    cleaned = raw.strip().replace("\\", "/")
    stem = os.path.splitext(os.path.basename(cleaned))[0]
    return stem or fallback


def _normalize_part(raw: str, *, fallback: str) -> str:
    lowered = raw.lower().strip()
    normalized = _NON_ALNUM_PATTERN.sub("-", lowered).strip("-")
    return normalized or fallback


def _build_base_job_id(model: str, data_config_path: str, now_ts: int) -> str:
    model_part = _normalize_part(_extract_stem(model, fallback="model"), fallback="model")
    data_part = _normalize_part(_extract_stem(data_config_path, fallback="data"), fallback="data")
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now_ts))
    return f"train-{model_part}-{data_part}-{timestamp}"


def resolve_job_id(
    job_id: str | None,
    *,
    model: str,
    data_config_path: str,
    state_store: JobStateStore,
    now_ts: int | None = None,
) -> str:
    if job_id:
        return job_id
    now = int(time.time()) if now_ts is None else now_ts
    base_id = _build_base_job_id(model, data_config_path, now)
    candidate = base_id
    suffix = 2
    while state_store.get(candidate) is not None:
        candidate = f"{base_id}-{suffix}"
        suffix += 1
    return candidate
