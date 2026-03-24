from __future__ import annotations

from typing import Any


def ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        **payload,
    }


def err(
    error_code: str,
    message: str,
    retryable: bool,
    hint: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ok": False,
        "errorCode": error_code,
        "error": message,
        "retryable": retryable,
        "hint": hint,
    }
    if payload:
        base.update(payload)
    return base

