from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class RemoteControlError(Exception):
    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        retryable: bool,
        hint: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable
        self.hint = hint
        self.payload = payload or {}


@dataclass(frozen=True)
class RemoteControlConfig:
    base_url: str
    bearer_token: str | None = None
    timeout_seconds: int = 30


class HttpControlClient:
    def __init__(self, config: RemoteControlConfig) -> None:
        self._config = config

    def start_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/train/start", json=payload)

    def stop_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/train/stop", json=payload)

    def get_training_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("GET", "/api/v1/train/status", params=payload)

    def run_validation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/validate/run", json=payload)

    def setup_env(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/env/setup", json=payload)

    def check_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/dataset/check", json=payload)

    def fix_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/dataset/fix", json=payload)

    def sync_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/dataset/sync", json=payload)

    def run_export(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/export/run", json=payload)

    def auto_tune(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/tune/auto", json=payload)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/jobs/{job_id}")

    def list_jobs(self, limit: int = 20) -> dict[str, Any]:
        return self._request("GET", "/api/v1/jobs", params={"limit": limit})

    def health_check(self) -> tuple[bool, str]:
        """Return (reachable, detail). Never raises."""
        try:
            self._request("GET", "/api/v1/jobs", params={"limit": 1})
            return True, f"{self._config.base_url} 可达"
        except RemoteControlError as exc:
            if exc.error_code == "REMOTE_UNAUTHORIZED":
                return False, f"认证失败：Token 无效或缺失（{self._config.base_url}）"
            return False, f"{exc.hint}（{self._config.base_url}）"
        except Exception as exc:
            return False, f"未知错误：{exc}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if self._config.bearer_token:
            headers["Authorization"] = f"Bearer {self._config.bearer_token}"

        url = f"{self._config.base_url.rstrip('/')}{path}"
        try:
            with httpx.Client(timeout=self._config.timeout_seconds) as client:
                resp = client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=headers,
                )
        except httpx.TimeoutException as exc:
            raise RemoteControlError(
                "REMOTE_TIMEOUT",
                str(exc),
                retryable=True,
                hint="控制面请求超时，请稍后重试",
            ) from exc
        except httpx.HTTPError as exc:
            raise RemoteControlError(
                "REMOTE_UNREACHABLE",
                str(exc),
                retryable=True,
                hint="无法连接训练控制面，请检查 YOLO_CONTROL_BASE_URL 与网络",
            ) from exc

        try:
            data = resp.json()
        except Exception as exc:
            raise RemoteControlError(
                "REMOTE_INVALID_RESPONSE",
                f"invalid JSON response from control API: {resp.text[:200]}",
                retryable=False,
                hint="检查训练控制面返回格式",
            ) from exc

        if 200 <= resp.status_code < 300:
            if isinstance(data, dict):
                return data
            raise RemoteControlError(
                "REMOTE_INVALID_RESPONSE",
                "control API response is not an object",
                retryable=False,
                hint="检查训练控制面返回格式",
            )

        error_code = "REMOTE_BAD_REQUEST"
        retryable = False
        hint = "请检查请求参数"
        if resp.status_code == 401:
            error_code = "REMOTE_UNAUTHORIZED"
            hint = "Bearer Token 无效或缺失"
        elif resp.status_code in {408, 429, 500, 502, 503, 504}:
            error_code = "REMOTE_SERVER_ERROR"
            retryable = True
            hint = "训练控制面暂时不可用，请稍后重试"
        message = str(data.get("message") or data.get("detail") or resp.text[:300])
        raise RemoteControlError(
            error_code,
            message,
            retryable=retryable,
            hint=hint,
            payload=data if isinstance(data, dict) else {},
        )
