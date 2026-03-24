from __future__ import annotations

import httpx


class FeishuNotifier:
    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    def send_text(self, text: str) -> None:
        payload = {"msg_type": "text", "content": {"text": text}}
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(self._webhook_url, json=payload)
                response.raise_for_status()
        except Exception:
            # 通知失败不应阻断训练主流程
            return

