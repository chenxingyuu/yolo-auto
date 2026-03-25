from __future__ import annotations

import httpx


class FeishuNotifier:
    def __init__(self, webhook_url: str, message_mode: str = "text") -> None:
        self._webhook_url = webhook_url
        self._message_mode = message_mode.strip().lower()

    def send_text(self, text: str) -> None:
        payload = {"msg_type": "text", "content": {"text": text}}
        self._post_payload(payload)

    def send_training_update(self, title: str, body: str) -> None:
        if self._message_mode == "card":
            self._send_post_or_text(title, body)
        else:
            self.send_text(f"{title}\n{body}")

    def _send_post_or_text(self, title: str, body: str) -> None:
        lines = [segment.strip() for segment in body.split("\n") if segment.strip()]
        if not lines:
            lines = [body.strip() or "(empty)"]
        content_rows: list[list[dict[str, str]]] = []
        for line in lines:
            content_rows.append([{"tag": "text", "text": line}])
        payload = {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": title,
                        "content": content_rows,
                    }
                }
            },
        }
        if not self._post_payload(payload):
            self.send_text(f"{title}\n{body}")

    def _post_payload(self, payload: dict) -> bool:
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(self._webhook_url, json=payload)
                response.raise_for_status()
                return True
        except Exception:
            return False
