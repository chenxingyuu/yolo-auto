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

    def send_training_completed_with_chart(
        self,
        title: str,
        body_text: str,
        chart_png_base64: str,
    ) -> None:
        # 仅在 card 模式下尝试发送带图片的交互卡片；否则降级为文本消息。
        if self._message_mode != "card":
            self.send_training_update(title, body_text)
            return

        md_text = f"{body_text}\n\n![chart](data:image/png;base64,{chart_png_base64})"
        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True, "enable_forward": True},
                "header": {"title": {"tag": "plain_text", "content": title}},
                "elements": [
                    {
                        "tag": "div",
                        "text": {"tag": "lark_md", "content": md_text},
                    }
                ],
            },
        }

        if not self._post_payload(payload):
            # 如果图片嵌入失败（例如数据 URL 不被渲染），退回纯文本。
            self.send_training_update(title, body_text)

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
