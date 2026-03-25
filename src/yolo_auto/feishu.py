from __future__ import annotations

import httpx


class FeishuNotifier:
    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    def send_training_update(self, title: str, body: str) -> None:
        payload = self._build_interactive_card_payload(
            title=title,
            md_text=body,
        )
        # 不做任何“降级”到文本消息；无论是否渲染成功，都保持统一的卡片消息发送策略。
        self._post_payload(payload)

    def send_training_completed_with_chart(
        self,
        title: str,
        body_text: str,
        chart_png_base64: str,
    ) -> None:
        md_text = f"{body_text}\n\n![chart](data:image/png;base64,{chart_png_base64})"
        payload = self._build_interactive_card_payload(
            title=title,
            md_text=md_text,
        )
        self._post_payload(payload)

    def _build_interactive_card_payload(self, *, title: str, md_text: str) -> dict:
        return {
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

    def _post_payload(self, payload: dict) -> bool:
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(self._webhook_url, json=payload)
                response.raise_for_status()
                return True
        except Exception:
            return False
