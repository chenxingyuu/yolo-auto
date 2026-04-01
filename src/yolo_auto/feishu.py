from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx


@dataclass(frozen=True)
class _AppBotConfig:
    app_id: str
    app_secret: str
    chat_id: str


class FeishuNotifier:
    def __init__(
        self,
        *,
        webhook_url: str | None = None,
        app_id: str | None = None,
        app_secret: str | None = None,
        chat_id: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._webhook_url = webhook_url.strip() if webhook_url else None
        if app_id and app_secret and chat_id:
            self._app_bot = _AppBotConfig(
                app_id=app_id.strip(),
                app_secret=app_secret.strip(),
                chat_id=chat_id.strip(),
            )
        else:
            self._app_bot = None
        self._timeout_seconds = timeout_seconds

        self._tenant_token: str | None = None
        self._tenant_token_expire_at: float = 0.0

    def send_training_update(self, title: str, body: str) -> None:
        payload = self._build_interactive_card_payload(
            title=title,
            md_text=body,
        )
        # 不做任何“降级”到文本消息；无论是否渲染成功，都保持统一的卡片消息发送策略。
        self._send_card(payload["card"])

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
        self._send_card(payload["card"])

    def send_training_completed_with_chart_png(
        self,
        *,
        title: str,
        body_md: str,
        chart_png: bytes,
        header_color: Literal["blue", "green", "red", "orange"] = "green",
        actions: list[dict[str, Any]] | None = None,
    ) -> None:
        image_key = self._upload_image(chart_png) if self._app_bot else None
        card = self._build_rich_card(
            title=title,
            md_text=body_md,
            header_color=header_color,
            image_key=image_key,
            actions=actions,
        )
        self._send_card(card)

    def send_rich_card(
        self,
        *,
        title: str,
        md_text: str,
        header_color: Literal["blue", "green", "red", "orange"] = "blue",
        actions: list[dict[str, Any]] | None = None,
    ) -> None:
        card = self._build_rich_card(
            title=title,
            md_text=md_text,
            header_color=header_color,
            actions=actions,
        )
        self._send_card(card)

    def send_rich_card_with_message_id(
        self,
        *,
        title: str,
        md_text: str,
        header_color: Literal["blue", "green", "red", "orange"] = "blue",
        actions: list[dict[str, Any]] | None = None,
    ) -> str | None:
        if not self._app_bot:
            return None
        card = self._build_rich_card(
            title=title,
            md_text=md_text,
            header_color=header_color,
            actions=actions,
        )
        return self._send_card_via_app_bot_with_message_id(card)

    def update_rich_card(
        self,
        *,
        message_id: str,
        title: str,
        md_text: str,
        header_color: Literal["blue", "green", "red", "orange"] = "blue",
        actions: list[dict[str, Any]] | None = None,
    ) -> bool:
        if not self._app_bot:
            return False
        card = self._build_rich_card(
            title=title,
            md_text=md_text,
            header_color=header_color,
            actions=actions,
        )
        return self._update_card_via_app_bot(message_id=message_id, card=card)

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

    def _build_rich_card(
        self,
        *,
        title: str,
        md_text: str,
        header_color: Literal["blue", "green", "red", "orange"] = "blue",
        image_key: str | None = None,
        actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        elements: list[dict[str, Any]] = [
            {"tag": "div", "text": {"tag": "lark_md", "content": md_text}},
        ]

        if image_key:
            elements.append({"tag": "hr"})
            elements.append(
                {
                    "tag": "img",
                    "img_key": image_key,
                    "alt": {"tag": "plain_text", "content": "chart"},
                }
            )

        if actions:
            elements.append({"tag": "hr"})
            elements.append({"tag": "action", "actions": actions})

        elements.append(
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    }
                ],
            }
        )

        return {
            "config": {"wide_screen_mode": True, "enable_forward": True},
            "header": {
                "template": header_color,
                "title": {"tag": "plain_text", "content": title},
            },
            "elements": elements,
        }

    def _send_card(self, card: dict[str, Any]) -> bool:
        # 优先走应用机器人（可支持图片等能力），失败再尝试 webhook 降级。
        if self._app_bot:
            try:
                ok_sent = self._send_card_via_app_bot(card)
                if ok_sent:
                    return True
            except Exception:
                pass
        if self._webhook_url:
            return self._send_card_via_webhook(card)
        return False

    def _send_card_via_webhook(self, card: dict[str, Any]) -> bool:
        payload = {"msg_type": "interactive", "card": card}
        return self._post_json_with_retry(self._webhook_url, payload, headers=None)

    def _send_card_via_app_bot(self, card: dict[str, Any]) -> bool:
        return self._send_card_via_app_bot_with_message_id(card) is not None

    def _send_card_via_app_bot_with_message_id(self, card: dict[str, Any]) -> str | None:
        assert self._app_bot is not None
        token = self._get_tenant_access_token()
        url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "receive_id": self._app_bot.chat_id,
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False),
        }
        data = self._post_json_with_retry_result(url, payload, headers=headers)
        if not data:
            return None
        message_id = str(data.get("message_id", "")).strip()
        return message_id or None

    def _update_card_via_app_bot(self, *, message_id: str, card: dict[str, Any]) -> bool:
        assert self._app_bot is not None
        token = self._get_tenant_access_token()
        url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False),
        }
        return (
            self._post_json_with_retry_result(
                url,
                payload,
                headers=headers,
                method="patch",
            )
            is not None
        )

    def _post_json_with_retry(
        self,
        url: str | None,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None,
        max_attempts: int = 3,
    ) -> bool:
        return (
            self._post_json_with_retry_result(
                url,
                payload,
                headers=headers,
                max_attempts=max_attempts,
            )
            is not None
        )

    def _post_json_with_retry_result(
        self,
        url: str | None,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None,
        max_attempts: int = 3,
        method: Literal["post", "patch"] = "post",
    ) -> dict[str, Any] | None:
        if not url:
            return None
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                timeout = httpx.Timeout(self._timeout_seconds)
                with httpx.Client(timeout=timeout) as client:
                    if method == "patch":
                        response = client.patch(url, json=payload, headers=headers)
                    else:
                        response = client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    if isinstance(data, dict):
                        return dict(data.get("data") or {})
                    return {}
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    time.sleep(0.4 * (2**attempt))
        _ = last_exc
        return None

    def _get_tenant_access_token(self) -> str:
        assert self._app_bot is not None
        now = time.time()
        if self._tenant_token and now < self._tenant_token_expire_at - 60:
            return self._tenant_token

        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        payload = {"app_id": self._app_bot.app_id, "app_secret": self._app_bot.app_secret}
        timeout = httpx.Timeout(self._timeout_seconds)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        token = str(data.get("tenant_access_token", "")).strip()
        expire_in = int(data.get("expire", 0) or 0)
        if not token or expire_in <= 0:
            raise RuntimeError(f"failed to get tenant_access_token: {data}")
        self._tenant_token = token
        self._tenant_token_expire_at = now + float(expire_in)
        return token

    def _upload_image(self, png_bytes: bytes) -> str:
        assert self._app_bot is not None
        token = self._get_tenant_access_token()
        url = "https://open.feishu.cn/open-apis/im/v1/images"
        headers = {"Authorization": f"Bearer {token}"}
        files = {
            "image_type": (None, "message"),
            "image": ("chart.png", png_bytes, "image/png"),
        }
        timeout = httpx.Timeout(self._timeout_seconds)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, files=files)
            resp.raise_for_status()
            data = resp.json()
        image_key = (
            str((data.get("data") or {}).get("image_key", "")).strip()
            if isinstance(data, dict)
            else ""
        )
        if not image_key:
            raise RuntimeError(f"failed to upload image: {data}")
        return image_key
