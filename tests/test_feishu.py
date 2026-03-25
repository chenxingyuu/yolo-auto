from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.feishu import FeishuNotifier


def test_send_training_completed_with_chart_card_mode() -> None:
    notifier = FeishuNotifier(webhook_url="http://example", message_mode="card")
    notifier._post_payload = MagicMock(return_value=True)

    notifier.send_training_completed_with_chart(
        title="t",
        body_text="b",
        chart_png_base64="BASE64",
    )

    assert notifier._post_payload.call_count == 1
    payload = notifier._post_payload.call_args.args[0]
    assert payload["msg_type"] == "interactive"
    md_content = payload["card"]["elements"][0]["text"]["content"]
    assert "data:image/png;base64,BASE64" in md_content

