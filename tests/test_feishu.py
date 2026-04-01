from __future__ import annotations

from unittest.mock import MagicMock

from yolo_auto.feishu import FeishuNotifier


def test_send_training_completed_with_chart_interactive_card() -> None:
    notifier = FeishuNotifier(webhook_url="http://example")
    notifier._send_card = MagicMock(return_value=True)

    notifier.send_training_completed_with_chart(
        title="t",
        body_text="b",
        chart_png_base64="BASE64",
    )

    assert notifier._send_card.call_count == 1
    card = notifier._send_card.call_args.args[0]
    md_content = card["elements"][0]["text"]["content"]
    assert "data:image/png;base64,BASE64" in md_content


def test_send_training_update_interactive_card() -> None:
    notifier = FeishuNotifier(webhook_url="http://example")
    notifier._send_card = MagicMock(return_value=True)

    notifier.send_training_update(
        title="t",
        body="b\nc",
    )

    assert notifier._send_card.call_count == 1
    card = notifier._send_card.call_args.args[0]
    md_content = card["elements"][0]["text"]["content"]
    assert md_content == "b\nc"


def test_send_rich_card_basic() -> None:
    notifier = FeishuNotifier(webhook_url="http://example")
    notifier._send_card = MagicMock(return_value=True)

    notifier.send_rich_card(title="T", md_text="hello", header_color="blue")

    assert notifier._send_card.call_count == 1
    card = notifier._send_card.call_args.args[0]
    assert card["header"]["title"]["content"] == "T"
    assert card["header"]["template"] == "blue"
    assert card["elements"][0]["text"]["content"] == "hello"


def test_send_rich_card_with_message_id_app_bot() -> None:
    notifier = FeishuNotifier(app_id="a", app_secret="b", chat_id="c")
    notifier._send_card_via_app_bot_with_message_id = MagicMock(return_value="om_xxx")

    message_id = notifier.send_rich_card_with_message_id(
        title="T",
        md_text="hello",
        header_color="blue",
    )

    assert message_id == "om_xxx"


def test_update_rich_card_app_bot() -> None:
    notifier = FeishuNotifier(app_id="a", app_secret="b", chat_id="c")
    notifier._update_card_via_app_bot = MagicMock(return_value=True)

    ok = notifier.update_rich_card(
        message_id="om_xxx",
        title="T",
        md_text="updated",
        header_color="green",
    )

    assert ok is True
    notifier._update_card_via_app_bot.assert_called_once()


def test_send_schema_card_with_message_id_app_bot() -> None:
    notifier = FeishuNotifier(app_id="a", app_secret="b", chat_id="c")
    notifier._send_card_via_app_bot_with_message_id = MagicMock(return_value="om_schema")

    message_id = notifier.send_schema_card_with_message_id(
        card={"schema": "2.0", "header": {}, "body": {}}
    )

    assert message_id == "om_schema"


def test_update_schema_card_app_bot() -> None:
    notifier = FeishuNotifier(app_id="a", app_secret="b", chat_id="c")
    notifier._update_card_via_app_bot = MagicMock(return_value=True)

    ok = notifier.update_schema_card(
        message_id="om_schema",
        card={"schema": "2.0", "header": {}, "body": {}},
    )

    assert ok is True
    notifier._update_card_via_app_bot.assert_called_once()

