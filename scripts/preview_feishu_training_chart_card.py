#!/usr/bin/env python3
"""向当前飞书群发送一张「训练里程碑」样式 Schema 2.0 卡片（合成折线图 + 三列指标区）。

仅用于本地预览。请在项目根目录配置 `.env`（FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_CHAT_ID 等）。

用法::

    uv run python scripts/preview_feishu_training_chart_card.py
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

from yolo_auto.config import load_settings
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.tools.status import _build_training_schema_card


def main() -> None:
    settings = load_settings()
    notifier = FeishuNotifier(
        webhook_url=settings.feishu_webhook_url,
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
        chat_id=settings.feishu_chat_id,
    )
    rows: list[dict[str, str]] = []
    for ep in range(1, 11):
        rows.append(
            {
                "epoch": str(ep),
                "train/box_loss": f"{0.5 - ep * 0.03:.4f}",
                "metrics/mAP50(B)": f"{0.1 + ep * 0.07:.4f}",
                "metrics/mAP50-95(B)": f"{0.08 + ep * 0.06:.4f}",
                "metrics/recall(B)": f"{0.2 + ep * 0.05:.4f}",
            }
        )
    card = _build_training_schema_card(
        title="YOLO模型训练里程碑（预览）",
        header_template="blue",
        epoch_text="10/100",
        elapsed_text="5m30s",
        eta_text="45m00s",
        updated_time_text="2099-01-01 12:00:00",
        mlflow_url=None,
        top_img_key=settings.feishu_card_img_key,
        top_img_fallback_key=settings.feishu_card_fallback_img_key,
        training_rows=rows,
    )
    message_id = notifier.send_schema_card_with_message_id(card=card)
    if not message_id:
        print(
            "发送失败：请检查 FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_CHAT_ID",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"已发送 message_id={message_id}")


if __name__ == "__main__":
    main()
