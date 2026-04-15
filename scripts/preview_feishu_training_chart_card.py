#!/usr/bin/env python3
# ruff: noqa: E402
"""向飞书群依次发送三张训练卡片预览：启动 → 里程碑 → 完成。

仅用于本地预览。请在项目根目录配置 `.env`（FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_CHAT_ID 等）。

用法::

    uv run python scripts/preview_feishu_training_chart_card.py
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

from yolo_auto.config import load_settings
from yolo_auto.feishu import FeishuNotifier
from yolo_auto.tools.status import (
    _build_training_schema_card,
    build_training_started_schema_card,
)

# 合成训练历史数据（20 epochs）
_ROWS: list[dict[str, str]] = []
for _ep in range(1, 21):
    _ROWS.append({
        "epoch": str(_ep),
        "train/box_loss": f"{1.8 - _ep * 0.06:.4f}",
        "metrics/mAP50(B)": f"{0.05 + _ep * 0.035:.4f}",
        "metrics/mAP50-95(B)": f"{0.03 + _ep * 0.025:.4f}",
        "metrics/recall(B)": f"{0.10 + _ep * 0.03:.4f}",
    })

def _send(notifier: FeishuNotifier, card: dict, label: str) -> str | None:
    msg_id = notifier.send_schema_card_with_message_id(card=card)
    if msg_id:
        print(f"[{label}] 已发送  message_id={msg_id}")
    else:
        print(f"[{label}] 发送失败", file=sys.stderr)
    return msg_id


def main() -> None:
    settings = load_settings()
    notifier = FeishuNotifier(
        webhook_url=settings.feishu_webhook_url,
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
        chat_id=settings.feishu_chat_id,
    )

    now_text = time.strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. 训练启动卡片 ──────────────────────────────────────────────────────
    start_card = build_training_started_schema_card(
        epochs=100,
        imgsz=640,
        batch_display="16",
        lr_display="0.01",
        optimizer_display="SGD",
        device_display="0",
        workers_display="8",
        fourth_metric_label="Patience",
        fourth_metric_value="50",
        extra_params_line="其他：augment=True，mosaic=1.0",
        started_time_text=now_text,
        mlflow_url=settings.mlflow_external_url,
        top_img_key=settings.feishu_card_img_key,
        top_img_fallback_key=settings.feishu_card_fallback_img_key,
        title="[YOLO] 训练已启动 · job-preview",
    )
    _send(notifier, start_card, "训练启动")

    time.sleep(1)

    # ── 2. 里程碑卡片（epoch 20/100，带折线图）────────────────────────────────
    milestone_card = _build_training_schema_card(
        title="[YOLO] 训练进行中 · job-preview",
        header_template="blue",
        epoch_text="20/100",
        elapsed_text="12m30s",
        eta_text="50m00s",
        updated_time_text=now_text,
        mlflow_url=settings.mlflow_external_url,
        top_img_key=settings.feishu_card_img_key,
        top_img_fallback_key=settings.feishu_card_fallback_img_key,
        training_rows=_ROWS,
    )
    _send(notifier, milestone_card, "里程碑")

    time.sleep(1)

    # ── 3. 训练完成卡片（绿色，带折线图）────────────────────────────────────
    completed_card = _build_training_schema_card(
        title="[YOLO] 训练已完成 · job-preview",
        header_template="green",
        epoch_text="100/100",
        elapsed_text="1h02m",
        eta_text="—",
        updated_time_text=now_text,
        mlflow_url=settings.mlflow_external_url,
        top_img_key=settings.feishu_card_img_key,
        top_img_fallback_key=settings.feishu_card_fallback_img_key,
        training_rows=_ROWS,
    )
    _send(notifier, completed_card, "训练完成")


if __name__ == "__main__":
    main()
