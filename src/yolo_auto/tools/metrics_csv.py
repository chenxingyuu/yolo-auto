from __future__ import annotations

from typing import Any


def _cell_float(row: dict[str, Any], keys: list[str]) -> float:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def parse_training_row(row: dict[str, Any]) -> dict[str, Any]:
    epoch_raw = row.get("epoch", "0")
    try:
        epoch = int(float(epoch_raw))
    except (TypeError, ValueError):
        epoch = 0

    map50 = _cell_float(
        row,
        ["metrics/mAP50(B)", "metrics/mAP50", "mAP50(B)", "mAP50"],
    )
    map5095 = _cell_float(
        row,
        ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95(B)", "mAP50-95"],
    )
    precision = _cell_float(
        row,
        ["metrics/precision(B)", "metrics/precision", "precision(B)", "precision"],
    )
    recall = _cell_float(
        row,
        ["metrics/recall(B)", "metrics/recall", "recall(B)", "recall"],
    )
    loss = _cell_float(
        row,
        ["train/box_loss", "train/box_loss(B)", "box_loss"],
    )

    return {
        "epoch": epoch,
        "loss": loss,
        "map50": map50,
        "map5095": map5095,
        "precision": precision,
        "recall": recall,
    }


def metric_value_from_parsed(parsed: dict[str, Any], primary_key: str) -> float:
    key = primary_key.strip().lower().replace(" ", "")
    if key in parsed:
        return float(parsed[key])
    if key in ("map5095", "map50-95", "metrics/map50-95", "metrics/map5095"):
        return float(parsed["map5095"])
    if key in ("map50", "metrics/map50"):
        return float(parsed["map50"])
    if key == "precision":
        return float(parsed["precision"])
    if key == "recall":
        return float(parsed["recall"])
    if key == "loss":
        return float(parsed["loss"])
    return float(parsed.get("map5095", 0.0))
