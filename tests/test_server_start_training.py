from __future__ import annotations


def test_yolo_start_training_empty_call_returns_structured_err() -> None:
    """空参数 {} 时返回业务错误，而非 Pydantic 多条校验异常。"""
    from yolo_auto.server import yolo_start_training

    out = yolo_start_training()
    assert out["ok"] is False
    assert out["errorCode"] == "MISSING_ARGUMENTS"
    assert out["missing"] == [
        "model",
        "dataConfigPath",
        "epochs",
        "imgSize",
        "batch",
    ]
