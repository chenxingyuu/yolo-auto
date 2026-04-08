from __future__ import annotations

from types import SimpleNamespace

import yolo_auto.tracker as tracker_mod
from yolo_auto.tracker import MLflowTracker, TrackerConfig


def test_register_model_from_run_sets_candidate_alias(monkeypatch) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(tracker_mod.mlflow, "set_tracking_uri", lambda _x: None)
    monkeypatch.setattr(tracker_mod.mlflow, "set_experiment", lambda _x: None)
    monkeypatch.setattr(
        tracker_mod.MlflowClient,
        "create_registered_model",
        lambda self, name: calls.update({"name": name}) or None,
    )
    monkeypatch.setattr(
        tracker_mod.MlflowClient,
        "create_model_version",
        lambda self, name, source, run_id: calls.update(
            {"uri": source, "name": name, "run_id": run_id}
        )
        or SimpleNamespace(version="7"),
    )
    monkeypatch.setattr(tracker_mod.MlflowClient, "__init__", lambda self: None)
    monkeypatch.setattr(
        tracker_mod.MlflowClient,
        "set_registered_model_alias",
        lambda self, name, alias, version: calls.update(
            {"alias_name": name, "alias": alias, "alias_version": version}
        ),
    )
    monkeypatch.setattr(
        tracker_mod.MlflowClient,
        "set_model_version_tag",
        lambda self, name, version, key, value: None,
    )
    monkeypatch.setattr(
        tracker_mod.MlflowClient,
        "update_model_version",
        lambda self, name, version, description: None,
    )

    tracker = MLflowTracker(
        TrackerConfig(
            tracking_uri="sqlite:///x.db",
            experiment_name="exp",
            model_registry_enable=True,
        )
    )
    out = tracker.register_model_from_run(
        run_id="r1",
        model_name="detector-a",
        artifact_subpath="best.pt",
        tags={"k": "v"},
    )

    assert out["ok"] is True
    assert calls["uri"] == "runs:/r1/best.pt"
    assert calls["name"] == "detector-a"
    assert calls["alias_name"] == "detector-a"
    assert calls["alias"] == "candidate"
    assert calls["alias_version"] == "7"


def test_model_name_template_render() -> None:
    tracker = MLflowTracker(
        TrackerConfig(
            tracking_uri="sqlite:///x.db",
            experiment_name="exp",
            model_registry_enable=False,
            model_name_template="yolo-{env}-{data}-{model}",
        )
    )
    name = tracker.build_model_name(env_id="prod", data_key="coco", model_key="yolo11n")
    assert name == "yolo-prod-coco-yolo11n"
