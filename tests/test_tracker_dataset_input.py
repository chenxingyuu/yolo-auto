from __future__ import annotations

from types import SimpleNamespace

import yolo_auto.tracker as tracker_mod
from yolo_auto.tracker import MLflowTracker, TrackerConfig


def _fake_run(run_id: str = "r1") -> SimpleNamespace:
    return SimpleNamespace(info=SimpleNamespace(run_id=run_id))


def test_start_run_logs_dataset_input(monkeypatch) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(tracker_mod.mlflow, "set_tracking_uri", lambda _x: None)
    monkeypatch.setattr(tracker_mod.mlflow, "set_experiment", lambda _x: None)
    monkeypatch.setattr(tracker_mod.mlflow, "active_run", lambda: None)
    monkeypatch.setattr(tracker_mod.mlflow, "start_run", lambda run_name: _fake_run("run-1"))
    monkeypatch.setattr(tracker_mod.mlflow, "log_params", lambda _x: None)
    monkeypatch.setattr(tracker_mod.mlflow, "set_tag", lambda _k, _v: None)

    def fake_from_pandas(df, source=None, name=None):
        calls["source"] = source
        calls["name"] = name
        calls["df_path"] = str(df["data_config_path"].iloc[0])
        return "dataset-obj"

    monkeypatch.setattr(tracker_mod.mlflow.data, "from_pandas", fake_from_pandas)
    monkeypatch.setattr(
        tracker_mod.mlflow,
        "log_input",
        lambda dataset, context=None: calls.update(
            {"dataset": dataset, "context": context}
        ),
    )

    tracker = MLflowTracker(TrackerConfig("sqlite:///x.db", "exp"))
    run_id = tracker.start_run(
        job_id="job-1",
        config={"data_config_path": "/workspace/datasets/a/data.yaml", "epochs": 1},
    )

    assert run_id == "run-1"
    assert calls["source"] == "/workspace/datasets/a/data.yaml"
    assert calls["name"] == "yolo_dataset_config"
    assert calls["df_path"] == "/workspace/datasets/a/data.yaml"
    assert calls["dataset"] == "dataset-obj"
    assert calls["context"] == "training"


def test_start_run_dataset_input_failure_does_not_break(monkeypatch) -> None:
    monkeypatch.setattr(tracker_mod.mlflow, "set_tracking_uri", lambda _x: None)
    monkeypatch.setattr(tracker_mod.mlflow, "set_experiment", lambda _x: None)
    monkeypatch.setattr(tracker_mod.mlflow, "active_run", lambda: None)
    monkeypatch.setattr(tracker_mod.mlflow, "start_run", lambda run_name: _fake_run("run-2"))
    monkeypatch.setattr(tracker_mod.mlflow, "log_params", lambda _x: None)
    monkeypatch.setattr(tracker_mod.mlflow, "set_tag", lambda _k, _v: None)
    monkeypatch.setattr(
        tracker_mod.mlflow.data,
        "from_pandas",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(tracker_mod.mlflow, "log_input", lambda *_args, **_kwargs: None)

    tracker = MLflowTracker(TrackerConfig("sqlite:///x.db", "exp"))
    run_id = tracker.start_run(
        job_id="job-2",
        config={"data_config_path": "/workspace/datasets/b/data.yaml"},
    )

    assert run_id == "run-2"
