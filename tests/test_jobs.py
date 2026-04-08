from __future__ import annotations

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.tools.jobs import delete_job, get_job


def _upsert(state_store, job_id: str, status: JobStatus) -> None:
    state_store.upsert(
        JobRecord(
            job_id=job_id,
            run_id="run-1",
            status=status,
            pid="1",
            paths={},
            created_at=1,
            updated_at=1,
        )
    )


def test_delete_job_success(state_store) -> None:
    _upsert(state_store, "job-done", JobStatus.COMPLETED)
    result = delete_job("job-done", state_store)
    assert result["ok"] is True
    assert result["deleted"] is True
    assert state_store.get("job-done") is None


def test_delete_job_reject_running(state_store) -> None:
    _upsert(state_store, "job-running", JobStatus.RUNNING)
    result = delete_job("job-running", state_store)
    assert result["ok"] is False
    assert result["errorCode"] == "JOB_DELETE_FORBIDDEN"


def test_delete_job_not_found(state_store) -> None:
    result = delete_job("missing-job", state_store)
    assert result["ok"] is False
    assert result["errorCode"] == "JOB_NOT_FOUND"


def test_get_job_refresh_parses_args_yaml_and_persists_train_params(
    state_store,
    mock_ssh,
    mock_notifier,
    monkeypatch,
) -> None:
    state_store.upsert(
        JobRecord(
            job_id="job-args",
            run_id="run-args",
            status=JobStatus.RUNNING,
            pid="1",
            paths={"jobDir": "/remote/jobs/job-args"},
            created_at=1,
            updated_at=1,
        )
    )

    def _fake_get_status(*_args, **_kwargs):
        return {"ok": True}

    monkeypatch.setattr("yolo_auto.tools.jobs.get_status", _fake_get_status)

    def _exec(cmd: str, *args, **kwargs):
        if "args.yaml" in cmd and cmd.strip().startswith("cat "):
            return (
                "imgsz: 640\nbatch: 16\noptimizer: SGD\nlr0: 0.01\n",
                "",
                0,
            )
        return ("", "", 0)

    mock_ssh.execute.side_effect = _exec

    result = get_job(
        "job-args",
        state_store,
        {"default": mock_ssh},
        mock_notifier,
        refresh=True,
    )
    assert result["ok"] is True
    stored = state_store.get("job-args")
    assert stored is not None
    assert stored.train_params == {
        "imgsz": 640,
        "batch": 16,
        "optimizer": "SGD",
        "lr0": 0.01,
    }
    assert result["record"]["trainParams"] == stored.train_params


def test_get_job_refresh_missing_args_yaml_does_not_error(
    state_store,
    mock_ssh,
    mock_notifier,
    monkeypatch,
) -> None:
    state_store.upsert(
        JobRecord(
            job_id="job-noargs",
            run_id="run-noargs",
            status=JobStatus.RUNNING,
            pid="1",
            paths={"jobDir": "/remote/jobs/job-noargs"},
            created_at=1,
            updated_at=1,
        )
    )

    monkeypatch.setattr("yolo_auto.tools.jobs.get_status", lambda *_a, **_k: {"ok": True})

    def _exec(cmd: str, *args, **kwargs):
        if "args.yaml" in cmd and cmd.strip().startswith("cat "):
            return ("", "No such file", 1)
        return ("", "", 0)

    mock_ssh.execute.side_effect = _exec

    result = get_job(
        "job-noargs",
        state_store,
        {"default": mock_ssh},
        mock_notifier,
        refresh=True,
    )
    assert result["ok"] is True
    stored = state_store.get("job-noargs")
    assert stored is not None
    assert stored.train_params is None


def test_get_job_refresh_invalid_yaml_does_not_persist_train_params(
    state_store,
    mock_ssh,
    mock_notifier,
    monkeypatch,
) -> None:
    state_store.upsert(
        JobRecord(
            job_id="job-badyaml",
            run_id="run-badyaml",
            status=JobStatus.RUNNING,
            pid="1",
            paths={"jobDir": "/remote/jobs/job-badyaml"},
            created_at=1,
            updated_at=1,
        )
    )

    monkeypatch.setattr("yolo_auto.tools.jobs.get_status", lambda *_a, **_k: {"ok": True})

    def _exec(cmd: str, *args, **kwargs):
        if "args.yaml" in cmd and cmd.strip().startswith("cat "):
            return ("a: [1, 2\n", "", 0)  # YAML parse error
        return ("", "", 0)

    mock_ssh.execute.side_effect = _exec

    result = get_job(
        "job-badyaml",
        state_store,
        {"default": mock_ssh},
        mock_notifier,
        refresh=True,
    )
    assert result["ok"] is True
    stored = state_store.get("job-badyaml")
    assert stored is not None
    assert stored.train_params is None
