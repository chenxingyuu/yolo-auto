from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.STOPPED},
    JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.STOPPED: set(),
}


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    if current == target:
        return True
    return target in ALLOWED_TRANSITIONS[current]


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    run_id: str
    status: JobStatus
    pid: str
    paths: dict[str, str]
    created_at: int
    updated_at: int
    env_id: str = "default"
    last_notified_state: JobStatus | None = None
    last_metrics_at: int | None = None
    train_epochs: int | None = None
    last_reported_epoch: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "runId": self.run_id,
            "status": self.status.value,
            "pid": self.pid,
            "paths": self.paths,
            "envId": self.env_id,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "lastNotifiedState": (
                self.last_notified_state.value if self.last_notified_state else None
            ),
            "lastMetricsAt": self.last_metrics_at,
            "trainEpochs": self.train_epochs,
            "lastReportedEpoch": self.last_reported_epoch,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> JobRecord:
        last_notified = data.get("lastNotifiedState")
        train_epochs_raw = data.get("trainEpochs")
        return JobRecord(
            job_id=str(data["jobId"]),
            run_id=str(data["runId"]),
            status=JobStatus(str(data["status"])),
            pid=str(data.get("pid", "")),
            paths=dict(data.get("paths", {})),
            env_id=str(data.get("envId", "default")),
            created_at=int(data["createdAt"]),
            updated_at=int(data["updatedAt"]),
            last_notified_state=JobStatus(last_notified) if last_notified else None,
            last_metrics_at=int(data["lastMetricsAt"]) if data.get("lastMetricsAt") else None,
            train_epochs=int(train_epochs_raw) if train_epochs_raw is not None else None,
            last_reported_epoch=int(data.get("lastReportedEpoch", 0)),
        )

