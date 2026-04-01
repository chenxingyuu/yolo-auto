from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from yolo_auto.state_store import JobStateStore


@pytest.fixture
def mock_ssh() -> MagicMock:
    ssh = MagicMock()
    # 默认：cat/tail 都返回空且成功，进程存活
    ssh.execute.return_value = ("", "", 0)
    ssh.execute_background.return_value = ("0", 0)
    ssh.process_alive.return_value = True
    ssh.tail_file.return_value = ("", "", 0)
    ssh.file_exists.return_value = True
    return ssh


@pytest.fixture
def mock_notifier() -> MagicMock:
    notifier = MagicMock()
    notifier.send_training_update.return_value = None
    return notifier


@pytest.fixture
def mock_tracker() -> MagicMock:
    tracker = MagicMock()
    tracker.start_run.return_value = "run-1"
    tracker.log_epoch.return_value = None
    tracker.finish_run.return_value = None
    tracker.kill_run.return_value = None
    return tracker


@pytest.fixture
def state_store(tmp_path: Path) -> JobStateStore:
    return JobStateStore(str(tmp_path / "jobs.db"))

