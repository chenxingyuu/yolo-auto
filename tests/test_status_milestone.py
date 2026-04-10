from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from yolo_auto.models import JobRecord, JobStatus
from yolo_auto.state_store import JobStateStore
from yolo_auto.tools.status import _build_progress_summary, get_status

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    job_id: str = "job-1",
    *,
    status: JobStatus = JobStatus.RUNNING,
    last_reported_epoch: int = 0,
    train_epochs: int = 100,
    feishu_message_id: str | None = None,
) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        run_id=job_id,
        status=status,
        pid="1234",
        paths={
            "metricsPath": "/workspace/jobs/job-1/results.csv",
            "logPath": "/workspace/jobs/job-1/train.log",
            "bestPath": "/workspace/jobs/job-1/weights/best.pt",
            "lastPath": "/workspace/jobs/job-1/weights/last.pt",
        },
        created_at=1000,
        updated_at=1000,
        train_epochs=train_epochs,
        last_reported_epoch=last_reported_epoch,
        feishu_message_id=feishu_message_id,
    )


def _make_remote_response(
    *,
    status: str = "running",
    epoch: int = 50,
    total_epochs: int = 100,
    elapsed: int = 3600,
    training_rows: list | None = None,
) -> dict:
    progress = round(epoch / total_epochs, 4) if total_epochs else 0.0
    return {
        "status": status,
        "processAlive": status == "running",
        "progress": progress,
        "metrics": {
            "epoch": epoch,
            "map50": 0.750,
            "map5095": 0.512,
            "precision": 0.800,
            "recall": 0.700,
            "loss": 1.23,
        },
        "trainingRows": training_rows or [],
        "elapsedSeconds": elapsed,
        "logTail": "",
    }


def _make_training_rows(n: int = 5) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "epoch": str(i),
            "metrics/mAP50(B)": str(round(0.1 * i, 3)),
            "metrics/mAP50-95(B)": str(round(0.05 * i, 3)),
            "metrics/recall(B)": str(round(0.08 * i, 3)),
            "train/box_loss": "1.0",
        })
    return rows


# ---------------------------------------------------------------------------
# _build_progress_summary
# ---------------------------------------------------------------------------

class TestBuildProgressSummary:
    def test_contains_epoch_fraction(self):
        s = _build_progress_summary(
            current_epoch=50, total_epochs=100,
            elapsed_seconds=3600, metrics={"map5095": 0.512},
            primary_metric_key="map5095",
        )
        assert "50/100" in s

    def test_contains_metric_value(self):
        s = _build_progress_summary(
            current_epoch=50, total_epochs=100,
            elapsed_seconds=3600, metrics={"map5095": 0.654},
            primary_metric_key="map5095",
        )
        assert "0.654" in s

    def test_contains_eta_when_epoch_is_partial(self):
        s = _build_progress_summary(
            current_epoch=50, total_epochs=100,
            elapsed_seconds=3600, metrics={"map5095": 0.5},
            primary_metric_key="map5095",
        )
        assert "ETA" in s

    def test_no_eta_when_epoch_zero(self):
        s = _build_progress_summary(
            current_epoch=0, total_epochs=100,
            elapsed_seconds=0, metrics={},
            primary_metric_key="map5095",
        )
        assert "ETA" not in s

    def test_no_eta_when_training_complete(self):
        s = _build_progress_summary(
            current_epoch=100, total_epochs=100,
            elapsed_seconds=7200, metrics={"map5095": 0.8},
            primary_metric_key="map5095",
        )
        assert "ETA" not in s

    def test_progress_bar_present(self):
        s = _build_progress_summary(
            current_epoch=50, total_epochs=100,
            elapsed_seconds=1800, metrics={"map5095": 0.5},
            primary_metric_key="map5095",
        )
        assert "[" in s and "]" in s
        assert "█" in s
        assert "░" in s

    def test_full_bar_at_100_percent(self):
        s = _build_progress_summary(
            current_epoch=100, total_epochs=100,
            elapsed_seconds=7200, metrics={"map5095": 0.8},
            primary_metric_key="map5095",
        )
        assert "░" not in s


# ---------------------------------------------------------------------------
# get_status — progressSummary in response
# ---------------------------------------------------------------------------

class TestGetStatusProgressSummary:
    def test_progress_summary_in_response(self, state_store: JobStateStore):
        record = _make_record(last_reported_epoch=0)
        state_store.upsert(record)

        client = MagicMock()
        # epoch=3 低于默认 every_n=5，不触发里程碑，notifier 不会被调用
        client.get_training_status.return_value = _make_remote_response(epoch=3)

        result = get_status(
            "job-1", "job-1", state_store, None, MagicMock(),
            control_client=client,
            feishu_report_every_n_epochs=5,
        )

        assert result["ok"] is True
        assert "progressSummary" in result
        assert "3/100" in result["progressSummary"]

    def test_progress_summary_contains_metric(self, state_store: JobStateStore):
        record = _make_record(last_reported_epoch=100)  # 已报告，不再触发
        state_store.upsert(record)

        client = MagicMock()
        client.get_training_status.return_value = _make_remote_response(epoch=30)

        result = get_status(
            "job-1", "job-1", state_store, None, MagicMock(),
            control_client=client, primary_metric_key="map5095",
        )

        assert "map5095" in result["progressSummary"]


# ---------------------------------------------------------------------------
# get_status — milestone trigger
# ---------------------------------------------------------------------------

class TestMilestoneTrigger:
    def test_milestone_fires_when_epoch_threshold_met(self, state_store: JobStateStore):
        record = _make_record(last_reported_epoch=0)
        state_store.upsert(record)

        notifier = MagicMock()
        notifier.send_schema_card_with_message_id.return_value = "msg-001"
        client = MagicMock()
        client.get_training_status.return_value = _make_remote_response(
            epoch=5, training_rows=_make_training_rows(5)
        )

        get_status(
            "job-1", "job-1", state_store, None, notifier,
            control_client=client,
            feishu_report_enable=True,
            feishu_report_every_n_epochs=5,
        )

        notifier.send_schema_card_with_message_id.assert_called_once()

    def test_milestone_updates_last_reported_epoch(self, state_store: JobStateStore):
        record = _make_record(last_reported_epoch=0)
        state_store.upsert(record)

        notifier = MagicMock()
        notifier.send_schema_card_with_message_id.return_value = "msg-001"
        client = MagicMock()
        client.get_training_status.return_value = _make_remote_response(epoch=10)

        get_status(
            "job-1", "job-1", state_store, None, notifier,
            control_client=client,
            feishu_report_enable=True,
            feishu_report_every_n_epochs=5,
        )

        updated = state_store.get("job-1")
        assert updated is not None
        assert updated.last_reported_epoch == 10

    def test_milestone_not_fired_when_epoch_below_threshold(self, state_store: JobStateStore):
        record = _make_record(last_reported_epoch=0)
        state_store.upsert(record)

        notifier = MagicMock()
        client = MagicMock()
        client.get_training_status.return_value = _make_remote_response(epoch=3)

        get_status(
            "job-1", "job-1", state_store, None, notifier,
            control_client=client,
            feishu_report_enable=True,
            feishu_report_every_n_epochs=5,
        )

        notifier.send_schema_card_with_message_id.assert_not_called()

    def test_milestone_not_fired_when_already_reported(self, state_store: JobStateStore):
        record = _make_record(last_reported_epoch=10)
        state_store.upsert(record)

        notifier = MagicMock()
        client = MagicMock()
        # epoch=12, last_reported=10, every_n=5 → 12 < 10+5=15, no trigger
        client.get_training_status.return_value = _make_remote_response(epoch=12)

        get_status(
            "job-1", "job-1", state_store, None, notifier,
            control_client=client,
            feishu_report_enable=True,
            feishu_report_every_n_epochs=5,
        )

        notifier.send_schema_card_with_message_id.assert_not_called()

    def test_milestone_not_fired_when_report_disabled(self, state_store: JobStateStore):
        record = _make_record(last_reported_epoch=0)
        state_store.upsert(record)

        notifier = MagicMock()
        client = MagicMock()
        client.get_training_status.return_value = _make_remote_response(epoch=50)

        get_status(
            "job-1", "job-1", state_store, None, notifier,
            control_client=client,
            feishu_report_enable=False,
            feishu_report_every_n_epochs=5,
        )

        notifier.send_schema_card_with_message_id.assert_not_called()

    def test_milestone_not_fired_when_every_n_is_zero(self, state_store: JobStateStore):
        record = _make_record(last_reported_epoch=0)
        state_store.upsert(record)

        notifier = MagicMock()
        client = MagicMock()
        client.get_training_status.return_value = _make_remote_response(epoch=50)

        get_status(
            "job-1", "job-1", state_store, None, notifier,
            control_client=client,
            feishu_report_enable=True,
            feishu_report_every_n_epochs=0,
        )

        notifier.send_schema_card_with_message_id.assert_not_called()

    def test_milestone_not_fired_when_status_is_completed(self, state_store: JobStateStore):
        record = _make_record(last_reported_epoch=0)
        state_store.upsert(record)

        notifier = MagicMock()
        client = MagicMock()
        client.get_training_status.return_value = _make_remote_response(
            status="completed", epoch=100
        )

        get_status(
            "job-1", "job-1", state_store, None, notifier,
            control_client=client,
            feishu_report_enable=True,
            feishu_report_every_n_epochs=5,
        )

        notifier.send_schema_card_with_message_id.assert_not_called()

    def test_milestone_patches_existing_card(self, state_store: JobStateStore):
        record = _make_record(last_reported_epoch=0, feishu_message_id="existing-msg")
        state_store.upsert(record)

        notifier = MagicMock()
        notifier.update_schema_card.return_value = True  # patch 成功
        client = MagicMock()
        client.get_training_status.return_value = _make_remote_response(epoch=5)

        get_status(
            "job-1", "job-1", state_store, None, notifier,
            control_client=client,
            feishu_report_enable=True,
            feishu_report_every_n_epochs=5,
        )

        notifier.update_schema_card.assert_called_once_with(
            message_id="existing-msg", card=notifier.update_schema_card.call_args.kwargs["card"]
        )
        notifier.send_schema_card_with_message_id.assert_not_called()


# ---------------------------------------------------------------------------
# get_status — training_rows passed to card builder
# ---------------------------------------------------------------------------

class TestMilestoneCardContent:
    def test_card_includes_chart_when_training_rows_present(self, state_store: JobStateStore):
        record = _make_record(last_reported_epoch=0)
        state_store.upsert(record)

        notifier = MagicMock()
        notifier.send_schema_card_with_message_id.return_value = "msg-001"
        client = MagicMock()
        client.get_training_status.return_value = _make_remote_response(
            epoch=5, training_rows=_make_training_rows(5)
        )

        with patch(
            "yolo_auto.tools.status._build_training_schema_card",
            wraps=__import__(
                "yolo_auto.tools.status", fromlist=["_build_training_schema_card"]
            )._build_training_schema_card,
        ) as mock_build:
            get_status(
                "job-1", "job-1", state_store, None, notifier,
                control_client=client,
                feishu_report_enable=True,
                feishu_report_every_n_epochs=5,
            )

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert kwargs["training_rows"] is not None
        assert len(kwargs["training_rows"]) == 5


# ---------------------------------------------------------------------------
# control_api.py — trainingRows in status response
# ---------------------------------------------------------------------------

class TestControlApiTrainingRows:
    def test_training_rows_in_response(self, tmp_path: Path):
        from yolo_auto.control_api import train_status

        csv_content = (
            "epoch,metrics/mAP50(B),metrics/mAP50-95(B),metrics/recall(B),"
            "train/box_loss,train/cls_loss,train/dfl_loss\n"
            "1,0.1,0.05,0.08,1.0,0.5,0.3\n"
            "2,0.2,0.10,0.16,0.9,0.4,0.2\n"
            "3,0.3,0.15,0.24,0.8,0.3,0.1\n"
        )
        metrics_file = tmp_path / "results.csv"
        metrics_file.write_text(csv_content, encoding="utf-8")

        result = train_status(
            jobId="job-1",
            pid=None,
            metricsPath=str(metrics_file),
            logPath=None,
            totalEpochs=10,
            createdAt=None,
        )

        assert "trainingRows" in result
        assert len(result["trainingRows"]) == 3
        assert result["trainingRows"][-1]["epoch"] == "3"

    def test_training_rows_sampled_when_too_many(self, tmp_path: Path):
        from yolo_auto.control_api import train_status

        cols = "epoch,metrics/mAP50(B),metrics/mAP50-95(B),metrics/recall(B)"
        cols += ",train/box_loss,train/cls_loss,train/dfl_loss"
        header = cols + "\n"
        rows = [f"{i},0.1,0.05,0.08,1.0,0.5,0.3\n" for i in range(1, 201)]
        metrics_file = tmp_path / "results.csv"
        metrics_file.write_text(header + "".join(rows), encoding="utf-8")

        result = train_status(
            jobId="job-1",
            pid=None,
            metricsPath=str(metrics_file),
            logPath=None,
            totalEpochs=200,
            createdAt=None,
        )

        assert len(result["trainingRows"]) <= 100
        # 最后一行必须包含
        assert result["trainingRows"][-1]["epoch"] == "200"

    def test_training_rows_empty_when_no_metrics_file(self):
        from yolo_auto.control_api import train_status

        result = train_status(
            jobId="job-1",
            pid=None,
            metricsPath="/nonexistent/results.csv",
            logPath=None,
            totalEpochs=100,
            createdAt=None,
        )

        assert result["trainingRows"] == []
