from __future__ import annotations

import csv
from io import StringIO

from yolo_auto.ssh_client import SSHClient
from yolo_auto.tracker import MLflowTracker


def get_status(
    job_id: str,
    run_id: str,
    jobs_dir: str,
    ssh_client: SSHClient,
    tracker: MLflowTracker,
) -> dict[str, object]:
    metrics_path = f"{jobs_dir}/{job_id}/results.csv"
    content, stderr_text, exit_code = ssh_client.execute(f"cat {metrics_path}")
    if exit_code != 0:
        return {
            "jobId": job_id,
            "status": "running",
            "progress": 0.0,
            "error": stderr_text.strip() or "results.csv not ready",
        }
    rows = list(csv.DictReader(StringIO(content)))
    if not rows:
        return {"jobId": job_id, "status": "running", "progress": 0.0}

    last_row = rows[-1]
    epoch = int(float(last_row.get("epoch", "0")))
    map50 = float(last_row.get("metrics/mAP50(B)", "0"))
    map5095 = float(last_row.get("metrics/mAP50-95(B)", "0"))
    precision = float(last_row.get("metrics/precision(B)", "0"))
    recall = float(last_row.get("metrics/recall(B)", "0"))
    loss = float(last_row.get("train/box_loss", "0"))

    tracker.log_epoch(
        run_id=run_id,
        metrics={
            "loss": loss,
            "map50": map50,
            "map5095": map5095,
            "precision": precision,
            "recall": recall,
        },
        step=epoch,
    )

    return {
        "jobId": job_id,
        "status": "running",
        "progress": round(min(1.0, epoch / 100), 4),
        "metrics": {
            "epoch": epoch,
            "loss": loss,
            "map50": map50,
            "map5095": map5095,
            "precision": precision,
            "recall": recall,
        },
        "artifacts": {
            "best": f"{jobs_dir}/{job_id}/weights/best.pt",
            "last": f"{jobs_dir}/{job_id}/weights/last.pt",
        },
    }

