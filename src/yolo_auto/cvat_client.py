from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from cvat_sdk.core.client import AccessTokenCredentials, Client
from cvat_sdk.core.proxies.types import Location


@dataclass(frozen=True)
class CVATConfig:
    url: str
    token: str
    org_slug: str | None = None


class CVATClient:
    def __init__(self, config: CVATConfig) -> None:
        self._config = config

    def list_projects(self) -> list[dict[str, Any]]:
        with self._build_client() as client:
            projects = client.projects.list()
            return [self._project_summary(p) for p in projects]

    def list_tasks(self, project_id: int | None = None) -> list[dict[str, Any]]:
        with self._build_client() as client:
            tasks = client.tasks.list()
            items: list[dict[str, Any]] = []
            for task in tasks:
                task_project_id = int(getattr(task, "project_id", 0) or 0)
                if project_id is not None and task_project_id != project_id:
                    continue
                items.append(self._task_summary(task))
            return items

    def list_formats(self) -> dict[str, list[dict[str, Any]]]:
        with self._build_client() as client:
            payload, _ = client.api_client.server_api.retrieve_annotation_formats()
            exporters = list(getattr(payload, "exporters", []) or [])
            importers = list(getattr(payload, "importers", []) or [])
            return {
                "exporters": [self._format_item(fmt) for fmt in exporters],
                "importers": [self._format_item(fmt) for fmt in importers],
            }

    def get_task_details(self, task_id: int) -> dict[str, Any]:
        with self._build_client() as client:
            task = client.tasks.retrieve(task_id)
            return self._task_detail(task)

    def analyze_task(self, task_id: int) -> dict[str, Any]:
        with self._build_client() as client:
            task = client.tasks.retrieve(task_id)
            annotations = task.get_annotations()
            labels = task.get_labels()
            meta = task.get_meta()

            label_name_by_id = {
                int(getattr(label, "id", -1)): str(getattr(label, "name", "unknown"))
                for label in labels
            }
            counts_by_label: dict[str, int] = {}

            tags = list(getattr(annotations, "tags", []) or [])
            shapes = list(getattr(annotations, "shapes", []) or [])
            tracks = list(getattr(annotations, "tracks", []) or [])

            for item in [*tags, *shapes]:
                label_id = int(getattr(item, "label_id", -1))
                label_name = label_name_by_id.get(label_id, f"label_{label_id}")
                counts_by_label[label_name] = counts_by_label.get(label_name, 0) + 1

            for track in tracks:
                track_label_id = int(getattr(track, "label_id", -1))
                track_label = label_name_by_id.get(track_label_id, f"label_{track_label_id}")
                track_shapes = list(getattr(track, "shapes", []) or [])
                counts_by_label[track_label] = (
                    counts_by_label.get(track_label, 0) + len(track_shapes)
                )

            total_annotations = sum(counts_by_label.values())
            frames = list(getattr(meta, "frames", []) or [])
            image_count = len(frames)
            avg_per_image = (
                round(total_annotations / image_count, 4) if image_count > 0 else 0.0
            )

            distribution: list[dict[str, Any]] = []
            sorted_items = sorted(
                counts_by_label.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for label_name, count in sorted_items:
                ratio = (
                    round((count / total_annotations) * 100, 2)
                    if total_annotations > 0
                    else 0.0
                )
                distribution.append({"label": label_name, "count": count, "ratioPercent": ratio})

            return {
                "task": self._task_detail(task),
                "stats": {
                    "imageCount": image_count,
                    "annotationCount": total_annotations,
                    "avgAnnotationsPerImage": avg_per_image,
                    "labelCount": len(distribution),
                    "distribution": distribution,
                },
            }

    def export_task_dataset(
        self,
        task_id: int,
        format_name: str = "YOLO 1.1",
        include_images: bool = True,
        *,
        status_check_period: int | None = None,
    ) -> bytes:
        with self._build_client() as client:
            task = client.tasks.retrieve(task_id)
            return self._export_entity_dataset(
                task,
                format_name,
                include_images=include_images,
                status_check_period=status_check_period,
            )

    def export_task_dataset_to_cloud(
        self,
        task_id: int,
        *,
        filename: str,
        cloud_storage_id: int,
        format_name: str = "YOLO 1.1",
        include_images: bool = True,
        status_check_period: int | None = None,
    ) -> None:
        with self._build_client() as client:
            task = client.tasks.retrieve(task_id)
            task.export_dataset(
                format_name,
                filename,
                include_images=include_images,
                status_check_period=status_check_period,
                location=Location.CLOUD_STORAGE,
                cloud_storage_id=cloud_storage_id,
            )

    def export_project_dataset(
        self,
        project_id: int,
        format_name: str = "YOLO 1.1",
        include_images: bool = True,
    ) -> bytes:
        with self._build_client() as client:
            project = client.projects.retrieve(project_id)
            return self._export_entity_dataset(project, format_name, include_images=include_images)

    def _build_client(self) -> Client:
        client = Client(self._config.url, check_server_version=False)
        client.login(AccessTokenCredentials(token=self._config.token))
        if self._config.org_slug:
            client.organization_slug = self._config.org_slug
        return client

    def _export_entity_dataset(
        self,
        entity: Any,
        format_name: str,
        *,
        include_images: bool,
        status_check_period: int | None = None,
    ) -> bytes:
        with NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            entity.export_dataset(
                format_name,
                str(tmp_path),
                include_images=include_images,
                status_check_period=status_check_period,
            )
            return tmp_path.read_bytes()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _project_summary(project: Any) -> dict[str, Any]:
        return {
            "id": int(project.id),
            "name": str(getattr(project, "name", "")),
            "taskCount": int(getattr(project, "task_count", 0) or 0),
            "status": str(getattr(project, "status", "")),
            "updatedDate": str(getattr(project, "updated_date", "")),
        }

    @staticmethod
    def _task_summary(task: Any) -> dict[str, Any]:
        return {
            "id": int(task.id),
            "name": str(getattr(task, "name", "")),
            "projectId": int(getattr(task, "project_id", 0) or 0),
            "size": int(getattr(task, "size", 0) or 0),
            "mode": str(getattr(task, "mode", "")),
            "status": str(getattr(task, "status", "")),
            "subset": str(getattr(task, "subset", "")),
            "updatedDate": str(getattr(task, "updated_date", "")),
        }

    def _task_detail(self, task: Any) -> dict[str, Any]:
        labels = task.get_labels()
        return {
            **self._task_summary(task),
            "labels": [
                {
                    "id": int(getattr(label, "id", -1)),
                    "name": str(getattr(label, "name", "")),
                    "type": str(getattr(label, "type", "")),
                }
                for label in labels
            ],
            "url": f"{self._config.url.rstrip('/')}/tasks/{task.id}",
        }

    @staticmethod
    def _format_item(item: Any) -> dict[str, Any]:
        return {
            "name": str(getattr(item, "name", "")),
            "displayName": str(getattr(item, "display_name", "")),
            "enabled": bool(getattr(item, "enabled", True)),
            "version": str(getattr(item, "version", "")),
            "ext": str(getattr(item, "ext", "")),
        }
