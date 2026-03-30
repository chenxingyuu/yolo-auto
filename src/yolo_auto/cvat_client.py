from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from cvat_sdk.api_client.model_utils import to_json
from cvat_sdk.core.client import AccessTokenCredentials, Client
from cvat_sdk.core.helpers import expect_status, make_request_headers
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
        format_name: str = "Ultralytics YOLO Detection 1.0",
        include_images: bool = False,
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
        format_name: str = "Ultralytics YOLO Detection 1.0",
        include_images: bool = False,
    ) -> str:
        """发起云导出（仅 POST，返回 rq_id）；不阻塞，请用 get_request / list_requests 轮询。"""
        with self._build_client() as client:
            task = client.tasks.retrieve(task_id)
            return self._initiate_cloud_dataset_export(
                client,
                task.api.create_dataset_export_endpoint,
                int(task.id),
                filename=filename,
                cloud_storage_id=cloud_storage_id,
                format_name=format_name,
                include_images=include_images,
            )

    def export_project_dataset(
        self,
        project_id: int,
        format_name: str = "Ultralytics YOLO Detection 1.0",
        include_images: bool = False,
    ) -> bytes:
        with self._build_client() as client:
            project = client.projects.retrieve(project_id)
            return self._export_entity_dataset(project, format_name, include_images=include_images)

    def export_project_dataset_to_cloud(
        self,
        project_id: int,
        *,
        filename: str,
        cloud_storage_id: int,
        format_name: str = "Ultralytics YOLO Detection 1.0",
        include_images: bool = False,
    ) -> str:
        """发起项目云导出（仅 POST，返回 rq_id）；不阻塞等待队列。"""
        with self._build_client() as client:
            project = client.projects.retrieve(project_id)
            return self._initiate_cloud_dataset_export(
                client,
                project.api.create_dataset_export_endpoint,
                int(project.id),
                filename=filename,
                cloud_storage_id=cloud_storage_id,
                format_name=format_name,
                include_images=include_images,
            )

    def get_request(self, rq_id: str) -> dict[str, Any]:
        """GET /api/requests/{id}，返回与 CVAT 队列一致的结构（status/progress/message 等）。"""
        with self._build_client() as client:
            request, _ = client.api_client.requests_api.retrieve(rq_id)
            return to_json(request)

    def list_requests(
        self,
        *,
        project_id: int | None = None,
        task_id: int | None = None,
        status: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        target: str | None = None,
        subresource: str | None = None,
        action: str | None = None,
    ) -> dict[str, Any]:
        """GET /api/requests，用于浏览导出队列（可过滤 project_id、task_id、status 等）。"""
        params: dict[str, Any] = {}
        if project_id is not None:
            params["project_id"] = project_id
        if task_id is not None:
            params["task_id"] = task_id
        if status is not None:
            params["status"] = status
        if page is not None:
            params["page"] = page
        if page_size is not None:
            params["page_size"] = page_size
        if target is not None:
            params["target"] = target
        if subresource is not None:
            params["subresource"] = subresource
        if action is not None:
            params["action"] = action

        with self._build_client() as client:
            data, _ = client.api_client.requests_api.list(**params)
            return {
                "count": data.count,
                "next": data.next,
                "previous": data.previous,
                "results": [to_json(r) for r in data.results],
            }

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
                location=Location.LOCAL,
            )
            return tmp_path.read_bytes()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _initiate_cloud_dataset_export(
        client: Client,
        endpoint: Any,
        entity_id: int,
        *,
        filename: str,
        cloud_storage_id: int,
        format_name: str,
        include_images: bool,
    ) -> str:
        query_params: dict[str, Any] = {
            "location": Location.CLOUD_STORAGE,
            "cloud_storage_id": cloud_storage_id,
            "filename": str(filename),
            "format": format_name,
            "save_images": include_images,
        }
        url = client.api_map.make_endpoint_url(
            endpoint.path, kwsub={"id": entity_id}, query_params=query_params
        )
        response = client.api_client.rest_client.request(
            method=endpoint.settings["http_method"],
            url=url,
            headers=make_request_headers(client.api_client),
        )
        expect_status(202, response)
        raw = response.data
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        rq_id = payload.get("rq_id")
        if not rq_id:
            raise ValueError("CVAT export response missing rq_id")
        return str(rq_id)

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
