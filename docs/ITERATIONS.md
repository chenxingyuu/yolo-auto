# 迭代记录

本文档按时间记录面向维护者与高级用户的迭代说明。**新条目追加在文件顶部**（越新越靠前）。

---

## 2026-04-14 — SAHI 切片支持（全景/高分辨率数据集）

### 目标摘要

全景摄像头输出的超宽图像（如 7680×1048，宽高比约 7.3:1）直接送入 YOLO 会被 letterbox 压缩为正方形，导致目标严重形变、检测精度下降。本次迭代引入 SAHI（Slicing Aided Hyper Inference）切片预处理工具，将大图切割为 640×640 的重叠瓦片并重新映射 bbox 标注，生成标准 YOLO 训练集后即可正常训练。

### 用户可见变更

| 区域 | 变更 |
|------|------|
| `yolo_sahi_slice`（新 MCP 工具） | 将源数据集按滑动窗口切片，输出新数据集并返回 `dataConfigPath`；支持配置瓦片尺寸、重叠比、最小 bbox 保留面积比。 |
| 标准工作流说明 | 在步骤 4（同步数据集）后插入步骤 4b，说明全景数据集的可选切片流程。 |
| `POST /api/v1/dataset/sahi-slice`（控制面新端点） | 训练容器侧实现：读取源 YAML → 滑动窗口切图 → YOLO bbox 坐标重映射 → 写出新数据集目录和 `data.yaml`。 |

### 推荐工作流（全景数据集）

```
yolo_sync_dataset
  → yolo_sahi_slice(dataConfigPath=..., outputDatasetName="pano-sliced")
  → yolo_check_dataset(dataConfigPath=<新路径>)
  → yolo_start_training(dataConfigPath=<新路径>, imgSize=640, ...)
```

### 切片参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sliceHeight` / `sliceWidth` | 640 | 瓦片尺寸，与 YOLO imgsz 对齐 |
| `overlapHeightRatio` / `overlapWidthRatio` | 0.2 | 相邻瓦片重叠比，防止边界漏检 |
| `minAreaRatio` | 0.1 | bbox 被裁剪后保留面积 < 原面积×此值时丢弃 |

7680×1048 图像使用默认参数，每帧约生成 **25 个瓦片**。

### 实现说明

- 切片与 bbox 重映射纯用 PIL（ultralytics 已包含），**控制面容器无需额外安装 `sahi` 包**。
- 每张切片输出为 `{原始stem}_{序号:04d}.jpg`，标注文件同名 `.txt`（YOLO 归一化格式）。
- `minAreaRatio` 过滤在切片边缘被大量裁切的 bbox，避免引入噪声样本。

### 兼容与迁移

- 纯新增，不修改现有工具与接口，旧工作流无需任何变更。
- 如不使用全景数据集，可完全跳过此步骤。

### 已知限制

- 当前仅处理 train / val / test 三个 split，不支持自定义 split 名。
- 切片后图像统一保存为 JPEG（quality=95），不保留原始格式。
- 推理侧 SAHI 分块推理 + NMS 合并不在本次范围内。

---

## 2026-04-08 — 数据血缘、导出 manifest、HTTP 错误码

### 目标摘要

收紧「MinIO/CVAT 导出 → 训练 → 验证 → 交付」可追溯性与可观测性：任务可关联数据来源；导出产物有机器可读清单；`get_status` 对 HTTP 控制面失败返回稳定错误码。

### 用户可见变更


| 区域                    | 变更                                                                                                                                                                                      |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yolo_sync_dataset`   | 成功时在 `payload` 中增加 `**provenance`**（bucket/prefix、`objectName`、`mcSourcePath`、`extractedDir`、`dataConfigPath`、`dataYamlRelative`、`datasetName` 等）。                                      |
| `yolo_start_training` | 可选 `**minioExportZip**`、`**datasetSlug**`、`**datasetVersionNote**`（支持 snake_case）；写入 `**JobRecord.datasetProvenance**` 及参数中的 `dataset_*` 前缀项。 |
| `yolo_export`         | 成功时在远程 `**{jobDir}/export-manifest.json**` 写入清单，返回体含 `**exportManifestPath**` 与内联 `**exportManifest**`（失败写入时含 `exportManifestWarning`）。                                                 |
| `yolo_get_status`     | 控制面连接类失败映射为 `**REMOTE_***` 系列 `errorCode`（如 `REMOTE_TIMEOUT`、`REMOTE_UNAUTHORIZED`、`REMOTE_UNREACHABLE`）及 `**retryable**` / `**hint**`。                                                         |


### 兼容与迁移

- **本地状态库**：`JobRecord` 增加可选字段 `**datasetProvenance`**；旧 `jobs.db` 无该字段时反序列化为 `null`，无需手工迁移。

### 推荐验证步骤

1. `yolo_sync_dataset` → 确认响应含 `provenance`。
2. `yolo_start_training` 传入与 `provenance` 对齐的可选血缘字段 → `yolo_get_job` 中可见 `datasetProvenance`。
3. `yolo_export` → 远程 `export-manifest.json` 存在且返回体含路径。
4. 人为断开控制面网络后 `yolo_get_status` → 返回 `REMOTE_*` 错误码而非裸堆栈。

### 已知限制

- `export-manifest.json` 依赖远程 SFTP 写入权限；失败不阻止导出命令已产生的文件列表返回。

---

