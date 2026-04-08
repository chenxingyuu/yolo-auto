# 迭代记录

本文档按时间记录面向维护者与高级用户的迭代说明。**新条目追加在文件顶部**（越新越靠前）。

---

## 2026-04-08 — 数据血缘、导出 manifest、SSH 错误码

### 目标摘要

收紧「MinIO/CVAT 导出 → 训练 → 验证 → 交付」可追溯性与可观测性：任务可关联数据来源；导出产物有机器可读清单；`get_status` 对 SSH 失败返回稳定错误码。

### 用户可见变更


| 区域                    | 变更                                                                                                                                                                                      |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yolo_sync_dataset`   | 成功时在 `payload` 中增加 `**provenance`**（bucket/prefix、`objectName`、`mcSourcePath`、`extractedDir`、`dataConfigPath`、`dataYamlRelative`、`datasetName` 等）。                                      |
| `yolo_start_training` | 可选 `**minioExportZip**`、`**datasetSlug**`、`**datasetVersionNote**`（支持 snake_case）；写入 `**JobRecord.datasetProvenance**` 及参数中的 `dataset_*` 前缀项。 |
| `yolo_export`         | 成功时在远程 `**{jobDir}/export-manifest.json**` 写入清单，返回体含 `**exportManifestPath**` 与内联 `**exportManifest**`（失败写入时含 `exportManifestWarning`）。                                                 |
| `yolo_get_status`     | SSH 连接类失败映射为 `**SSH_***` 系列 `errorCode`（如 `SSH_TIMEOUT`、`SSH_AUTH_FAILED`、`SSH_CONNECT_REFUSED`）及 `**retryable**` / `**hint**`。                                                         |


### 兼容与迁移

- **本地状态库**：`JobRecord` 增加可选字段 `**datasetProvenance`**；旧 `jobs.db` 无该字段时反序列化为 `null`，无需手工迁移。

### 推荐验证步骤

1. `yolo_sync_dataset` → 确认响应含 `provenance`。
2. `yolo_start_training` 传入与 `provenance` 对齐的可选血缘字段 → `yolo_get_job` 中可见 `datasetProvenance`。
3. `yolo_export` → 远程 `export-manifest.json` 存在且返回体含路径。
4. 人为断开 SSH 后 `yolo_get_status` → 返回 `SSH_*` 错误码而非裸堆栈。

### 已知限制

- `export-manifest.json` 依赖远程 SFTP 写入权限；失败不阻止导出命令已产生的文件列表返回。

---

