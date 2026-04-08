# 迭代记录

本文档按时间记录面向维护者与高级用户的迭代说明。**新条目追加在文件顶部**（越新越靠前）。

---

## 2026-04-08 — 数据血缘、验证写 MLflow、导出 manifest、SSH 错误码

### 目标摘要

收紧「MinIO/CVAT 导出 → 训练 → 验证 → 交付」可追溯性与可观测性：任务与 MLflow 可关联数据来源；独立验证指标进入同一 run；导出产物有机器可读清单；`get_status` 对 SSH 失败返回稳定错误码。

### 用户可见变更


| 区域                    | 变更                                                                                                                                                                                      |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yolo_sync_dataset`   | 成功时在 `payload` 中增加 `**provenance`**（bucket/prefix、`objectName`、`mcSourcePath`、`extractedDir`、`dataConfigPath`、`dataYamlRelative`、`datasetName` 等）。                                      |
| `yolo_start_training` | 可选 `**minioExportZip**`、`**datasetSlug**`、`**datasetVersionNote**`（支持 snake_case）；写入 `**JobRecord.datasetProvenance**`、MLflow **tags**（`yolo_minio_export_zip` 等）及参数中的 `dataset_*` 前缀项。 |
| `yolo_validate`       | 默认将 `**val_precision` / `val_recall` / `val_map50` / `val_map5095`** 记入对应 MLflow run（step 固定为 999999）；可选 `**logToMlflow**` 关闭单次上报。                                                      |
| `yolo_export`         | 成功时在远程 `**{jobDir}/export-manifest.json**` 写入清单，返回体含 `**exportManifestPath**` 与内联 `**exportManifest**`（失败写入时含 `exportManifestWarning`）。                                                 |
| `yolo_get_status`     | SSH 连接类失败映射为 `**SSH_***` 系列 `errorCode`（如 `SSH_TIMEOUT`、`SSH_AUTH_FAILED`、`SSH_CONNECT_REFUSED`）及 `**retryable**` / `**hint**`。                                                         |
| `yolo://config`       | 增加只读字段 `**validateLogToMlflow**`（对应环境变量总开关）。                                                                                                                                            |


### 配置


| 变量                            | 说明                                                                                                  |
| ----------------------------- | --------------------------------------------------------------------------------------------------- |
| `YOLO_VALIDATE_LOG_TO_MLFLOW` | 默认 `true`；为 `false` 时全局不向 MLflow 写入验证指标（单次仍可用工具参数覆盖意图时需服务端逻辑：当前为 **env 与 `logToMlflow` 同时满足** 才写入）。 |


### 兼容与迁移

- **本地状态库**：`JobRecord` 增加可选字段 `**datasetProvenance`**；旧 `jobs.db` 无该字段时反序列化为 `null`，无需手工迁移。
- **MLflow**：验证指标使用独立 step，避免与训练 epoch 曲线混在同一 step。

### 推荐验证步骤

1. `yolo_sync_dataset` → 确认响应含 `provenance`。
2. `yolo_start_training` 传入与 `provenance` 对齐的可选血缘字段 → `yolo_get_job` / MLflow UI 中可见 tags。
3. 训练完成后 `yolo_validate` → MLflow run 中出现 `val_*` 指标。
4. `yolo_export` → 远程 `export-manifest.json` 存在且返回体含路径。
5. 人为断开 SSH 后 `yolo_get_status` → 返回 `SSH_*` 错误码而非裸堆栈。

### 已知限制

- 验证指标写入失败时仍返回验证成功，并在 payload 中带 `**mlflowLogWarning`** / `**mlflowValidationLogged**`。
- `export-manifest.json` 依赖远程 SFTP 写入权限；失败不阻止导出命令已产生的文件列表返回。

---

