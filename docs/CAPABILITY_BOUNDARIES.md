# yolo-auto 能力边界说明

本文档描述 **当前分支** 下本项目的定位、已覆盖能力与**刻意不覆盖**的范围，便于评估是否适合你的场景。

## 1. 项目定位

**yolo-auto 是一个「编排层」**：通过 MCP（Model Context Protocol）把 Cursor / Claude 等客户端与 **远程 SSH 上的 Ultralytics YOLO 训练环境**、**MLflow 追踪**、**飞书通知**、可选的 **CVAT / MinIO** 串联起来。

- **不是** Ultralytics 的 fork，也 **不替代** 你在远程机器上已安装的 `yolo` CLI 与 CUDA 环境。
- **不负责** 在本地起 GPU 容器或安装驱动；默认假设训练在 **可 SSH 登录的远程主机** 上执行。

## 2. 能力范围（能做什么）

### 2.1 训练与实验生命周期

| 能力 | 说明 |
|------|------|
| 远程环境检查 | `yolo_setup_env`：SSH 上检查工作目录、模型权重、数据集 YAML 等 |
| 数据集质检 / 修复 | `yolo_check_dataset`、`yolo_fix_dataset`：针对 YOLO 风格 data.yaml + 图片/标签目录 |
| 启动训练 | `yolo_start_training`：在远程 **异步** 拉起 `yolo detect train ...`，写入本地任务状态与 MLflow |
| 状态与指标 | `yolo_get_status`：读远程 `results.csv`、更新 MLflow、触发飞书里程碑/终态卡片 |
| 停止 / 验证 / 导出 | `yolo_stop_training`、`yolo_validate`、`yolo_export` |
| 任务列表与详情 | `yolo_list_jobs`、`yolo_get_job`（可选 `refresh` 拉状态） |
| 删除本地记录 | `yolo_delete_job`：**仅删除本地 SQLite 中的任务状态**，不删远程日志与权重 |

未显式传入 `jobId` 时，会按 **模型与数据集路径** 生成可读 ID，并在状态库中 **自动避让重名**（见 `tools/job_naming.py`）。

### 2.2 实验追踪与对比

- **MLflow**：参数与指标写入指定 `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_NAME`；训练会打 **分组类 tags**（如 `yolo_job_id`、`yolo_data_stem` 等），便于 `search_runs` 筛选。
- **排行榜资源**：`yolo://mlflow/leaderboard` 提供按 **主指标**（`YOLO_PRIMARY_METRIC`）排序的 Top 10；可通过 `MLFLOW_LEADERBOARD_FILTER` 缩小范围。
- **自动调参** `yolo_auto_tune`：**串行** 跑多个 trial，并与 MLflow 中同筛选条件下的最优 run 对照（`bestFromTrials` / `bestFromMlflow` / `disagreement`）。

### 2.3 通知与后台轮询

- **飞书**：里程碑、启动/完成/失败等依赖 `FEISHU_*`；应用机器人路径下支持 **schema 卡片** 与更新（需配置 `message_id` 等，见实现与 `.env.example`）。
- **`yolo-auto-watch` Worker**：对本地状态中 `running` 的任务周期性执行与 `yolo_get_status` 等价的逻辑，**避免 IDE 必须一直轮询**。需与 MCP **共享同一 `YOLO_STATE_FILE`**。

### 2.4 数据流水线（可选）

- **MinIO**：`yolo_sync_dataset` 与 `yolo://minio/datasets` 等依赖远程已配置 `mc` 与 bucket。

### 2.5 只读上下文（MCP Resources）

提供配置摘要、活跃/历史任务、远程数据集 YAML 列表、模型列表、GPU/系统信息、训练参数指南、CVAT 列表、任务日志片段等（详见 README「MCP Resources」）；**不消耗 tool 配额**，但部分内容会 **SSH 或访问 MLflow**。

## 3. 边界与限制（不保证什么）

### 3.1 架构与运行时

- **训练不在 MCP 进程内执行**：长时间计算全部在 **SSH 目标机**；MCP 只做命令下发与状态拉取。
- **异步模型**：`start_training` 成功只表示「已提交远程进程并登记状态」；真实进度依赖 **`get_status` 或 Worker**，否则 MLflow/飞书可能长期不更新。
- **单机 Worker**：设计上是 **单实例** 持锁轮询；多实例重复跑 Worker 或 IDE 极高频 `get_status` 会放大 SSH/飞书压力。

### 3.2 调参与搜索

- `yolo_auto_tune` 为 **串行 trial**，不是分布式超参搜索；搜索空间通过笛卡尔积展开，受 `max_trials` 截断。
- `learningRate` 在 `yolo_start_training` 中为 **可选**；`optimizer=auto` 时 Ultralytics 可能 **忽略** 手动传入的 lr 等，行为以官方文档与 `train.log` 为准。

### 3.3 状态存储与删除

- 任务状态存在 **本地 SQLite**（默认 `.state/jobs.db`，可从旧 `.json` 迁移）。**不包含** 远程文件系统上的「唯一真相」；若手工删远程作业而本地未同步，可能出现不一致。
- `yolo_delete_job` **不会** 停止远程进程，也 **不会** 删除权重与日志；仅清理本地记录，且 **queued/running** 会被拒绝。

### 3.4 MLflow 与网络

- MCP 与 Worker 使用的 `MLFLOW_TRACKING_URI` 必须 **实际可写可读**；若追踪库在远程而 URI 指向本机且远程无法访问，则远程训练进程写 MLflow 的行为取决于你如何部署（常见做法是追踪服务对 MCP/训练机均可达）。
- `MLFLOW_EXTERNAL_URL` 仅用于生成 **人类可点击的链接**（如飞书卡片），不改变追踪后端本身。

### 3.5 飞书与消息形态

- 未配置有效飞书凭证时，相关能力 **静默跳过或降级**，不应假设「一定有推送」。
- 卡片能力（含图片 key、更新同一条消息）依赖 **飞书开放平台能力与权限**，受租户策略与机器人能力限制。

### 3.6 CVAT / MinIO / SSH

- 均为 **可选集成**：环境变量不全时，对应工具或资源会失败或返回空结果，需自行补齐运维与网络连通性。
- SSH 认证、跳板机、多环境 `YOLO_SSH_ENVS` 由部署方维护；本项目不做堡垒机逻辑。

### 3.7 平台

- Worker 使用 **`fcntl` 文件锁**（类 Unix）。在 **原生 Windows** 上若未使用 WSL 等兼容层，锁行为可能与 Linux/macOS 不一致，需自行验证。

### 3.8 数据集与任务类型

- 质检/修复逻辑面向 **典型 YOLO 检测数据布局**；分割、姿态、OBB 等 **若 CLI/数据格式不同**，部分检查与修复可能不适用或需自行扩展。
- 资源列表类接口（如远程 `find` YAML）有 **条数/深度** 等实现级上限，超大仓库不应依赖「枚举全部文件」。

## 4. 与 README 的关系

- **操作步骤、环境变量表、工具/资源清单** 以仓库根目录 `README.md` 为准。
- 本文档侧重 **能力边界与架构假设**，随功能迭代应同步更新；若二者冲突，以 **代码与 README 中的具体参数** 为最终依据。

## 5. 版本提示

当前依赖将 **MLflow 固定为 3.10.1**（见 `pyproject.toml`）。升级主版本可能带来 API/安全中间件行为变化，需单独回归。
