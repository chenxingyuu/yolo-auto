# YOLO Auto MCP Workflow

通过 `MCP + Skill` 在 Cursor / Claude Code 中一句话触发 YOLO 自动化训练，并在训练过程中持续推送飞书进度。

## 这个项目能做什么

- 在 AI 对话里触发训练：`setup_env` → `start_training` → `get_status` →（可选）`auto_tune`
- 自动记录实验到 MLflow（参数、指标）
- 飞书通知：启动、**按 epoch 里程碑**、完成/失败/停止、调参 trial 结果
- **V3**：`list_jobs` / `get_job` 在对话中查看本地任务列表；`auto_tune` 返回与 MLflow `compare_runs` 的对照；可选后台 `yolo-auto-watch` 轮询

## 技术栈

- Python 3.12
- `uv`（虚拟环境与依赖管理）
- `mcp` Python SDK（MCP Server）
- `paramiko`（SSH 到远程 GPU 容器）
- `mlflow`（实验追踪）
- `httpx`（飞书 webhook）
- `python-dotenv`（加载项目根目录 `.env`）

## 目录结构

```text
yolo-auto/
├── .cursor/mcp.json
├── .env.example
├── pyproject.toml
└── src/yolo_auto/
    ├── server.py          # MCP 入口
    ├── worker.py         # 可选后台轮询（飞书里程碑）
    ├── config.py
    ├── ssh_client.py
    ├── feishu.py
    ├── tracker.py
    ├── state_store.py
    ├── models.py
    ├── errors.py
    └── tools/
        ├── setup_env.py
        ├── training.py
        ├── status.py
        ├── tuner.py
        ├── jobs.py        # list / get job
        └── metrics_csv.py
```

## 1. 环境准备

### 1.1 安装依赖

在项目根目录执行：

```bash
uv sync
```

### 1.2 配置环境变量

复制并填写 `.env.example`：

```bash
cp .env.example .env
```

| 变量 | 说明 |
|------|------|
| `YOLO_SSH_*` | 远程训练容器 SSH |
| `FEISHU_WEBHOOK_URL` | 飞书群机器人 Webhook |
| `FEISHU_REPORT_ENABLE` | 是否启用训练中里程碑推送（默认 `true`） |
| `FEISHU_REPORT_EVERY_N_EPOCHS` | 每 N 个 epoch 推一次（`0` 表示关闭里程碑，仅状态变化时推送） |
| `FEISHU_MESSAGE_MODE` | `text` 或 `card`（`card` 使用飞书 `post`，失败时降级为文本） |
| `YOLO_PRIMARY_METRIC` | 主指标名，写入 MLflow 与里程碑文案，如 `map5095` |
| `MLFLOW_*` | 追踪后端与实验名 |
| `YOLO_WORK_DIR` / `JOBS_DIR` / … | 远程路径与本地 `jobs.json` |
| `YOLO_WATCH_POLL_INTERVAL_SECONDS` | 后台 Worker 轮询间隔（秒） |
| `YOLO_WATCH_LOCK_FILE` | Worker 文件锁路径，避免多实例抢锁 |

> 根目录 `.env` 会在 `load_settings()` 时自动加载，**不会**覆盖你已 `export` 的同名变量。

## 2. 在 Cursor 注册 MCP

项目内 `.cursor/mcp.json` 示例（按你的本机路径修改 `--directory`）：

```json
{
  "mcpServers": {
    "yolo-auto": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/Users/chenxingyu/work/chenxingyu/yolo-auto",
        "python",
        "-m",
        "yolo_auto.server"
      ]
    }
  }
}
```

## 3. 本地自检

### 3.1 代码检查

```bash
uv run ruff check .
```

### 3.2 验证 MCP 工具注册数量（应为 7）

```bash
YOLO_SSH_HOST=127.0.0.1 \
YOLO_SSH_PORT=2222 \
YOLO_SSH_USER=test \
YOLO_SSH_KEY_PATH=/tmp/id_rsa \
FEISHU_WEBHOOK_URL=https://example.com/hook \
MLFLOW_TRACKING_URI=./mlruns \
MLFLOW_EXPERIMENT_NAME=yolo-auto \
YOLO_WORK_DIR=/workspace/yolo-auto \
YOLO_DATASETS_DIR=/workspace/datasets \
YOLO_JOBS_DIR=/workspace/jobs \
YOLO_STATE_FILE=.state/jobs.json \
uv run python -c "from yolo_auto.server import mcp; print('mcp-tools-ok', len(mcp._tool_manager.list_tools()))"
```

预期：`mcp-tools-ok 7`。

## 4. MCP 工具一览

| 工具名 | 作用 |
|--------|------|
| `yolo_setup_env` | 检查远程环境与数据配置 |
| `yolo_start_training` | 启动训练（异步），写入状态 |
| `yolo_get_status` | 拉取 `results.csv`、更新 MLflow、飞书里程碑/完成通知 |
| `yolo_stop_training` | 停止训练 |
| `yolo_auto_tune` | 串行调参；返回 `bestFromTrials`、`bestFromMlflow`、`disagreement` |
| `yolo_list_jobs` | 列出最近任务（含 `epochHint`） |
| `yolo_get_job` | 按 `jobId` 查看记录；`refresh=true` 时顺带调用 `get_status` |

在对话中可以自然语言描述，由模型按需调用上述工具。

## 5. 可选：后台 Watch Worker

无需一直让 IDE 轮询时，可在项目根目录另开终端：

```bash
uv run yolo-auto-watch
```

或使用等价入口：

```bash
uv run python -m yolo_auto.worker
```

Worker 会获取 `YOLO_WATCH_LOCK_FILE` 排他锁，对状态为 `running` 的任务周期性调用与 `yolo_get_status` 相同的逻辑（含里程碑飞书）。**建议同一时间只运行一个 Worker**；若 IDE 与 Worker 同时高频 `get_status`，仍可能增加 SSH 与飞书频率。

## 6. 飞书与里程碑

- **状态去抖**：完成/失败/停止时仍按「状态变更」通知，避免重复终态刷屏。
- **里程碑**：训练进行中，当 `epoch >= lastReportedEpoch + N` 且 `FEISHU_REPORT_ENABLE=true`、`N>0` 时推送；`lastReportedEpoch` 持久化在 `.state/jobs.json`。
- **`FEISHU_MESSAGE_MODE=card`**：使用飞书 `post` 富文本；请求失败时自动改发纯文本。

## 7. MLflow 查看结果

```bash
uv run mlflow ui --backend-store-uri ./mlruns --port 5001
```

浏览器打开 `http://127.0.0.1:5001`，查看参数、指标与 run 状态。`yolo_auto_tune` 返回的 `mlflowTopRuns` 与 `bestFromTrials` 可对照排查不一致。

## 8. 常见问题

- **`.env` 不生效**：确认文件在**项目根目录**且变量名与 `.env.example` 一致；或先 `export` 再启动进程。
- **SSH 失败**：检查 `YOLO_SSH_*`、密钥权限、容器 `sshd`。
- **`results.csv` 暂无**：训练刚启动，稍后重试 `yolo_get_status`。
- **飞书收不到**：检查 Webhook、机器人是否在群内。
- **Worker 与 IDE 重复通知**：降低轮询频率，或仅保留一种轮询方式。

## 9. 后续可增强

- 远程 `train.pid` 精确 stop
- `pytest` + mock SSH 回归测试
- 多 `envId` SSH 配置映射
