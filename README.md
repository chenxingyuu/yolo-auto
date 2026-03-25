# YOLO Auto MCP Workflow

通过 `MCP + Skill` 在 Cursor / Claude Code 中一句话触发 YOLO 自动化训练，并在训练过程中持续推送飞书进度。

## 这个项目能做什么

- 在 AI 对话里触发训练：`setup_env -> start_training -> get_status -> auto_tune`
- 自动记录实验到 MLflow（参数、指标、模型产物）
- 关键阶段自动推送飞书消息（开始、里程碑、结束、失败）
- 支持串行调参与失败跳过（V2 稳定性模式）

## 技术栈

- Python 3.12
- `uv`（虚拟环境与依赖管理）
- `mcp` Python SDK（MCP Server）
- `paramiko`（SSH 到远程 GPU 容器）
- `mlflow`（实验追踪）
- `httpx`（飞书 webhook）

## 目录结构

```text
yolo-auto/
├── .cursor/mcp.json
├── .env.example
├── pyproject.toml
└── src/yolo_auto/
    ├── server.py
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
        └── tuner.py
```

## 1. 环境准备

### 1.1 安装依赖

在项目根目录执行：

```bash
uv sync
```

### 1.2 配置环境变量

复制并填写 `.env.example`（或直接 export）：

```bash
cp .env.example .env
```

核心变量说明：

- `YOLO_SSH_HOST` / `YOLO_SSH_PORT` / `YOLO_SSH_USER` / `YOLO_SSH_KEY_PATH`：远程训练容器 SSH 连接
- `FEISHU_WEBHOOK_URL`：飞书群机器人 webhook
- `MLFLOW_TRACKING_URI`：MLflow 存储地址（默认本地 `./mlruns`）
- `MLFLOW_EXPERIMENT_NAME`：实验名
- `YOLO_WORK_DIR` / `YOLO_DATASETS_DIR` / `YOLO_JOBS_DIR`：远程路径
- `YOLO_STATE_FILE`：本地任务状态文件（默认 `.state/jobs.json`）

> 注意：当前版本会自动加载项目根目录 `.env`（且不会覆盖你已 `export` 的同名变量）。

## 2. 在 Cursor 注册 MCP

项目内已提供 `.cursor/mcp.json`，默认配置如下（已按本机路径写好）：

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

如果你的项目路径变了，请同步改 `--directory`。

## 3. 本地自检

### 3.1 代码检查

```bash
uv run ruff check .
```

### 3.2 验证 MCP 工具是否成功注册

```bash
YOLO_SSH_HOST=127.0.0.1 \
YOLO_SSH_PORT=2222 \
YOLO_SSH_USER=test \
YOLO_SSH_KEY_PATH=/tmp/id_rsa \
FEISHU_WEBHOOK_URL=https://example.com/hook \
MLFLOW_TRACKING_URI=./mlruns \
MLFLOW_EXPERIMENT_NAME=yolo-auto \
YOLO_WORK_DIR=/workspace/yolo-openclaw \
YOLO_DATASETS_DIR=/workspace/datasets \
YOLO_JOBS_DIR=/workspace/jobs \
YOLO_STATE_FILE=.state/jobs.json \
uv run python -c "from yolo_auto.server import mcp; print('mcp-tools-ok', len(mcp._tool_manager.list_tools()))"
```

预期输出包含：`mcp-tools-ok 5`

## 4. 在 Cursor / Claude Code 里怎么用

你可以在对话里直接下自然语言指令，例如：

- “先检查训练环境可用性，数据配置是 `/workspace/datasets/helmet.yaml`”
- “用 yolov8s 启动训练：epochs 100，img 640，batch 16，lr 0.01”
- “每隔一段时间帮我查询状态，直到完成”
- “基于这个任务做串行调参，最多 4 组”

对应 MCP 工具：

- `yolo.setup_env`
- `yolo.start_training`
- `yolo.get_status`
- `yolo.stop_training`
- `yolo.auto_tune`

## 5. 飞书汇报机制

系统会在以下时机发飞书消息：

- 训练启动
- 调参每个 trial 的阶段结果
- 状态变化（完成/失败/停止）

V2 已做“状态去抖”：同一状态不会反复刷屏。

## 6. MLflow 查看结果

如果你用默认本地存储：

```bash
uv run mlflow ui --backend-store-uri ./mlruns --port 5001
```

打开 `http://127.0.0.1:5001` 查看实验：

- 参数：model / epochs / batch / lr
- 指标：loss / map50 / map5095 / precision / recall
- 运行状态：FINISHED / KILLED

## 7. 常见问题

- SSH 连接失败：
  - 检查 `YOLO_SSH_*` 是否正确
  - 检查 key 权限与远程 `authorized_keys`
- `results.csv` 不存在：
  - 训练可能刚开始，稍后重试 `yolo.get_status`
  - 检查数据路径和 yolo 命令是否正确
- 飞书没收到消息：
  - 检查 webhook 地址是否有效
  - 机器人是否被移出群聊
- MLflow 看不到 run：
  - 确认 `MLFLOW_TRACKING_URI` 与启动 `mlflow ui` 的目录一致

## 8. 下一步建议

- 增加 `.env` 自动加载（例如 `python-dotenv`）
- 增加 pytest + mock SSH 的回归测试
- 增加 PID 文件机制，进一步降低 stop 误杀风险

