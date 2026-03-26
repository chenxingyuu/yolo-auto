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
| `YOLO_PRIMARY_METRIC` | 主指标名，写入 MLflow 与里程碑文案，如 `map5095` |
| `MLFLOW_*` | 追踪后端与实验名 |
| `YOLO_WORK_DIR` / `JOBS_DIR` / `MODELS_DIR` / … | 远程路径（工作区、输出、预训练权重）与本地 `jobs.json` |
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

### 3.2 验证 MCP 注册数量（7 tools + 6 prompts + 9 resources）

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
uv run python -c "
from yolo_auto.server import mcp
t=len(mcp._tool_manager.list_tools())
p=len(mcp._prompt_manager.list_prompts())
r=len(mcp._resource_manager.list_resources())
print(f'tools={t} prompts={p} resources={r}')
"
```

预期：`tools=7 prompts=6 resources=9`。

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

## 4.1 MCP Prompts（提示模板）一览

Prompts 是预置的工作流模板，客户端（Cursor / Claude）可按名选用，一句话触发复杂流程。

| Prompt 名 | 参数 | 作用 |
|-----------|------|------|
| `quick-train` | `dataset`（必填）, `model`, `epochs` | 环境检查 → 启动训练 → 首次状态确认，全流程一键启动 |
| `dashboard` | 无 | 拉取所有任务、刷新 running 指标，生成全局状态看板 |
| `compare-experiments` | `job_ids`（逗号分隔） | 对比多个实验的指标与参数，给出最佳推荐和下一步建议 |
| `smart-tune` | `dataset`（必填）, `model`, `goal` | 根据业务目标设计搜索空间并一键执行 auto_tune |
| `diagnose` | `job_id` | 诊断训练异常：卡住/失败/指标差，给出修复方案 |
| `report` | `period`（默认"今天"） | 生成可直接转发的训练进展报告（摘要 + 明细 + 计划） |

## 4.2 MCP Resources（只读上下文）一览

Resources 是只读数据端点，客户端可按需拉取为模型提供上下文，不消耗 tool 调用额度。

| Resource URI | 名称 | 说明 |
|-------------|------|------|
| `yolo://config` | current-config | 当前生效的环境配置概要（已脱敏，不含密钥） |
| `yolo://jobs/active` | active-jobs | 运行中/排队中的任务列表（本地读取，不触发 SSH） |
| `yolo://jobs/history` | job-history | 最近 50 条任务记录（含已完成/失败/停止） |
| `yolo://mlflow/leaderboard` | mlflow-leaderboard | MLflow 实验按主指标排序的 Top 10 排行榜 |
| `yolo://datasets` | remote-datasets | 远程 datasets 目录下的 YAML 配置文件列表 |
| `yolo://models` | remote-models | 远程 models 目录下的预训练权重文件（.pt）列表及大小 |
| `yolo://env/gpu` | gpu-info | GPU 型号、显存总量/已用/空闲、利用率、温度（nvidia-smi） |
| `yolo://env/system` | system-info | CPU 核数与型号、内存用量、/workspace 磁盘用量 |
| `yolo://guide/training-params` | training-params-guide | YOLO 训练参数速查与推荐值（Markdown 格式） |

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
- **飞书通知**：所有消息均使用卡片（`interactive`），不再降级为文本消息。

## 7. MLflow 查看结果

项目依赖将 **MLflow 固定为 3.10.1**。本地用 **追踪服务**（`mlflow server`，含 UI）启动，并监听所有网卡以便局域网内其它设备访问：

```bash
uvx --from mlflow==3.10.1 mlflow server \
  --backend-store-uri ./mlruns \
  --host 0.0.0.0 \
  --port 5001 \
  --cors-allowed-origins '*'
```

**局域网说明**：`--host 0.0.0.0` 只负责「监听所有网卡」。MLflow 3 的安全中间件还要求你在命令行里显式配置下面至少一项，否则会看到 *Security middleware … localhost-only* 的提示，且跨设备访问 API 可能被拦：

- **`--cors-allowed-origins`**：允许的浏览器 `Origin`。内网开发可临时用 `*`（**勿用于公网**）；更安全写法是列出具体源，例如 `--cors-allowed-origins 'http://192.168.1.10:5001'`。

`docker/docker-compose.app.yaml` 里的 `mlflow-ui` 已带上与内网开发匹配的 `--allowed-hosts` / `--cors-allowed-origins`（CORS 为 `*`），可按环境收紧。

- 本机浏览器：`http://127.0.0.1:5001`
- 局域网其它设备：`http://<这台机器的局域网 IP>:5001`
- 飞书卡片里的追踪链接：请把 `MLFLOW_EXTERNAL_URL` 设为上述「外部能打开的」地址，例如 `http://192.168.1.10:5001`

`yolo_auto_tune` 返回的 `mlflowTopRuns` 与 `bestFromTrials` 可对照排查不一致。

## 7.1 Docker 部署（HTTP MCP + worker + sqlite MLflow）

本节用于把本项目以 **两个容器** 方式部署：

- **mcp 容器**：对外提供 MCP HTTP Endpoint（Streamable HTTP），供 Cursor/Claude 通过 URL 连接
- **worker 容器**：后台轮询任务状态并更新 MLflow/飞书
- （可选）**mlflow-ui 容器**：`uvx --from mlflow==3.10.1 mlflow server`，绑定 `0.0.0.0:5001`，便于本机与局域网浏览器访问

> 训练/GPU 侧不在本 compose 内；mcp/worker 通过 SSH 连接你已有的远程 GPU 机器/容器。

### 7.1.1 准备 SSH 私钥（推荐用 secrets）

将用于 SSH 的私钥放到：

`docker/secrets/yolo_ssh_key`

并确保权限安全（只读、不可被提交到 git）。

### 7.1.2 启动 compose

在项目根目录执行：

```bash
docker compose -f docker/docker-compose.app.yaml up -d --build
```

服务端口：

- MCP HTTP：`http://127.0.0.1:8000/mcp/`
- healthcheck：`http://127.0.0.1:8000/health`
- MLflow UI（可选）：`http://127.0.0.1:5001`

### 7.1.3 环境变量（关键项）

compose 内默认把 MLflow backend store 切到 sqlite：

- `MLFLOW_TRACKING_URI=sqlite:////data/mlflow/mlflow.db`
- `MLFLOW_EXPERIMENT_NAME=yolo-auto`

任务状态共享文件（mcp 与 worker 必须一致）：

- `YOLO_STATE_FILE=/data/state/jobs.json`
- `YOLO_WATCH_LOCK_FILE=/data/state/watch.lock`

SSH（建议用 `YOLO_SSH_ENVS`，或回退 `YOLO_SSH_HOST/PORT/USER`）：

- `YOLO_SSH_KEY_PATH=/run/secrets/yolo_ssh_key`

飞书：

- `FEISHU_WEBHOOK_URL=...`

### 7.1.4 在 Cursor 配置 HTTP MCP（示例）

你需要把 MCP server 配置为 URL 连接（具体 UI 入口依 Cursor 版本而定），并填：

- MCP URL：`http://127.0.0.1:8000/mcp/`

如需公网部署，建议在反向代理/网关层加鉴权，并限制来源 IP。

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
