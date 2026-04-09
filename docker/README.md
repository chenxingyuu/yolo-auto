# YOLO AUTO 训练容器（HTTP 控制面）

本目录构建的镜像用于 **GPU 服务器上长期运行**：提供 Ultralytics/YOLO 训练环境与 HTTP 控制面（FastAPI）。
本地的 MCP 服务不在该容器内运行；`yolo-auto-app`（mcp/worker）通过 HTTP 调用本容器接口完成训练编排。

## 与本地 MCP / `.env` 的对应关系

本地项目根目录 `.env` 中建议配置与容器一致（与 [`.env.example`](../.env.example) 对齐）：

| 变量 | 容器内默认路径 | 说明 |
|------|----------------|------|
| `YOLO_WORK_DIR` | `/workspace` | 训练工作目录 |
| `YOLO_DATASETS_DIR` | `/workspace/datasets` | 数据集根，通常用 volume 挂载 |
| `YOLO_JOBS_DIR` | `/workspace/jobs` | 任务输出（weights、日志、`results.csv` 等） |
| `YOLO_MODELS_DIR` | `/workspace/models` | 预训练模型目录 |
| `YOLO_CONTROL_BASE_URL` | `http://<train-host>:18080` | MCP 访问训练控制面的地址 |
| `YOLO_CONTROL_BEARER_TOKEN` | 自定义 | 控制面鉴权 Token（建议开启） |

若你单独改过 `.env`，请确认路径变量与容器挂载一致。

升级自旧默认路径的用户：若仍使用 `YOLO_WORK_DIR=/workspace/yolo-openclaw`，请在 `.env` 中改为 `/workspace/yolo-auto`（或自行在容器内保留原路径并挂载）。

## 前置条件（GPU）

- 宿主机安装 NVIDIA 驱动。
- 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)，使 `docker run --gpus all` 或 Compose 的 `gpus: all` 可用。

## 构建镜像

在本目录下执行：

```bash
docker build -t yolo-auto:latest .
```

镜像特性：

- 基础镜像：`ultralytics/ultralytics:8.4.30`（可按需更换 tag）。
- Python 与 Ultralytics 来自基础镜像；预创建 `/workspace/datasets`、`/workspace/jobs`、`/workspace/models`。
- **`/workspace/models`** 内已预下载 YOLOv5、v8、11、26 共 20 个官方 detect 权重（n/s/m/l/x 全尺寸），训练时直接用 `/workspace/models/yolov8n.pt` 等绝对路径即可，无需重复下载。
- 以 `root` 运行，`WORKDIR` 为 `/workspace/`。
- 控制面监听 **18080**。
- `HEALTHCHECK`：探测本机 `18080` 端口是否可连。

## 运行容器（docker run）

在 GPU 服务器上示例：

```bash
docker run -d \
  --gpus all \
  --name yolo-auto \
  -p 18080:18080 \
  -v /data/yolo-datasets:/workspace/datasets \
  -v /data/yolo-jobs:/workspace/jobs \
  -v /data/yolo-models:/workspace/models \
  -e YOLO_CONTROL_BEARER_TOKEN=your-token \
  yolo-auto:latest
```

- `/workspace/datasets`：数据集根目录。
- `/workspace/jobs`：训练输出目录。
- `/workspace/models`：预训练权重目录（首次启动时已有镜像内置的 .pt 文件；挂载 volume 后持久化，后续自行下载的权重也不会丢失）。
- `YOLO_CONTROL_BEARER_TOKEN`：控制面鉴权令牌，MCP 侧需保持一致。

## 使用 Docker Compose

在 **`docker/`** 目录下：

```bash
docker compose up -d
```

（可选）使用本地 Dockerfile 构建而非仅拉取镜像：

```bash
docker compose build --no-cache
docker compose up -d
```

`docker-compose.yaml` 已暴露控制面端口 `18080` 并配置 `gpus: all`；若当前 Compose 版本不支持该字段，可改用与 `docker run --gpus all` 等效的方式启动，或升级 Docker Compose。

## MCP 侧变量

本地 `.env` 需填写：

- `YOLO_CONTROL_BASE_URL=http://<train-host>:18080`
- `YOLO_CONTROL_BEARER_TOKEN=<same-token>`
- `YOLO_CONTROL_TIMEOUT_SECONDS=30`
