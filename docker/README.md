# YOLO AUTO 训练容器（MCP 远程 SSH）

本目录构建的镜像用于 **GPU 服务器上长期运行**：提供 `sshd` 与 Ultralytics/YOLO 环境。本地的 **MCP 不在容器内运行**；Cursor 等客户端通过本仓库的 `yolo_auto` MCP 经 SSH 连接该容器，在容器内执行 `yolo detect train` 等命令。

## 与本地 MCP / `.env` 的对应关系

本地项目根目录 `.env` 中的远程路径建议与容器内一致（与 [`.env.example`](../.env.example) 对齐）：

| 变量 | 容器内默认路径 | 说明 |
|------|----------------|------|
| `YOLO_WORK_DIR` | `/workspace/yolo-auto` | 训练时 `cd` 到此目录；数据集 YAML 建议放在此目录下或使用绝对路径 |
| `YOLO_DATASETS_DIR` | `/workspace/datasets` | 数据集根，通常用 volume 挂载 |
| `YOLO_JOBS_DIR` | `/workspace/jobs` | 任务输出（weights、日志、`results.csv` 等） |

若你单独改过 `.env`，请确认 **`YOLO_WORK_DIR` 与镜像内目录一致**，否则 `yolo_setup_env` 会报目录或数据配置不存在。

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

- 基础镜像：`ultralytics/ultralytics:v8.4.22`（可按需更换 tag）。
- Python 与 Ultralytics 来自基础镜像；预创建 `/workspace/yolo-auto`、`/workspace/datasets`、`/workspace/jobs`。
- 以 `root` 运行，`WORKDIR` 为 `/workspace/yolo-auto`。
- `sshd` 监听 **2222**，禁用密码登录，仅 **公钥** 登录 root。
- `HEALTHCHECK`：探测本机 `2222` 端口是否可连。

## 运行容器（docker run）

在 GPU 服务器上示例：

```bash
docker run -d \
  --gpus all \
  --name yolo-auto \
  -p 2222:2222 \
  -v /data/yolo-datasets:/workspace/datasets \
  -v /data/yolo-jobs:/workspace/jobs \
  -v /path/to/your/authorized_keys:/root/.ssh/authorized_keys:ro \
  yolo-auto:latest
```

- `/workspace/yolo-auto`：工作区（镜像已创建；可将 `data.yaml` 等通过额外 `-v` 挂入）。
- `/workspace/datasets`：数据集根目录。
- `/workspace/jobs`：训练输出目录。
- `authorized_keys`：宿主机上的公钥列表文件，只读挂载到容器内 **`/root/.ssh/authorized_keys`**（用户名为 root）。

首次联调前请放行宿主机防火墙 **2222**（若跨机 SSH）。

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

`docker-compose.yaml` 中已将宿主机 `/root/.ssh/authorized_keys` 映射到容器 **`/root/.ssh/authorized_keys`**；请按你的环境修改宿主机公钥路径。已配置 `gpus: all`；若当前 Compose 版本不支持该字段，可改用与 `docker run --gpus all` 等效的方式启动，或升级 Docker Compose。

## MCP 侧 SSH 变量

本地 `.env` 需填写 `YOLO_SSH_HOST`、`YOLO_SSH_PORT`（默认 `2222`）、`YOLO_SSH_USER`（通常为 `root`）、`YOLO_SSH_KEY_PATH`（私钥路径），与上述容器 SSH 暴露方式一致。
