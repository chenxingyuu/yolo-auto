# 这里用你已经构建好的镜像当基础镜像
# 如果你是用官方 Dockerfile 构建的，就填你自己的 tag，比如：
# FROM ultralytics/ultralytics:latest
FROM ultralytics/ultralytics:8.4.30

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

# 安装 openssh-server（保持 root）和 MinIO 客户端依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 安装 MinIO Client（mc），用于同步 MinIO/S3 数据集到本地训练目录
RUN curl -fsSL "https://dl.min.io/client/mc/release/linux-amd64/mc" -o /usr/local/bin/mc \
&& chmod +x /usr/local/bin/mc \
&& mc --version

ENV WORKDIR=/workspace/
RUN mkdir -p ${WORKDIR} /workspace/datasets /workspace/jobs /workspace/models

WORKDIR ${WORKDIR}

# 将训练控制面代码放入训练工作区目录
RUN mkdir -p /workspace/yolo-auto-control
RUN cd /workspace/yolo-auto-control && uv init && uv add fastapi uvicorn pydantic pyyaml -i https://mirrors.aliyun.com/pypi/simple
COPY src/yolo_auto/control_api.py /workspace/yolo-auto-control/control_api.py

EXPOSE 18080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD bash -c 'echo > /dev/tcp/127.0.0.1/18080' || exit 1

USER root

CMD ["/workspace/yolo-auto-control/.venv/bin/uvicorn", "--app-dir", "/workspace/yolo-auto-control", "control_api:app", "--host", "0.0.0.0", "--port", "18080"]