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
    openssh-server \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 安装 MinIO Client（mc），用于同步 MinIO/S3 数据集到本地训练目录
RUN curl -fsSL "https://dl.min.io/client/mc/release/linux-amd64/mc" -o /usr/local/bin/mc \
&& chmod +x /usr/local/bin/mc \
&& mc --version

RUN mkdir -p /var/run/sshd

ENV WORKDIR=/workspace/
RUN mkdir -p ${WORKDIR} /workspace/datasets /workspace/jobs /workspace/models

WORKDIR ${WORKDIR}

RUN cd /workspace/models \
    && python -c "from ultralytics import YOLO; \
models = [ \
    'yolov5n.pt','yolov5s.pt','yolov5m.pt','yolov5l.pt','yolov5x.pt', \
    'yolov8n.pt','yolov8s.pt','yolov8m.pt','yolov8l.pt','yolov8x.pt', \
    'yolo11n.pt','yolo11s.pt','yolo11m.pt','yolo11l.pt','yolo11x.pt', \
    'yolo26n.pt','yolo26s.pt','yolo26m.pt','yolo26l.pt','yolo26x.pt', \
]; [(YOLO(m), print(f'OK: {m}')) for m in models]"

# SSH 配置：禁用密码登录，只允许公钥，允许 root 公钥登录
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin without-password/' /etc/ssh/sshd_config && \
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh && \
    touch /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys

EXPOSE 2222

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD bash -c 'echo > /dev/tcp/127.0.0.1/2222' || exit 1

USER root

CMD ["/usr/sbin/sshd", "-D", "-p", "2222"]