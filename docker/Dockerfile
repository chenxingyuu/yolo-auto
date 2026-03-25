# 这里用你已经构建好的镜像当基础镜像
# 如果你是用官方 Dockerfile 构建的，就填你自己的 tag，比如：
# FROM ultralytics/ultralytics:latest
FROM ultralytics/ultralytics:v8.4.22

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

# 安装 openssh-server（保持 root）
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd

ENV WORKDIR=/workspace/yolo-auto
RUN mkdir -p ${WORKDIR} /workspace/datasets /workspace/jobs

WORKDIR ${WORKDIR}

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