FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast, reproducible Python packaging)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency manifests first for better layer caching
COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev

# Copy application code
COPY src ./src

ENV PYTHONPATH=/app/src

EXPOSE 8000

# Default to running MCP server (override in compose for worker)
CMD ["uv", "run", "yolo-auto-mcp"]
