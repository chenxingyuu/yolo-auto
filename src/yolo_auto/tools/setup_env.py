from __future__ import annotations

from yolo_auto.errors import err, ok
from yolo_auto.ssh_client import SSHClient


def setup_env(ssh_client: SSHClient, work_dir: str, data_config_path: str) -> dict[str, object]:
    version_cmd = "python -c 'import ultralytics; print(ultralytics.__version__)'"
    stdout_text, stderr_text, exit_code = ssh_client.execute(version_cmd)
    if exit_code != 0:
        return err(
            error_code="ENV_UNREACHABLE",
            message=stderr_text.strip() or "ultralytics not available",
            retryable=True,
            hint="检查 SSH、Python 环境与 ultralytics 安装",
            payload={"reachable": False},
        )

    check_cmd = f"test -d {work_dir} && test -f {data_config_path}"
    _, check_err, check_code = ssh_client.execute(check_cmd)
    if check_code != 0:
        return err(
            error_code="DATA_CONFIG_INVALID",
            message=check_err.strip() or "workDir or dataConfigPath missing",
            retryable=False,
            hint="确认 dataConfigPath 与工作目录在远程容器内存在",
            payload={
                "reachable": True,
                "workDir": work_dir,
                "yoloVersion": stdout_text.strip(),
                "validData": False,
            },
        )

    return ok(
        {
            "reachable": True,
            "workDir": work_dir,
            "yoloVersion": stdout_text.strip(),
            "validData": True,
        }
    )

