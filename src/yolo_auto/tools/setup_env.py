from __future__ import annotations

from yolo_auto.ssh_client import SSHClient


def setup_env(ssh_client: SSHClient, work_dir: str, data_config_path: str) -> dict[str, object]:
    version_cmd = "python -c 'import ultralytics; print(ultralytics.__version__)'"
    stdout_text, stderr_text, exit_code = ssh_client.execute(version_cmd)
    if exit_code != 0:
        return {"reachable": False, "error": stderr_text.strip() or "ultralytics not available"}

    check_cmd = f"test -d {work_dir} && test -f {data_config_path}"
    _, check_err, check_code = ssh_client.execute(check_cmd)
    if check_code != 0:
        return {
            "reachable": True,
            "workDir": work_dir,
            "yoloVersion": stdout_text.strip(),
            "validData": False,
            "error": check_err.strip() or "workDir or dataConfigPath missing",
        }

    return {
        "reachable": True,
        "workDir": work_dir,
        "yoloVersion": stdout_text.strip(),
        "validData": True,
    }

