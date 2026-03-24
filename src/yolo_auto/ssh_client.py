from __future__ import annotations

from dataclasses import dataclass

import paramiko


@dataclass(frozen=True)
class SSHConfig:
    host: str
    port: int
    user: str
    key_path: str


class SSHClient:
    def __init__(self, config: SSHConfig) -> None:
        self._config = config

    def execute(self, command: str, timeout: int = 30) -> tuple[str, str, int]:
        client = self._connect()
        try:
            _, stdout, stderr = client.exec_command(command, timeout=timeout)
            stdout_text = stdout.read().decode("utf-8", errors="replace")
            stderr_text = stderr.read().decode("utf-8", errors="replace")
            exit_code = int(stdout.channel.recv_exit_status())
            return stdout_text, stderr_text, exit_code
        finally:
            client.close()

    def execute_background(self, command: str) -> tuple[str, int]:
        bg_cmd = f"nohup bash -lc '{command}' > /dev/null 2>&1 & echo $!"
        stdout_text, stderr_text, exit_code = self.execute(bg_cmd, timeout=10)
        if exit_code != 0:
            raise RuntimeError(f"Failed to run background command: {stderr_text.strip()}")
        pid_text = stdout_text.strip().splitlines()[0] if stdout_text.strip() else "0"
        return pid_text, exit_code

    def read_file(self, path: str) -> str:
        client = self._connect()
        try:
            sftp = client.open_sftp()
            with sftp.file(path, "r") as f:
                content = f.read().decode("utf-8", errors="replace")
                return content
        finally:
            client.close()

    def _connect(self) -> paramiko.SSHClient:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=self._config.host,
            port=self._config.port,
            username=self._config.user,
            key_filename=self._config.key_path,
            timeout=10,
        )
        return ssh

