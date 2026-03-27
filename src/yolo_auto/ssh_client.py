from __future__ import annotations

import time
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

    def execute(
        self,
        command: str,
        timeout: int = 30,
        retries: int = 2,
        retry_delay_seconds: float = 0.4,
    ) -> tuple[str, str, int]:
        attempt = 0
        while True:
            try:
                client = self._connect()
                try:
                    _, stdout, stderr = client.exec_command(command, timeout=timeout)
                    stdout_text = stdout.read().decode("utf-8", errors="replace")
                    stderr_text = stderr.read().decode("utf-8", errors="replace")
                    exit_code = int(stdout.channel.recv_exit_status())
                    return stdout_text, stderr_text, exit_code
                finally:
                    client.close()
            except Exception:
                if attempt >= retries:
                    raise
                attempt += 1
                time.sleep(retry_delay_seconds)

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

    def upload_bytes(self, data: bytes, remote_path: str) -> None:
        client = self._connect()
        try:
            sftp = client.open_sftp()
            with sftp.file(remote_path, "wb") as f:
                f.write(data)
        finally:
            client.close()

    def file_exists(self, path: str) -> bool:
        _, _, exit_code = self.execute(f"test -f {path}")
        return exit_code == 0

    def directory_exists(self, path: str) -> bool:
        _, _, exit_code = self.execute(f"test -d {path}")
        return exit_code == 0

    def process_alive(self, pid: str) -> bool:
        if not pid:
            return False
        _, _, exit_code = self.execute(f"kill -0 {pid}")
        return exit_code == 0

    def tail_file(self, path: str, lines: int = 200) -> tuple[str, str, int]:
        return self.execute(f"tail -n {lines} {path}")

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

