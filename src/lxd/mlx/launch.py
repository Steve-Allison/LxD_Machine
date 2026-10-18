"""Start a localhost MLX subprocess and wait until its port accepts connections."""

import json
import socket
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

_launch_lock = threading.Lock()


def port_open(host: str, port: int, *, timeout_secs: float = 1.0) -> bool:
    """Return whether ``host:port`` accepts a TCP connection."""
    try:
        with socket.create_connection((host, port), timeout=timeout_secs):
            return True
    except OSError:
        return False


def ensure_subprocess(
    *,
    host: str,
    port: int,
    command: list[str],
    log_path: Path,
    pid_path: Path,
    startup_timeout_secs: int,
    label: str,
    env: dict[str, str] | None = None,
) -> None:
    """Start ``command`` when ``host:port`` is closed.

    The child loads its model before it binds, so a successful TCP connect
    means the server is ready to take requests. A child that exits is
    reported immediately from the log tail instead of waiting out the
    startup timeout.
    """
    with _launch_lock:
        if port_open(host, port):
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
        pid_path.write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "command": command,
                    "log_path": str(log_path),
                    "started_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if _wait_until_ready(
            host,
            port,
            timeout_secs=startup_timeout_secs,
            process=process,
        ):
            return
        exit_code = process.poll()
        tail = _tail_log(log_path)
        if exit_code is None:
            raise RuntimeError(
                f"{label} was started but did not accept connections on {host}:{port} "
                f"within {startup_timeout_secs} seconds. Check {log_path}. Last log lines:\n{tail}"
            )
        raise RuntimeError(
            f"{label} exited with code {exit_code}. Check {log_path}. Last log lines:\n{tail}"
        )


def _wait_until_ready(
    host: str,
    port: int,
    *,
    timeout_secs: int,
    process: subprocess.Popen[bytes],
) -> bool:
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        if port_open(host, port):
            return True
        time.sleep(0.5)
    return False


def _tail_log(log_path: Path, *, lines: int = 20) -> str:
    if not log_path.exists():
        return "<no log output>"
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = content[-lines:]
    return "\n".join(tail) if tail else "<no log output>"
