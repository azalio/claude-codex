from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(port: int, process: subprocess.Popen, log_path: Path) -> None:
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError(f"Proxy exited during startup; see {log_path}")
        try:
            with urllib.request.urlopen(url, timeout=0.25) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Proxy did not become ready; see {log_path}")


def main() -> None:
    claude = shutil.which("claude")
    if not claude:
        raise SystemExit("claude executable not found in PATH")
    port = int(os.environ.get("CLAUDE_CODEX_PORT") or _free_port())
    state = Path.home() / ".local" / "state" / "claude-codex"
    state.mkdir(parents=True, exist_ok=True)
    log_path = state / "proxy.log"
    log = log_path.open("a")
    proxy = subprocess.Popen(
        [sys.executable, "-m", "claude_codex.proxy", "--port", str(port)],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait(port, proxy, log_path)
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
        env["ANTHROPIC_AUTH_TOKEN"] = "claude-codex-local"
        env["DISABLE_TELEMETRY"] = "1"
        env["DISABLE_ERROR_REPORTING"] = "1"
        session_id = str(uuid.uuid4())
        header = f"X-Session-Id: {session_id}"
        existing = env.get("ANTHROPIC_CUSTOM_HEADERS")
        env["ANTHROPIC_CUSTOM_HEADERS"] = f"{existing}\n{header}" if existing else header
        model = env.get("CLAUDE_CODEX_MODEL", "gpt-5.6-sol")
        print(f"Claude Code → Codex subscription ({model}); proxy 127.0.0.1:{port}", file=sys.stderr)
        result = subprocess.run([claude, *sys.argv[1:]], env=env)
        raise SystemExit(result.returncode)
    finally:
        if proxy.poll() is None:
            os.killpg(proxy.pid, signal.SIGTERM)
            try:
                proxy.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proxy.pid, signal.SIGKILL)
        log.close()


if __name__ == "__main__":
    main()
