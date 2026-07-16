from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024


def _listen_socket(port: int = 0) -> socket.socket:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen()
    listener.set_inheritable(True)
    return listener


def _wait(port: int, process: subprocess.Popen, log_path: Path, startup_id: str) -> None:
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError(f"Proxy exited during startup; see {log_path}")
        try:
            with urllib.request.urlopen(url, timeout=0.25) as response:
                payload = json.loads(response.read())
                if response.status == 200 and payload.get("startup_id") == startup_id:
                    return
        except (OSError, ValueError, json.JSONDecodeError):
            time.sleep(0.1)
    raise RuntimeError(f"Proxy did not become ready; see {log_path}")


def _terminate(proxy: subprocess.Popen, grace: float = 3.0) -> None:
    """Stop the proxy's process group. Idempotent and resilient to repeated
    SIGINT, so Ctrl+C (even mashed) never leaves the proxy orphaned."""
    if proxy.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(proxy.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while proxy.poll() is None and time.monotonic() < deadline:
        # A second Ctrl+C surfaces here as KeyboardInterrupt, not TimeoutExpired;
        # swallow both so the SIGKILL backstop below is never skipped.
        with suppress(subprocess.TimeoutExpired, KeyboardInterrupt):
            proxy.wait(timeout=0.2)
    if proxy.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(proxy.pid, signal.SIGKILL)
        with suppress(subprocess.TimeoutExpired, KeyboardInterrupt):
            proxy.wait(timeout=grace)


def _configure_context_identity(env: dict[str, str], model: str) -> str | None:
    explicit = env.get("ANTHROPIC_MODEL")
    if explicit:
        return explicit
    if model == "gpt-5.6" or model.startswith("gpt-5.6-"):
        env["ANTHROPIC_MODEL"] = "claude-opus-4-8"
        return env["ANTHROPIC_MODEL"]
    return None


def _log_max_bytes() -> int:
    raw = os.environ.get("CLAUDE_CODEX_LOG_MAX_BYTES")
    try:
        limit = int(raw) if raw is not None else DEFAULT_LOG_MAX_BYTES
    except ValueError:
        return DEFAULT_LOG_MAX_BYTES
    return limit if limit > 0 else DEFAULT_LOG_MAX_BYTES


def _append_rotated_log(log_path: Path, data: bytes, *, max_bytes: int) -> None:
    if max_bytes < 1:
        return
    while data:
        try:
            size = log_path.stat().st_size
        except FileNotFoundError:
            size = 0
        except OSError:
            return
        if size >= max_bytes:
            try:
                log_path.replace(log_path.with_name(f"{log_path.name}.1"))
            except (FileNotFoundError, OSError):
                return
            size = 0
        chunk_size = min(max_bytes - size, len(data))
        try:
            with log_path.open("ab") as log:
                log.write(data[:chunk_size])
        except OSError:
            return
        data = data[chunk_size:]


def _drain_proxy_output(source: BinaryIO, log_path: Path, *, max_bytes: int) -> None:
    with source:
        while chunk := source.read1(8192):
            _append_rotated_log(log_path, chunk, max_bytes=max_bytes)


def main() -> None:
    claude = shutil.which("claude")
    if not claude:
        raise SystemExit("claude executable not found in PATH")
    listener = _listen_socket(int(os.environ.get("CLAUDE_CODEX_PORT", "0")))
    port = int(listener.getsockname()[1])
    state = Path.home() / ".local" / "state" / "claude-codex"
    state.mkdir(parents=True, exist_ok=True)
    log_path = state / "proxy.log"
    log_max_bytes = _log_max_bytes()
    startup_id = uuid.uuid4().hex
    try:
        proxy = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "claude_codex.proxy",
                "--fd",
                str(listener.fileno()),
                "--startup-id",
                startup_id,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(listener.fileno(),),
        )
    finally:
        listener.close()
    if proxy.stdout is None:
        _terminate(proxy)
        raise RuntimeError("Proxy stdout pipe was not created")
    log_thread = threading.Thread(
        target=_drain_proxy_output,
        args=(proxy.stdout, log_path),
        kwargs={"max_bytes": log_max_bytes},
        daemon=True,
    )
    log_thread.start()
    # Backstop: guarantees teardown even if the finally below is bypassed.
    atexit.register(_terminate, proxy)
    try:
        _wait(port, proxy, log_path, startup_id)
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
        context_identity = _configure_context_identity(env, model)
        context_note = f"; Claude context {context_identity}" if context_identity else ""
        print(
            f"Claude Code → Codex subscription ({model}){context_note}; proxy 127.0.0.1:{port}",
            file=sys.stderr,
        )
        result = subprocess.run([claude, *sys.argv[1:]], env=env)
        raise SystemExit(result.returncode)
    finally:
        _terminate(proxy)
        log_thread.join(timeout=1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # The finally in main() has already torn the proxy down; exit cleanly
        # (128 + SIGINT) instead of dumping a traceback.
        sys.exit(130)
