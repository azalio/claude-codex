from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

import claude_codex.launcher as launcher
from claude_codex.launcher import _listen_socket, _terminate, _wait

IGNORE_SIGTERM = (
    "import signal, time; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "print('ready', flush=True); "
    "time.sleep(60)"
)


def response_for(startup_id: str):
    def open_url(*args: Any, **kwargs: Any) -> Response:
        del args, kwargs
        return Response(startup_id)

    return open_url


class Response:
    status = 200

    def __init__(self, startup_id: str) -> None:
        self.startup_id = startup_id

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self) -> bytes:
        return json.dumps({"startup_id": self.startup_id}).encode()


def test_gpt_5_6_uses_standard_claude_context_identity() -> None:
    env: dict[str, str] = {}

    result = launcher._configure_context_identity(env, "gpt-5.6-sol")

    assert result == "claude-opus-4-8"
    assert env["ANTHROPIC_MODEL"] == result


def test_context_identity_preserves_explicit_model() -> None:
    env = {"ANTHROPIC_MODEL": "claude-custom"}

    result = launcher._configure_context_identity(env, "gpt-5.6-sol")

    assert result == "claude-custom"
    assert env["ANTHROPIC_MODEL"] == "claude-custom"


def test_other_upstream_models_do_not_set_claude_context_identity() -> None:
    env: dict[str, str] = {}

    assert launcher._configure_context_identity(env, "gpt-5.4") is None
    assert "ANTHROPIC_MODEL" not in env


def test_log_max_bytes_uses_default_for_invalid_values(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_CODEX_LOG_MAX_BYTES", raising=False)
    assert launcher._log_max_bytes() == launcher.DEFAULT_LOG_MAX_BYTES

    monkeypatch.setenv("CLAUDE_CODEX_LOG_MAX_BYTES", "invalid")
    assert launcher._log_max_bytes() == launcher.DEFAULT_LOG_MAX_BYTES

    monkeypatch.setenv("CLAUDE_CODEX_LOG_MAX_BYTES", "0")
    assert launcher._log_max_bytes() == launcher.DEFAULT_LOG_MAX_BYTES


def test_log_max_bytes_accepts_positive_override(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CODEX_LOG_MAX_BYTES", "2048")

    assert launcher._log_max_bytes() == 2048


def test_append_rotated_log_preserves_one_previous_file(tmp_path: Path) -> None:
    log_path = tmp_path / "proxy.log"
    backup_path = tmp_path / "proxy.log.1"
    log_path.write_bytes(b"abcdef")
    backup_path.write_bytes(b"older log")

    launcher._append_rotated_log(log_path, b"ghijkl", max_bytes=8)

    assert backup_path.read_bytes() == b"abcdefgh"
    assert log_path.read_bytes() == b"ijkl"


def test_append_rotated_log_keeps_file_below_limit(tmp_path: Path) -> None:
    log_path = tmp_path / "proxy.log"
    log_path.write_bytes(b"small")

    launcher._append_rotated_log(log_path, b"er", max_bytes=8)

    assert log_path.read_bytes() == b"smaller"
    assert not (tmp_path / "proxy.log.1").exists()


def test_drain_proxy_output_writes_partial_pipe_data_before_eof(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    source = os.fdopen(read_fd, "rb")
    log_path = tmp_path / "proxy.log"
    drain = threading.Thread(
        target=launcher._drain_proxy_output,
        args=(source, log_path),
        kwargs={"max_bytes": 1024},
    )
    drain.start()
    try:
        os.write(write_fd, b"partial")
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if log_path.exists() and log_path.read_bytes() == b"partial":
                break
            time.sleep(0.01)
        assert log_path.read_bytes() == b"partial"
    finally:
        os.close(write_fd)
        drain.join(timeout=1)


def test_listen_socket_reserves_port() -> None:
    listener = _listen_socket()
    try:
        port = listener.getsockname()[1]
        with pytest.raises(OSError):
            _listen_socket(port)
    finally:
        listener.close()


def test_wait_rejects_unrelated_health_server(monkeypatch, tmp_path: Path) -> None:
    process = Mock(spec=subprocess.Popen)
    process.poll.side_effect = [None, 1]
    monkeypatch.setattr("claude_codex.launcher.urllib.request.urlopen", response_for("other"))
    monkeypatch.setattr("claude_codex.launcher.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Proxy exited"):
        _wait(8111, process, tmp_path / "proxy.log", "expected")


def test_wait_accepts_matching_proxy(monkeypatch, tmp_path: Path) -> None:
    process = Mock(spec=subprocess.Popen)
    process.poll.return_value = None
    monkeypatch.setattr("claude_codex.launcher.urllib.request.urlopen", response_for("expected"))

    _wait(8111, process, tmp_path / "proxy.log", "expected")


def test_terminate_kills_proxy() -> None:
    proxy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    _terminate(proxy)
    assert proxy.poll() is not None


def test_terminate_force_kills_despite_interrupt(monkeypatch) -> None:
    # Proxy ignores SIGTERM (mimics uvicorn blocking on an in-flight stream), and
    # the user mashes Ctrl+C during teardown: it must still be SIGKILLed.
    proxy = subprocess.Popen(
        [sys.executable, "-c", IGNORE_SIGTERM],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert proxy.stdout is not None
    assert proxy.stdout.readline() == "ready\n"
    real_wait = proxy.wait
    interrupts = {"left": 3}

    def flaky_wait(timeout=None):
        if interrupts["left"] > 0:
            interrupts["left"] -= 1
            raise KeyboardInterrupt
        return real_wait(timeout=timeout)

    monkeypatch.setattr(proxy, "wait", flaky_wait)
    try:
        _terminate(proxy, grace=0.3)
        assert proxy.poll() is not None
        assert interrupts["left"] == 0  # the interrupts really did fire during teardown
    finally:
        with suppress(ProcessLookupError):
            os.killpg(proxy.pid, signal.SIGKILL)
