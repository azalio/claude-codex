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


@pytest.mark.parametrize("inherited", [None, "", "claude-opus-5-5"])
@pytest.mark.parametrize("model", ["gpt-6.1", "gpt-6.1-sol"])
def test_default_context_identity_names_actual_backend(inherited: str | None, model: str) -> None:
    env = {} if inherited is None else {"ANTHROPIC_MODEL": inherited}

    result = launcher._configure_context_identity(env, model)

    assert result == "claude-codex/" + model
    assert env["ANTHROPIC_MODEL"] == result


@pytest.mark.parametrize("model", ["claude-custom", "claude-opus-5-5[1m]"])
def test_context_identity_preserves_explicit_model(model: str) -> None:
    env = {"ANTHROPIC_MODEL": model}

    result = launcher._configure_context_identity(env, "gpt-6.1-sol")

    assert result == model
    assert env["ANTHROPIC_MODEL"] == model


def test_other_upstream_models_use_actual_backend_identity() -> None:
    env: dict[str, str] = {}

    assert launcher._configure_context_identity(env, "gpt-5.4") == "claude-codex/gpt-5.4"
    assert env["ANTHROPIC_MODEL"] == "claude-codex/gpt-5.4"


def test_old_opus_identity_is_replaced_by_actual_backend() -> None:
    env = {"ANTHROPIC_MODEL": "claude-opus-5-5"}

    assert launcher._configure_context_identity(env, "gpt-5.4") == "claude-codex/gpt-5.4"
    assert env["ANTHROPIC_MODEL"] == "claude-codex/gpt-5.4"


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


@pytest.mark.parametrize("form", ["absent", "json", "file", "equals"])
def test_proxy_settings_override_routing_and_preserve_customizations(tmp_path, form) -> None:
    settings = {
        "env": {"ANTHROPIC_BASE_URL": "https://wrong.invalid", "MY_VARIABLE": "keep"},
        "hooks": {"SessionStart": []},
        "permissions": {"allow": ["Read"]},
    }
    raw = json.dumps(settings)
    path = tmp_path / "settings.json"
    path.write_text(raw)
    option = {
        "absent": [],
        "json": ["--settings", raw],
        "file": ["--settings", str(path)],
        "equals": [f"--settings={raw}"],
    }[form]
    overrides = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:1234"}
    with launcher._proxy_settings_args([*option, "-p", "hello"], overrides) as result:
        private = Path(result[0].split("=", 1)[1] if form == "equals" else result[1])
        assert private.stat().st_mode & 0o777 == 0o600
        merged = json.loads(private.read_text())
        assert merged["env"]["ANTHROPIC_BASE_URL"] == overrides["ANTHROPIC_BASE_URL"]
        assert result[-2:] == ["-p", "hello"]
        if form != "absent":
            assert merged["env"]["MY_VARIABLE"] == "keep"
            assert merged["hooks"] == settings["hooks"]
            assert merged["permissions"] == settings["permissions"]
    assert not private.exists()
    assert path.read_text() == raw


def test_proxy_settings_preserve_arguments_after_separator() -> None:
    args = ["--", "--settings", "a literal prompt"]
    with launcher._proxy_settings_args(args, {"ANTHROPIC_BASE_URL": "local"}) as result:
        assert result[2:] == args


@pytest.mark.parametrize(
    "configured, expected_reasoning, auxiliary_model, auxiliary_reasoning, expected_auxiliary",
    [
        (None, "medium", None, None, ("gpt-6-luna", "low")),
        ("", "medium", "", "", ("gpt-6-luna", "low")),
        ("xhigh", "xhigh", None, None, ("gpt-6-luna", "low")),
        (None, "medium", None, "high", ("gpt-6-luna", "high")),
        ("xhigh", "xhigh", "gpt-6-sol", "medium", ("gpt-6-sol", "medium")),
    ],
)
@pytest.mark.parametrize("outcome", ["exit", "error", "interrupt"])
def test_launcher_keeps_settings_private_and_cleans_up(
    monkeypatch,
    tmp_path,
    outcome,
    configured,
    expected_reasoning,
    auxiliary_model,
    auxiliary_reasoning,
    expected_auxiliary,
) -> None:
    import io

    monkeypatch.setattr(launcher.shutil, "which", lambda _: "/bin/claude")
    monkeypatch.setattr(launcher.Path, "home", lambda: tmp_path)
    listener = Mock()
    listener.getsockname.return_value = ("127.0.0.1", 1234)
    listener.fileno.return_value = 42
    monkeypatch.setattr(launcher, "_listen_socket", lambda _: listener)
    proxy = Mock()
    proxy.stdout = io.BytesIO()
    if configured is None:
        monkeypatch.delenv("CLAUDE_CODEX_REASONING", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_CODEX_REASONING", configured)

    for key, value in (
        ("CLAUDE_CODEX_AUXILIARY_MODEL", auxiliary_model),
        ("CLAUDE_CODEX_AUXILIARY_REASONING", auxiliary_reasoning),
    ):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    def assert_auxiliary(env):
        assert (
            env["CLAUDE_CODEX_AUXILIARY_MODEL"], env["CLAUDE_CODEX_AUXILIARY_REASONING"]
        ) == expected_auxiliary

    def start_proxy(*args, **kwargs):
        assert kwargs["env"]["CLAUDE_CODEX_REASONING"] == expected_reasoning
        assert_auxiliary(kwargs["env"])
        return proxy

    monkeypatch.setattr(launcher.subprocess, "Popen", start_proxy)
    monkeypatch.setattr(launcher, "_wait", lambda *a: None)
    monkeypatch.setattr(launcher, "_terminate", lambda *a: None)
    monkeypatch.setattr(launcher.atexit, "register", lambda *a: None)
    secret = "synthetic-service-token"
    header_secret = "synthetic-header-token"
    settings = {
        "env": {"MCP_SERVICE_TOKEN": secret},
        "hooks": {"SessionStart": []},
        "permissions": {"allow": ["Read"]},
    }
    original = tmp_path / "settings.json"
    original.write_text(json.dumps(settings))
    monkeypatch.setattr(launcher.sys, "argv", ["claude-codex", "--settings", str(original), "-p", "hello"])
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://wrong.invalid")
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_MANTLE", "1")
    monkeypatch.setenv("CLAUDE_CODE_AUTO_MODE_SERVER", "1")
    monkeypatch.setenv("CLAUDE_CODEX_AUXILIARY_ENDPOINT", "http://localhost:11434/v1/responses")
    monkeypatch.setenv("CLAUDE_CODE_ATTRIBUTION_HEADER", "1")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", f"X-Private: {header_secret}")
    monkeypatch.setattr(launcher, "_picker_settings", lambda port, model: {
        "modelPicker": {"replaceBuiltInOptions": True, "options": [
            {"model": "claude-codex/" + model, "label": model},
        ]},
    })
    captured_paths = []

    def run(command, *, env):
        assert env["CLAUDE_CODEX_REASONING"] == expected_reasoning
        assert_auxiliary(env)
        assert command[0:2] == ["/bin/claude", "--settings"]
        assert command[-2:] == ["-p", "hello"]
        assert not any(secret in arg or header_secret in arg for arg in command)
        private = Path(command[2])
        captured_paths.append(private)
        assert private.is_file()
        assert private.stat().st_mode & 0o777 == 0o600
        merged = json.loads(private.read_text())
        pinned = merged["env"]
        assert_auxiliary(pinned)
        assert pinned["MCP_SERVICE_TOKEN"] == secret
        assert merged["hooks"] == settings["hooks"]
        assert merged["permissions"] == settings["permissions"]
        assert merged["modelPicker"]["replaceBuiltInOptions"] is True
        assert all("gpt" in row["label"] for row in merged["modelPicker"]["options"])
        assert pinned["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1234"
        assert pinned["ANTHROPIC_AUTH_TOKEN"] == "claude-codex-local"
        assert pinned["CLAUDE_CODE_USE_VERTEX"] == "0"
        assert pinned["CLAUDE_CODE_USE_MANTLE"] == "0"
        assert pinned["CLAUDE_CODE_AUTO_MODE_SERVER"] == "0"
        assert pinned["CLAUDE_CODEX_AUXILIARY_ENDPOINT"] == "http://localhost:11434/v1/responses"
        assert pinned["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
        assert pinned["CLAUDE_CODE_GATEWAY_HINT_HEADERS"] == "1"
        assert pinned["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"
        assert pinned["ENABLE_TOOL_SEARCH"] == "false"
        assert pinned["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
        assert pinned["CLAUDE_CODE_GATEWAY_MODEL_DISCOVERY_TIMEOUT_MS"] == "10000"
        assert "X-Session-Id:" in pinned["ANTHROPIC_CUSTOM_HEADERS"]
        assert header_secret in pinned["ANTHROPIC_CUSTOM_HEADERS"]
        assert all(env[key] == value for key, value in pinned.items() if key != "MCP_SERVICE_TOKEN")
        if outcome == "error":
            raise RuntimeError("synthetic launch failure")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        return Mock(returncode=7)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    expected = {"exit": SystemExit, "error": RuntimeError, "interrupt": KeyboardInterrupt}[outcome]
    with pytest.raises(expected) as exc:
        launcher.main()
    if outcome == "exit":
        assert exc.value.code == 7
    assert len(captured_paths) == 1
    assert not captured_paths[0].exists()
    assert json.loads(original.read_text()) == settings


def test_picker_settings_replaces_builtin_models_with_catalog(monkeypatch):
    class CatalogResponse(Response):
        def read(self):
            return json.dumps({"data": [
                {"id": "claude-codex/gpt-first", "display_name": "GPT First"},
                {"id": "claude-opus-5-5", "display_name": "Misleading fallback"},
            ]}).encode()

    monkeypatch.setattr(launcher.urllib.request, "urlopen", lambda *args, **kwargs: CatalogResponse(""))
    picker = launcher._picker_settings(1234, "gpt-first")["modelPicker"]
    assert picker["replaceBuiltInOptions"] is True
    assert picker["options"] == [{
        "model": "claude-codex/gpt-first", "label": "GPT First",
        "description": "ChatGPT subscription",
    }]


def test_picker_failure_still_hides_anthropic_models(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("unavailable")
    monkeypatch.setattr(launcher.urllib.request, "urlopen", unavailable)
    picker = launcher._picker_settings(1234, "gpt-first")["modelPicker"]
    assert picker["replaceBuiltInOptions"] is True
    assert picker["options"][0]["model"] == "claude-codex/gpt-first"


def test_picker_overrides_are_temporary_and_preserve_other_settings():
    original = {"modelPicker": {"options": [{"model": "opus"}]}, "hooks": {}}
    args = ["--settings", json.dumps(original)]
    picker = {"modelPicker": {"replaceBuiltInOptions": True, "options": [
        {"model": "claude-codex/gpt-first", "label": "GPT First"},
    ]}}
    with launcher._proxy_settings_args(args, {}, picker) as result:
        settings = json.loads(Path(result[1]).read_text())
        assert settings["modelPicker"] == picker["modelPicker"]
        assert settings["hooks"] == {}
    assert json.loads(args[1]) == original
