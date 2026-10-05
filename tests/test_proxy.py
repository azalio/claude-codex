from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from claude_codex.auth import Tokens
from claude_codex.proxy import (
    _compact_at_tokens,
    _compaction_timeout_seconds,
    _remote_compact_enabled,
    create_app,
)


@pytest.fixture(autouse=True)
def isolated_installation_id(monkeypatch, tmp_path: Path) -> Path:
    path = tmp_path / "installation_id"
    monkeypatch.setattr("claude_codex.proxy.INSTALLATION_ID_PATH", path)
    monkeypatch.setattr("claude_codex.proxy.CHECKPOINT_CACHE_PATH", tmp_path / "checkpoints")
    monkeypatch.setattr(
        "claude_codex.auxiliary_log.AUXILIARY_LOG_PATH", tmp_path / "auxiliary/requests.jsonl"
    )
    return path


class FakeAuth:
    async def get(self, *, force_refresh: bool = False, stale_access: str | None = None) -> Tokens:
        del force_refresh, stale_access
        return Tokens("access", "refresh", int(time.time() * 1000) + 60_000, "acc-123", "test")

    def load(self) -> Tokens:
        return Tokens("access", "refresh", int(time.time() * 1000) + 60_000, "acc-123", "test")


@pytest.mark.parametrize("value", [None, "invalid"])
def test_compact_at_tokens_defaults_to_180k(monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv("CLAUDE_CODEX_COMPACT_AT", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", value)

    assert _compact_at_tokens() == 180_000


@pytest.mark.parametrize("value, expected", [("0", 0), ("-1", 0), ("123456", 123_456)])
def test_compact_at_tokens_preserves_overrides(monkeypatch, value, expected) -> None:
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", value)

    assert _compact_at_tokens() == expected


@pytest.mark.parametrize("value", [None, "", "invalid", "0", "-1", "nan", "inf"])
def test_compaction_timeout_defaults_to_fifteen_minutes(monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv("CLAUDE_CODEX_COMPACTION_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_CODEX_COMPACTION_TIMEOUT", value)
    assert _compaction_timeout_seconds() == 900.0


@pytest.mark.parametrize("value", ["0.05", "240", "1800"])
def test_compaction_timeout_preserves_positive_override(monkeypatch, value) -> None:
    monkeypatch.setenv("CLAUDE_CODEX_COMPACTION_TIMEOUT", value)
    assert _compaction_timeout_seconds() == float(value)


def test_remote_compact_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_CODEX_REMOTE_COMPACT", raising=False)
    assert not _remote_compact_enabled()
    monkeypatch.setenv("CLAUDE_CODEX_REMOTE_COMPACT", "true")
    assert _remote_compact_enabled()


async def test_proxy_streams_anthropic_events(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_CODEX_MODEL", raising=False)
    monkeypatch.delenv("CLAUDE_CODEX_REASONING", raising=False)
    captured: dict = {}

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["account"] = request.headers.get("ChatGPT-Account-Id")
        captured["session"] = request.headers.get("session-id")
        captured["thread"] = request.headers.get("thread-id")
        captured["installation"] = request.headers.get("x-codex-installation-id")
        captured["window"] = request.headers.get("x-codex-window-id")
        captured["originator"] = request.headers.get("originator")
        captured["body"] = json.loads(request.content)
        events = [
            {"type": "response.created", "response": {"id": "resp_test"}},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "ok"},
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 3, "output_tokens": 1}},
            },
        ]
        content = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"
            for event in events
        )
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/messages",
            headers={"x-session-id": "session-test"},
            json={
                "model": "claude-opus",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert "event: message_start" in response.text
    assert '"text":"ok"' in response.text
    assert captured["url"] == "https://codex.test/responses"
    assert captured["account"] == "acc-123"
    assert captured["session"] == "session-test"
    assert captured["originator"] == "codex_cli_rs"
    assert captured["thread"]
    assert captured["installation"]
    assert captured["window"]
    assert captured["body"]["model"] == "gpt-6.1-sol"
    assert captured["body"]["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert captured["body"]["prompt_cache_key"] == "session-test"
    assert captured["body"]["client_metadata"] == {
        "x-codex-installation-id": captured["installation"],
        "session_id": captured["session"],
        "thread_id": captured["thread"],
        "x-codex-window-id": captured["window"],
    }
    assert "max_output_tokens" not in captured["body"]
    assert captured["body"]["input"][0]["content"][0]["text"] == "hi"
    await upstream_client.aclose()


async def test_proxy_reuses_persistent_installation_id_across_app_lifetimes(tmp_path: Path) -> None:
    installation_id_path = tmp_path / "installation_id"
    installations: list[str] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        installations.append(request.headers["x-codex-installation-id"])
        event = {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 1, "output_tokens": 1}},
        }
        return httpx.Response(
            200,
            text=f"event: response.completed\ndata: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    for _ in range(2):
        app = create_app(
            auth=FakeAuth(),
            client=upstream_client,
            endpoint="https://codex.test/responses",
            installation_id_path=installation_id_path,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={"model": "claude-opus", "max_tokens": 10, "messages": []},
            )
        assert response.status_code == 200

    assert installations == [installation_id_path.read_text().strip()] * 2
    await upstream_client.aclose()


async def test_proxy_prefers_native_session_over_launcher_session() -> None:
    captured: list[dict[str, object]] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(
            {
                "session": request.headers["session-id"],
                "thread": request.headers["thread-id"],
                "cache_key": body["prompt_cache_key"],
            }
        )
        event = {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 1, "output_tokens": 1}},
        }
        return httpx.Response(
            200,
            text=f"event: response.completed\ndata: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        for native_session in ("native-a", "native-b"):
            response = await client.post(
                "/v1/messages",
                headers={
                    "x-session-id": "launcher-session",
                    "anthropic-session-id": "legacy-session",
                    "X-Claude-Code-Session-Id": native_session,
                    "x-claude-code-request-class": "main",
                    "x-claude-code-future-hint": "opaque-value",
                    "anthropic-beta": "unknown-future-beta",
                },
                json={"model": "claude-opus", "max_tokens": 10, "messages": []},
            )
            assert response.status_code == 200

    assert [entry["session"] for entry in captured] == ["native-a", "native-b"]
    assert [entry["cache_key"] for entry in captured] == ["native-a", "native-b"]
    assert captured[0]["thread"] != captured[1]["thread"]
    await upstream_client.aclose()


async def test_proxy_reuses_complete_codex_cache_identity_for_claude_session() -> None:
    """A follow-up must carry the whole Codex cache-routing contract.

    The backend cache is scoped by more than ``prompt_cache_key``. Model its
    routing key here so a regression in any identity header/body field turns
    the second request into a miss.
    """

    captured: list[dict[str, dict[str, Any]]] = []
    warmed_cache_keys: set[tuple[str, ...]] = set()

    async def upstream(request: httpx.Request) -> httpx.Response:
        headers = {
            name: request.headers[name]
            for name in (
                "originator",
                "user-agent",
                "x-codex-installation-id",
                "session-id",
                "thread-id",
                "x-client-request-id",
                "x-codex-window-id",
            )
        }
        body = json.loads(request.content)
        assert headers["originator"] == "codex_cli_rs"
        assert headers["user-agent"].startswith("Codex/")
        assert headers["x-client-request-id"] == headers["thread-id"]
        assert body["prompt_cache_key"] == headers["session-id"]
        assert body["client_metadata"] == {
            "x-codex-installation-id": headers["x-codex-installation-id"],
            "session_id": headers["session-id"],
            "thread_id": headers["thread-id"],
            "x-codex-window-id": headers["x-codex-window-id"],
        }

        cache_key = (*headers.values(), body["prompt_cache_key"])
        cached_tokens = 3072 if cache_key in warmed_cache_keys else 0
        warmed_cache_keys.add(cache_key)
        captured.append({"headers": headers, "body": body})
        event = {
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 4096,
                    "input_tokens_details": {"cached_tokens": cached_tokens},
                    "output_tokens": 1,
                }
            },
        }
        return httpx.Response(
            200,
            text=f"event: response.completed\ndata: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        responses = []
        for session_id in ("session-a", "session-a", "session-b"):
            responses.append(
                await client.post(
                    "/v1/messages",
                    headers={"x-session-id": session_id},
                    json={"model": "claude-opus", "max_tokens": 10, "messages": []},
                )
            )

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert [response.json()["usage"] for response in responses] == [
        {"input_tokens": 4096, "output_tokens": 1},
        {"input_tokens": 1024, "cache_read_input_tokens": 3072, "output_tokens": 1},
        {"input_tokens": 4096, "output_tokens": 1},
    ]

    first_headers = captured[0]["headers"]
    second_headers = captured[1]["headers"]
    third_headers = captured[2]["headers"]
    assert first_headers == second_headers
    assert first_headers != third_headers
    assert first_headers["session-id"] == "session-a"
    assert third_headers["session-id"] == "session-b"
    assert first_headers["x-codex-installation-id"] == third_headers["x-codex-installation-id"]
    assert first_headers["thread-id"] != third_headers["thread-id"]
    assert first_headers["x-codex-window-id"] != third_headers["x-codex-window-id"]
    await upstream_client.aclose()


async def test_proxy_compacts_context_and_reuses_replacement_history(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", "200")
    monkeypatch.setenv("CLAUDE_CODEX_REMOTE_COMPACT", "1")
    normal_inputs: list[list[dict[str, object]]] = []
    compact_inputs: list[dict[str, object]] = []
    normal_windows: list[str] = []
    replacement_history = [
        {"role": "system", "content": [{"type": "input_text", "text": "do not forward"}]},
        {"role": "developer", "content": [{"type": "input_text", "text": "do not forward"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "summary"}]},
    ]
    usable_replacement_history = [replacement_history[-1]]

    async def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/compact"):
            compact_inputs.append(body)
            return httpx.Response(200, json={"output": replacement_history})

        input_items = body["input"]
        normal_inputs.append(input_items)
        normal_windows.append(request.headers["x-codex-window-id"])
        input_tokens = 200 if len(normal_inputs) == 1 else 10
        event = {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": input_tokens, "output_tokens": 1}},
        }
        return httpx.Response(
            200,
            text=f"event: response.completed\ndata: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    first_messages = [{"role": "user", "content": "first"}]
    compacted_messages = [
        *first_messages,
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "second"},
    ]
    after_compact_messages = [
        *compacted_messages,
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "third"},
    ]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        for messages in (first_messages, compacted_messages, after_compact_messages):
            response = await client.post(
                "/v1/messages",
                headers={"anthropic-session-id": "compact-session"},
                json={"model": "claude-opus", "max_tokens": 10, "stream": True, "messages": messages},
            )
            assert response.status_code == 200

    assert len(compact_inputs) == 1
    assert compact_inputs[0]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "first"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "answer one"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "second"}]},
    ]
    assert compact_inputs[0]["model"] == "gpt-6.1-sol"
    assert compact_inputs[0]["prompt_cache_key"] == "compact-session"
    assert normal_inputs[2] == usable_replacement_history + [
        {"role": "assistant", "content": [{"type": "output_text", "text": "answer two"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "third"}]},
    ]
    assert normal_windows[0] != normal_windows[1]
    assert normal_windows[1] == normal_windows[2]
    log = capsys.readouterr().out
    assert "codex_compact" in log
    assert "result=started input_tokens=200 threshold=200" in log
    assert "result=success implementation=remote" in log
    assert "replacement_items=1" in log
    await upstream_client.aclose()


async def test_proxy_falls_back_to_local_compact_after_remote_disconnect(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", "100")
    monkeypatch.setenv("CLAUDE_CODEX_REMOTE_COMPACT", "1")
    normal_inputs: list[list[dict[str, object]]] = []
    remote_compact_calls = 0
    local_compact_calls = 0
    windows: list[str] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal remote_compact_calls, local_compact_calls
        if request.url.path.endswith("/compact"):
            remote_compact_calls += 1
            raise httpx.RemoteProtocolError("Server disconnected without sending a response")

        body = json.loads(request.content)
        input_items = body["input"]
        last_content = input_items[-1].get("content") if input_items else []
        if (
            isinstance(last_content, list)
            and last_content
            and last_content[0].get("text", "").startswith("You are performing a context checkpoint")
        ):
            local_compact_calls += 1
            assert "text" not in body
            events = [
                {"type": "response.output_text.delta", "output_index": 0, "delta": "checkpoint"},
                {"type": "response.completed", "response": {"usage": {"input_tokens": 100}}},
            ]
        else:
            assert body["text"]["format"]["type"] == "json_schema"
            normal_inputs.append(input_items)
            windows.append(request.headers["x-codex-window-id"])
            input_tokens = 100 if len(normal_inputs) == 1 else 10
            events = [
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": input_tokens, "output_tokens": 1}},
                }
            ]
        content = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    first_messages = [{"role": "user", "content": "first"}]
    second_messages = [
        *first_messages,
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "second"},
    ]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        for messages in (first_messages, second_messages):
            response = await client.post(
                "/v1/messages",
                headers={"anthropic-session-id": "fallback-session"},
                json={
                    "model": "claude-opus",
                    "max_tokens": 10,
                    "stream": True,
                    "messages": messages,
                    "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
                },
            )
            assert response.status_code == 200

    assert remote_compact_calls == 1
    assert local_compact_calls == 1
    assert normal_inputs[1] == [
        {"role": "user", "content": [{"type": "input_text", "text": "first"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "second"}]},
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Context checkpoint summary:\ncheckpoint"}],
        },
    ]
    assert windows[0] != windows[1]
    log = capsys.readouterr().out
    assert "result=remote_unavailable error=RemoteProtocolError" in log
    assert "result=success implementation=local" in log
    assert "replacement_items=3" in log
    await upstream_client.aclose()


async def test_proxy_isolates_compaction_branches_within_launcher_session(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", "200")
    monkeypatch.setenv("CLAUDE_CODEX_REMOTE_COMPACT", "1")
    normal_inputs: list[list[dict[str, object]]] = []
    normal_threads: list[str] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/compact"):
            text = json.dumps(body["input"])
            summary = "summary-a" if "a2" in text else "summary-b"
            return httpx.Response(
                200,
                json={"output": [{"role": "user", "content": [{"type": "input_text", "text": summary}]}]},
            )

        input_items = body["input"]
        normal_inputs.append(input_items)
        normal_threads.append(request.headers["thread-id"])
        initial_user_text = (
            input_items[0]["content"][0].get("text")
            if len(input_items) == 1 and input_items and isinstance(input_items[0].get("content"), list)
            else None
        )
        input_tokens = 200 if initial_user_text in {"a", "b"} else 10
        event = {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": input_tokens, "output_tokens": 1}},
        }
        return httpx.Response(
            200,
            text=f"event: response.completed\ndata: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    branch_a1 = [{"role": "user", "content": "a"}]
    branch_a2 = [*branch_a1, {"role": "assistant", "content": "answer-a1"}, {"role": "user", "content": "a2"}]
    branch_a3 = [*branch_a2, {"role": "assistant", "content": "answer-a2"}, {"role": "user", "content": "a3"}]
    branch_b1 = [{"role": "user", "content": "b"}]
    branch_b2 = [*branch_b1, {"role": "assistant", "content": "answer-b1"}, {"role": "user", "content": "b2"}]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        for messages in (branch_a1, branch_a2, branch_b1, branch_b2, branch_a3):
            response = await client.post(
                "/v1/messages",
                headers={"x-session-id": "shared-launcher-session"},
                json={"model": "claude-opus", "max_tokens": 10, "stream": True, "messages": messages},
            )
            assert response.status_code == 200

    assert normal_inputs[1] == [{"role": "user", "content": [{"type": "input_text", "text": "summary-a"}]}]
    assert normal_inputs[3] == [{"role": "user", "content": [{"type": "input_text", "text": "summary-b"}]}]
    assert normal_inputs[4] == [
        {"role": "user", "content": [{"type": "input_text", "text": "summary-a"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "answer-a2"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "a3"}]},
    ]
    assert normal_threads[1] == normal_threads[4]
    assert normal_threads[1] != normal_threads[3]
    assert "result=discarded reason=history_changed" not in capsys.readouterr().out
    await upstream_client.aclose()


async def test_nonstream_surfaces_cached_input_usage(capsys, isolated_installation_id: Path) -> None:
    async def upstream(_: httpx.Request) -> httpx.Response:
        events = [
            {"type": "response.created", "response": {"id": "resp_test"}},
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 4096,
                        "input_tokens_details": {"cached_tokens": 3072, "cache_write_tokens": 1024},
                        "output_tokens": 2,
                    },
                },
            },
        ]
        content = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"
            for event in events
        )
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={"model": "claude-opus", "max_tokens": 10, "messages": []},
        )

    assert response.status_code == 200
    assert response.json()["usage"] == {
        "input_tokens": 0,
        "cache_read_input_tokens": 3072,
        "cache_creation_input_tokens": 1024,
        "output_tokens": 2,
    }
    assert "".join(line for line in capsys.readouterr().out.splitlines(keepends=True)
                   if line.startswith("codex_cache ")) == (
        "codex_cache source=upstream "
        f"client_id={isolated_installation_id.read_text().strip()} "
        "session_source=default session_id=claude-codex result=hit input_tokens=4096 "
        "cached_tokens=3072 cache_write_tokens=1024\n"
    )
    await upstream_client.aclose()


async def test_nonstream_marks_unreported_cache_write_usage(capsys, isolated_installation_id: Path) -> None:
    async def upstream(_: httpx.Request) -> httpx.Response:
        event = {
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 4096,
                    "input_tokens_details": {"cached_tokens": 3072},
                    "output_tokens": 2,
                }
            },
        }
        return httpx.Response(
            200,
            text=f"event: response.completed\ndata: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={"model": "claude-opus", "max_tokens": 10, "messages": []},
        )

    assert response.status_code == 200
    assert response.json()["usage"] == {
        "input_tokens": 1024,
        "cache_read_input_tokens": 3072,
        "output_tokens": 2,
    }
    assert "".join(line for line in capsys.readouterr().out.splitlines(keepends=True)
                   if line.startswith("codex_cache ")) == (
        "codex_cache source=upstream "
        f"client_id={isolated_installation_id.read_text().strip()} "
        "session_source=default session_id=claude-codex result=hit input_tokens=4096 "
        "cached_tokens=3072 cache_write_tokens=unreported\n"
    )
    await upstream_client.aclose()


async def test_streaming_backend_error_preserves_http_status() -> None:
    async def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream failed")

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={"model": "claude-opus", "max_tokens": 10, "stream": True, "messages": []},
        )

    assert response.status_code == 500
    assert response.json()["error"]["message"] == "upstream failed"
    await upstream_client.aclose()


async def test_retries_transport_disconnect_before_first_sse_event() -> None:
    calls = 0

    async def upstream(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.RemoteProtocolError("disconnected")
        event = {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 3, "output_tokens": 1}},
        }
        return httpx.Response(
            200,
            text=f"event: response.completed\ndata: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={"model": "claude-opus", "max_tokens": 10, "messages": []},
        )

    assert response.status_code == 200
    assert calls == 2
    await upstream_client.aclose()


async def test_streaming_backend_429_is_rate_limit_error() -> None:
    async def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"error":{"message":"usage limit"}}')

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={"model": "claude-opus", "max_tokens": 10, "stream": True, "messages": []},
        )

    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"
    await upstream_client.aclose()


async def test_nonstream_backend_429_maps_status_and_retry_after() -> None:
    async def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"error":{"message":"usage limit"}}', headers={"retry-after": "42"})

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://proxy.test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={"model": "claude-opus", "max_tokens": 10, "messages": []},
        )

    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"
    assert response.headers["retry-after"] == "42"
    await upstream_client.aclose()


async def test_nonstream_transport_error_is_an_anthropic_error() -> None:
    calls = 0

    async def upstream(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.RemoteProtocolError("Server disconnected without sending a response")

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={"model": "claude-opus", "max_tokens": 10, "messages": []},
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"
    assert calls == 2
    await upstream_client.aclose()


async def test_count_tokens_endpoint() -> None:
    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    app = create_app(auth=FakeAuth(), client=upstream_client)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        ok = await client.post(
            "/v1/messages/count_tokens",
            json={"model": "claude-opus", "messages": [{"role": "user", "content": "hello"}]},
        )
        bad = await client.post("/v1/messages/count_tokens", json=[])

    assert ok.status_code == 200
    assert ok.json()["input_tokens"] > 0
    assert bad.status_code == 400
    assert bad.json()["error"]["type"] == "invalid_request_error"
    await upstream_client.aclose()


async def test_stream_pings_during_upstream_gap(monkeypatch) -> None:
    monkeypatch.setattr("claude_codex.proxy.PING_INTERVAL_SECONDS", 0.02)

    class PausedBody(httpx.AsyncByteStream):
        def __init__(self, content: str) -> None:
            self.content = content.encode()

        async def __aiter__(self):
            # Headers уже получены; reasoning задерживает только SSE body.
            await asyncio.sleep(0.1)
            yield self.content

    async def upstream(_: httpx.Request) -> httpx.Response:
        events = [
            {"type": "response.created", "response": {"id": "resp_test"}},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "ok"},
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 3, "output_tokens": 1}},
            },
        ]
        content = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"
            for event in events
        )
        return httpx.Response(200, stream=PausedBody(content), headers={"content-type": "text/event-stream"})

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={"model": "claude-opus", "max_tokens": 10, "stream": True, "messages": []},
        )

    assert response.status_code == 200
    assert "event: ping" in response.text
    assert '"text":"ok"' in response.text
    assert "event: message_stop" in response.text
    await upstream_client.aclose()


async def test_rejects_invalid_request_shape() -> None:
    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    app = create_app(auth=FakeAuth(), client=upstream_client)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post("/v1/messages", json=[])

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    await upstream_client.aclose()


async def test_nonstream_failed_response_is_error() -> None:
    async def upstream(_: httpx.Request) -> httpx.Response:
        event = {
            "type": "response.failed",
            "response": {"error": {"message": "backend failed"}},
        }
        return httpx.Response(
            200,
            text=f"event: response.failed\ndata: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_app(auth=FakeAuth(), client=upstream_client, endpoint="https://codex.test/responses")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={"model": "claude-opus", "max_tokens": 10, "messages": []},
        )

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "backend failed"
    await upstream_client.aclose()


async def test_proxy_isolates_agents_and_forwards_open_hint_headers() -> None:
    captured = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append((dict(request.headers), json.loads(request.content)))
        return httpx.Response(200, text='data: {"type":"response.completed"}\n\n')

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            for agent in ("worker-a", "worker-b", "worker-a"):
                response = await client.post(
                    "/v1/messages?beta=true",
                    headers={
                        "X-Claude-Code-Session-Id": "shared-session",
                        "x-claude-code-agent-id": agent,
                        "x-claude-code-parent-agent-id": "parent",
                        "x-claude-code-future-hint": "opaque",
                        "authorization": "Bearer local-credential",
                        "x-api-key": "local-key",
                    },
                    json={"messages": [{"role": "user", "content": "same prefix"}]},
                )
                assert response.status_code == 200

    first, second, repeated = captured
    assert first[0]["thread-id"] != second[0]["thread-id"]
    assert first[0]["thread-id"] == repeated[0]["thread-id"]
    assert first[1]["prompt_cache_key"] != second[1]["prompt_cache_key"]
    assert first[1]["prompt_cache_key"] == repeated[1]["prompt_cache_key"]
    assert first[0]["session-id"] == second[0]["session-id"] == "shared-session"
    assert first[0]["x-claude-code-agent-id"] == "worker-a"
    assert first[0]["x-claude-code-parent-agent-id"] == "parent"
    assert first[0]["x-claude-code-future-hint"] == "opaque"
    assert first[0]["authorization"] == "Bearer access"
    assert "x-api-key" not in first[0]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [400, 403, 404, 429, 529])
async def test_proxy_preserves_backend_rejection_and_retry_headers(stream, status) -> None:
    # Recovery Claude Code ищет исходную формулировку, даже в длинной ошибке.
    message = "details " * 100 + "capability_rejected: prompt_too_long"
    headers = {
        "retry-after": "42",
        "x-should-retry": "false",
        "anthropic-ratelimit-unified-future-status": "restricted",
        "set-cookie": "private=upstream",
    }

    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"error": {"message": message, "code": "context_length_exceeded"}}, headers=headers
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post("/v1/messages", json={"messages": [], "stream": stream})

    assert response.status_code == status
    assert response.json()["error"]["message"] == f"prompt is too long: {message}"
    assert response.json()["error"]["code"] == "context_length_exceeded"
    assert response.headers["retry-after"] == "42"
    assert response.headers["x-should-retry"] == "false"
    assert response.headers["anthropic-ratelimit-unified-future-status"] == "restricted"
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [200, 429])
async def test_response_headers_normalize_retry_date_and_keep_future_limits(stream, status) -> None:
    def upstream(_: httpx.Request) -> httpx.Response:
        headers = {
            "retry-after": format_datetime(datetime.now(UTC) + timedelta(seconds=120), usegmt=True),
            "x-should-retry": "true",
            "anthropic-ratelimit-unified-future-reset": "123456",
        }
        if status == 429:
            return httpx.Response(429, json={"error": {"message": "limited"}}, headers=headers)
        return httpx.Response(200, text='data: {"type":"response.completed"}\n\n', headers=headers)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post("/v1/messages", json={"messages": [], "stream": stream})

    assert response.status_code == status
    assert 119 <= int(response.headers["retry-after"]) <= 120
    assert response.headers["x-should-retry"] == "true"
    assert response.headers["anthropic-ratelimit-unified-future-reset"] == "123456"
    if stream and status == 200:
        assert response.headers["content-type"].startswith("text/event-stream")


@pytest.mark.parametrize(
    "feature",
    [
        {"safeguards": {}},
        {"context_management": {"edits": []}},
        {"tools": [{"type": "advisor_20260301", "name": "advisor"}]},
        {"tools": [{"name": "Read", "defer_loading": True}]},
        {"messages": [{"role": "user", "content": [{"type": "tool_reference", "tool_name": "Read"}]}]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "tool_reference", "tool_name": "Read"}],
                        }
                    ],
                }
            ]
        },
    ],
)
async def test_unsupported_anthropic_capabilities_are_rejected_before_inference(feature) -> None:
    def upstream(_: httpx.Request) -> httpx.Response:
        pytest.fail("Unsupported capability must not reach Codex")

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post("/v1/messages?beta=true", json={"messages": [], **feature})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    if "safeguards" in feature:
        assert "CLAUDE_CODE_AUTO_MODE_SERVER=0" in response.json()["error"]["message"]
    if feature.get("tools", [{}])[0].get("type") == "advisor_20260301":
        assert "Input tag 'advisor_20260301'" in response.json()["error"]["message"]


@pytest.mark.parametrize("stream", [False, True])
async def test_truncated_upstream_is_not_reported_as_a_completed_answer(stream) -> None:
    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='data: {"type":"response.output_text.delta","delta":"partial"}\n\n')

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post("/v1/messages", json={"messages": [], "stream": stream})

    if stream:
        assert "event: error" in response.text
        assert "event: message_stop" not in response.text
        assert "event: message_delta" not in response.text
        assert '"stop_reason":"end_turn"' not in response.text
    else:
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "api_error"


@pytest.mark.parametrize(
    "configured, effort, expected",
    [
        (None, None, "medium"),
        ("", None, "medium"),
        (None, "medium", "medium"),
        (None, "high", "high"),
        (None, "max", "xhigh"),
        ("medium", "max", "medium"),
        ("xhigh", "medium", "xhigh"),
        ("low", "high", "low"),
    ],
)
async def test_output_effort_is_translated_unless_backend_override_is_set(
    monkeypatch, configured, effort, expected
) -> None:
    if configured is None:
        monkeypatch.delenv("CLAUDE_CODEX_REASONING", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_CODEX_REASONING", configured)
    captured = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, text='data: {"type":"response.completed"}\n\n')

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            body = {"messages": []}
            if effort is not None:
                body["output_config"] = {"effort": effort}
            response = await client.post("/v1/messages", json=body)
    assert response.status_code == 200
    assert captured[0]["reasoning"]["effort"] == expected


@pytest.mark.parametrize(
    "request_class, auxiliary_model, auxiliary_effort, expected_model, expected_effort",
    [
        (None, "gpt-6-luna", "low", "gpt-6.1-sol", "medium"),
        ("main", "gpt-6-luna", "low", "gpt-6.1-sol", "medium"),
        ("subagent", "gpt-6-luna", "low", "gpt-6.1-sol", "medium"),
        ("workflow", "gpt-6-luna", "low", "gpt-6.1-sol", "medium"),
        ("compaction", "gpt-6-luna", "low", "gpt-6.1-sol", "medium"),
        ("auxiliary", None, None, "gpt-6.1-sol", "medium"),
        ("auxiliary", "", "", "gpt-6.1-sol", "medium"),
        ("auxiliary", None, "low", "gpt-6.1-sol", "low"),
        ("auxiliary", "gpt-6-luna", None, "gpt-6-luna", "low"),
        ("auxiliary", "gpt-6-luna", "", "gpt-6-luna", "low"),
        ("auxiliary", "gpt-6-luna", "high", "gpt-6-luna", "high"),
    ],
)
async def test_auxiliary_routing_keeps_main_and_other_request_classes_unchanged(
    monkeypatch, request_class, auxiliary_model, auxiliary_effort, expected_model, expected_effort
) -> None:
    monkeypatch.setenv("CLAUDE_CODEX_MODEL", "gpt-6.1-sol")
    monkeypatch.setenv("CLAUDE_CODEX_REASONING", "medium")
    for name, value in (
        ("CLAUDE_CODEX_AUXILIARY_MODEL", auxiliary_model),
        ("CLAUDE_CODEX_AUXILIARY_REASONING", auxiliary_effort),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    captured = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, text='data: {"type":"response.completed"}\n\n')

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            headers = {"x-claude-code-session-id": "routing-test"}
            if request_class is not None:
                headers["x-claude-code-request-class"] = request_class
            response = await client.post(
                "/v1/messages",
                headers=headers,
                json={
                    "model": "claude-sonnet-5",
                    "output_config": {"effort": "max"},
                    "messages": [{"role": "user", "content": "Synthetic routing check."}],
                },
            )
    assert response.status_code == 200
    assert len(captured) == 1
    assert captured[0]["model"] == expected_model
    assert captured[0]["reasoning"]["effort"] == expected_effort


async def test_startup_probe_and_discovery_fallback_to_the_configured_backend(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CODEX_MODEL", "gpt-test-backend")
    monkeypatch.setenv("ANTHROPIC_MODEL", "anthropic-codex-alias")

    def upstream(_: httpx.Request) -> httpx.Response:
        assert _.method == "GET" and _.url.path.endswith("/models")
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            probe = await client.head("/api/hello")
            models = await client.get(
                "/v1/models?limit=1000", headers={"Authorization": "Bearer local", "x-api-key": "local"}
            )

    assert probe.status_code == 204
    assert not probe.content
    assert models.status_code == 200
    assert models.json()["data"] == [
        {
            "type": "model",
            "id": "anthropic-codex-alias",
            "display_name": "Codex: gpt-test-backend",
            "description": (
                "Routes to gpt-test-backend via the Codex subscription; "
                "Anthropic server safeguards are unavailable."
            ),
        }
    ]
    assert models.json()["has_more"] is False


@pytest.mark.parametrize("event_data", ["[]", "not-json"])
@pytest.mark.parametrize("stream", [False, True])
async def test_malformed_upstream_event_is_an_api_error(event_data, stream) -> None:
    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=f"data: {event_data}\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://proxy.test"
        ) as client:
            response = await client.post("/v1/messages", json={"messages": [], "stream": stream})
    if stream:
        assert "event: error" in response.text
        assert "event: message_stop" not in response.text
    else:
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "api_error"


async def test_delta_after_block_stop_is_not_relayed_to_claude() -> None:
    def upstream(_: httpx.Request) -> httpx.Response:
        events = [
            {"type": "response.output_text.delta", "output_index": 0, "delta": "visible"},
            {"type": "response.output_item.done", "output_index": 0, "item": {"type": "message"}},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "late-content"},
            {"type": "response.completed"},
        ]
        return httpx.Response(200, text="".join(f"data: {json.dumps(event)}\n\n" for event in events))

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            response = await client.post("/v1/messages", json={"messages": [], "stream": True})
    assert "event: error" in response.text
    assert "late-content" not in response.text
    assert response.text.count("event: content_block_stop") == 1
    assert "event: message_stop" not in response.text


async def test_auxiliary_classifier_does_not_reuse_main_compaction_state(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", "100")
    monkeypatch.setenv("CLAUDE_CODEX_REMOTE_COMPACT", "0")
    captured = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append((dict(request.headers), json.loads(request.content)))
        tokens = 100 if len(captured) <= 2 else 10
        event = {"type": "response.completed", "response": {"usage": {"input_tokens": tokens}}}
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            for request_class in ("main", "auxiliary", "auxiliary"):
                response = await client.post(
                    "/v1/messages",
                    headers={
                        "x-claude-code-session-id": "shared",
                        "x-claude-code-request-class": request_class,
                    },
                    json={"messages": [{"role": "user", "content": "same history"}], "system": request_class},
                )
                assert response.status_code == 200
    assert len(captured) == 3
    assert captured[0][0]["thread-id"] != captured[1][0]["thread-id"]
    assert captured[0][1]["prompt_cache_key"] != captured[1][1]["prompt_cache_key"]
    assert captured[0][1]["input"] == captured[1][1]["input"]
    assert captured[1][1]["input"] == captured[2][1]["input"]

@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("event_type", ["response.failed", "error", "flat_error"])
async def test_context_overflow_sse_is_a_nonretryable_invalid_request(stream, event_type) -> None:
    message = "Your input exceeds the context window of this model. Please adjust your input and try again."
    error = {"code": "context_length_exceeded", "message": message, "param": "input"}

    def upstream(_: httpx.Request) -> httpx.Response:
        event = {"type": "response.failed" if event_type == "response.failed" else "error"}
        if event_type == "response.failed":
            event["response"] = {"error": error}
        elif event_type == "flat_error":
            event.update(error)
        else:
            event["error"] = error
        return httpx.Response(
            200, text=f"event: {event['type']}\ndata: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://proxy.test"
        ) as client:
            response = await client.post("/v1/messages", json={"messages": [], "stream": stream})
    if stream:
        assert response.status_code == 200
        data = next(json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: "))
        assert "event: message_stop" not in response.text
    else:
        assert response.status_code == 400
        assert response.headers["x-should-retry"] == "false"
        data = response.json()
    assert data["error"]["type"] == "invalid_request_error"
    assert data["error"]["message"].lower().startswith("prompt is too long")
    assert message in data["error"]["message"]
    assert data["error"]["code"] == "context_length_exceeded"
    assert data["error"]["param"] == "input"


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("large_field", ["messages", "system", "tools"])
async def test_compaction_checks_current_request_without_prior_large_usage(
    monkeypatch, restored, large_field
) -> None:
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", "100")
    monkeypatch.delenv("CLAUDE_CODEX_REMOTE_COMPACT", raising=False)
    summaries = []
    inputs = []

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        items = body["input"]
        last = items[-1]
        is_summary = last.get("role") == "user" and last["content"][0]["text"].startswith(
            "You are performing a context checkpoint"
        )
        if is_summary:
            summaries.append(body)
            delta = {"type": "response.output_text.delta", "delta": "handoff"}
            prefix = f"event: response.output_text.delta\ndata: {json.dumps(delta)}\n\n"
        else:
            inputs.append(items)
            prefix = ""
        event = {"type": "response.completed", "response": {"usage": {"input_tokens": 50}}}
        return httpx.Response(
            200, text=prefix + f"event: response.completed\ndata: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    messages = [{"role": "user", "content": "first"}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        app = create_app(auth=FakeAuth(), client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            if not restored:
                response = await client.post("/v1/messages", json={"messages": messages})
                assert response.status_code == 200
                messages.append({"role": "assistant", "content": "answer"})
            body = {"messages": messages}
            if large_field == "messages":
                messages.append({"role": "user", "content": "x" * 800})
            elif large_field == "system":
                body["system"] = "x" * 800
            else:
                body["tools"] = [
                    {"name": "lookup", "description": "x" * 800, "input_schema": {"type": "object"}}
                ]
            response = await client.post("/v1/messages", json=body)
            assert response.status_code == 200
    assert len(summaries) == 1
    assert "x" * 800 in json.dumps(summaries[0])
    assert inputs[-1][-1]["content"][0]["text"] == "Context checkpoint summary:\nhandoff"


@pytest.mark.parametrize("stream", [False, True])
async def test_native_compaction_recovers_history_larger_than_backend_budget(monkeypatch, stream):
    monkeypatch.setattr("claude_codex.proxy.COMPACTION_CHUNK_TOKENS", 512, raising=False)
    leaves = []
    normal = []

    async def upstream(request):
        body = json.loads(request.content)
        size = sum(
            len(json.dumps(body.get(key, ""), ensure_ascii=False)) for key in ("instructions", "input")
        )
        if size > 2048:
            return httpx.Response(
                400, json={"error": {"code": "context_length_exceeded", "message": "too big"}}
            )
        first_text = body["input"][0]["content"][0]["text"] if body["input"] else ""
        if first_text.startswith("Transcript segment "):
            leaves.append(first_text.partition("\n")[2])
        elif first_text.startswith("Checkpoint summaries"):
            pass
        else:
            normal.append(body)
        events = [
            {"type": "response.output_text.delta", "delta": "checkpoint"},
            {"type": "response.completed", "response": {"usage": {"input_tokens": 100}}},
        ]
        return httpx.Response(
            200,
            text="".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events),
        )

    messages = [
        {"role": "user", "content": "important-user-rule:" + "a" * 2800},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"query": "source-evidence"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call-1", "content": "actual-result:" + "b" * 2800}
        ]},
        {"role": "user", "content": "Write a checkpoint preserving all user requirements."},
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:
        app = create_app(auth=FakeAuth(), client=backend)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/v1/messages",
                headers={
                    "x-claude-code-session-id": "native-restore", "x-claude-code-request-class": "compaction",
                },
                json={"system": "native checkpoint instructions", "messages": messages, "stream": stream},
            )
    assert response.status_code == 200
    assert "event: error" not in response.text
    assert len(leaves) > 1
    transcript = json.loads("".join(leaves))
    assert transcript["instructions"] == "native checkpoint instructions"
    assert transcript["input"][0]["content"][0]["text"] == messages[0]["content"]
    assert transcript["input"][1]["call_id"] == "call-1"
    assert transcript["input"][2]["output"] == messages[2]["content"][0]["content"]
    assert normal[-1]["instructions"] == "native checkpoint instructions"
    assert normal[-1]["input"][-1]["content"][0]["text"] == messages[-1]["content"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("phase", ["headers", "body"])
async def test_native_compaction_has_total_deadline(monkeypatch, stream, phase):
    monkeypatch.setattr("claude_codex.proxy.COMPACTION_TIMEOUT_SECONDS", 0.03, raising=False)
    closed = []

    class EndlessBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            try:
                while True:
                    await asyncio.sleep(0.005)
                    yield b": keepalive\n\n"
            finally:
                closed.append(True)

    async def upstream(request):
        if phase == "headers":
            try:
                await asyncio.sleep(10)
            finally:
                closed.append(True)
        return httpx.Response(200, stream=EndlessBody())

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:
        app = create_app(auth=FakeAuth(), client=backend)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://proxy.test"
        ) as client:
            response = await asyncio.wait_for(client.post(
                "/v1/messages", headers={"x-claude-code-request-class": "compaction"},
                json={"messages": [], "stream": stream},
            ), timeout=0.2)
    assert closed
    if stream and phase == "body":
        assert "compaction_timeout" in response.text
        assert "event: message_stop" not in response.text
    else:
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "compaction_timeout"
        assert response.headers["x-should-retry"] == "false"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", ["response.failed", "response.incomplete"])
async def test_failed_checkpoint_cancels_siblings_without_sending_normal_inference(
    monkeypatch, stream, failure
):
    monkeypatch.setattr("claude_codex.proxy.COMPACTION_CHUNK_TOKENS", 512)
    started = asyncio.Event()
    cancelled = []
    requests = []

    async def upstream(request):
        body = json.loads(request.content)
        text = body["input"][0]["content"][0]["text"]
        requests.append(text)
        assert text.startswith("Transcript segment ")
        if text.startswith("Transcript segment 1\n"):
            await started.wait()
            if failure == "response.failed":
                response = {"error": {
                    "code": "context_length_exceeded",
                    "message": "Your input exceeds the context window",
                }}
            else:
                response = {"incomplete_details": {"reason": "max_output_tokens"}}
            events = [
                {"type": "response.output_text.delta", "delta": "partial"},
                {"type": failure, "response": response},
            ]
            return httpx.Response(
                200, text="".join(
                    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
                ),
            )
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:
        app = create_app(auth=FakeAuth(), client=backend)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://proxy.test"
        ) as client:
            response = await asyncio.wait_for(client.post(
                "/v1/messages", headers={"x-claude-code-request-class": "compaction"},
                json={"messages": [{"role": "user", "content": "x" * 8000}], "stream": stream},
            ), timeout=0.5)
    expected = "context_length_exceeded" if failure == "response.failed" else "compaction_incomplete"
    assert response.status_code == 400
    assert response.json()["error"]["code"] == expected
    assert response.headers["x-should-retry"] == "false"
    assert len(requests) > 1
    assert cancelled


@pytest.mark.parametrize("stream", [False, True])
async def test_oversized_image_checkpoint_fails_explicitly_before_inference(monkeypatch, stream):
    monkeypatch.setattr("claude_codex.proxy.COMPACTION_CHUNK_TOKENS", 512)
    requests = []

    async def upstream(request):
        requests.append(request)
        raise AssertionError("Image data must not be silently converted into a textual checkpoint")

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:
        app = create_app(auth=FakeAuth(), client=backend)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/v1/messages", headers={"x-claude-code-request-class": "compaction"},
                json={"stream": stream, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "x" * 8000},
                    {"type": "image", "source": {
                        "type": "url", "url": "https://example.invalid/image.png",
                    }},
                ]}]},
            )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "compaction_images"
    assert not requests


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("large_field", ["instructions", "tools"])
async def test_native_checkpoint_recovers_oversized_execution_context(monkeypatch, stream, large_field):
    monkeypatch.setattr("claude_codex.proxy.COMPACTION_CHUNK_TOKENS", 512)
    leaves = []
    normal_requests = []
    oversized = []

    async def upstream(request):
        body = json.loads(request.content)
        size = sum(
            len(json.dumps(body.get(key, ""), ensure_ascii=False).encode())
            for key in ("instructions", "input", "tools")
        )
        if size > 2048:
            oversized.append(size)
            return httpx.Response(
                400, json={"error": {"code": "context_length_exceeded", "message": "too big"}}
            )
        text = body["input"][0]["content"][0]["text"]
        if text.startswith("Transcript segment "):
            leaves.append(text.partition("\n")[2])
        elif not text.startswith("Checkpoint summaries"):
            normal_requests.append(body)
        events = [
            {"type": "response.output_text.delta", "delta": "checkpoint: CRITICAL_POLICY is preserved"},
            {"type": "response.completed", "response": {}},
        ]
        return httpx.Response(
            200, text="".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
            ),
        )

    instruction = "Return a handoff in <summary> tags, preserving all requirements."
    body = {"stream": stream, "system": "native checkpoint instructions", "messages": [
        {"role": "user", "content": instruction},
    ]}
    if large_field == "instructions":
        body["system"] = "CRITICAL_POLICY: do not publish. " + "s" * 6000
    else:
        body["tools"] = [{
            "name": "lookup", "description": "CRITICAL_POLICY: do not publish. " + "t" * 6000,
            "input_schema": {"type": "object"},
        }]
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:
        app = create_app(auth=FakeAuth(), client=backend)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/v1/messages", headers={"x-claude-code-request-class": "compaction"}, json=body,
            )
    assert response.status_code == 200
    assert "event: error" not in response.text
    assert not oversized
    assert len(leaves) > 1
    transcript = json.loads("".join(leaves))
    assert transcript["instructions"] == body["system"]
    if large_field == "tools":
        assert transcript["tools"][0]["description"] == body["tools"][0]["description"]
        assert normal_requests[-1]["instructions"] == body["system"]
    else:
        assert len(normal_requests[-1]["instructions"]) < len(body["system"])
    assert not normal_requests[-1].get("tools")
    assert normal_requests[-1]["tool_choice"] == "none"
    assert normal_requests[-1]["parallel_tool_calls"] is False
    assert normal_requests[-1]["reasoning"]["effort"] == "low"
    assert "CRITICAL_POLICY" in normal_requests[-1]["input"][0]["content"][0]["text"]
    assert normal_requests[-1]["input"][-1]["content"][0]["text"] == instruction


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("content", [
    pytest.param("x" * 1800, id="ascii"), pytest.param("я" * 900, id="unicode"),
])
async def test_local_checkpoint_budget_includes_added_prompt(monkeypatch, stream, content):
    monkeypatch.setattr("claude_codex.proxy.COMPACTION_CHUNK_TOKENS", 512)
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", "1")
    monkeypatch.delenv("CLAUDE_CODEX_REMOTE_COMPACT", raising=False)
    leaves = []
    normal = []
    oversized = []

    async def upstream(request):
        body = json.loads(request.content)
        size = sum(
            len(json.dumps(body.get(key, ""), ensure_ascii=False).encode())
            for key in ("instructions", "input", "tools")
        )
        if size > 2048:
            oversized.append(size)
            return httpx.Response(
                400, json={"error": {"code": "context_length_exceeded", "message": "too big"}}
            )
        text = body["input"][0]["content"][0]["text"]
        if text.startswith("Transcript segment "):
            leaves.append(text)
        elif not text.startswith("Checkpoint summaries"):
            normal.append(body)
        events = [
            {"type": "response.output_text.delta", "delta": "checkpoint"},
            {"type": "response.completed", "response": {"usage": {"input_tokens": 100}}},
        ]
        return httpx.Response(
            200, text="".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:
        app = create_app(auth=FakeAuth(), client=backend)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://proxy.test"
        ) as client:
            response = await client.post(
                "/v1/messages", json={
                    "system": "small", "messages": [{"role": "user", "content": content}], "stream": stream,
                },
            )
    assert response.status_code == 200
    assert not oversized
    assert leaves
    assert normal[-1]["input"][-1]["content"][0]["text"] == "Context checkpoint summary:\ncheckpoint"


@pytest.mark.parametrize("request_class", [None, "main", "subagent", "workflow", "compaction", "auxiliary"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["deepseek-v4.1-flash:cloud", "my-local-classifier:latest"])
async def test_auxiliary_endpoint_isolated_from_codex_auth(
    monkeypatch, request_class, stream, model, capsys, tmp_path
):
    monkeypatch.setenv("CLAUDE_CODEX_AUXILIARY_ENDPOINT", "http://localhost:11434/v1/responses")
    monkeypatch.setenv("CLAUDE_CODEX_AUXILIARY_MODEL", model)
    monkeypatch.setenv("CLAUDE_CODEX_REASONING", "medium")
    monkeypatch.delenv("CLAUDE_CODEX_AUXILIARY_REASONING", raising=False)
    captured = []
    auth = FakeAuth()
    auth_calls = []

    async def get(**kwargs):
        auth_calls.append(kwargs)
        return await FakeAuth().get(**kwargs)

    auth.get = get

    def upstream(request):
        captured.append(request)
        events = [
            {"type": "response.output_text.delta", "output_index": 0, "delta": "VERDICT"},
            {"type": "response.completed", "response": {"usage": {"input_tokens": 3, "output_tokens": 1}}},
        ]
        return httpx.Response(200, text="".join("data: " + json.dumps(e) + "\n\n" for e in events))

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        app = create_app(auth=auth, client=http)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            headers = {
                "x-claude-code-session-id": "private-session",
                "x-claude-code-agent-id": "private-agent",
            }
            if request_class:
                headers["x-claude-code-request-class"] = request_class
            r = await c.post("/v1/messages", headers=headers, json={
                "model": "claude-codex", "stream": stream,
                "messages": [{"role": "user", "content": "Synthetic check."}],
            })
    assert r.status_code == 200
    assert "VERDICT" in r.text
    assert len(captured) == 1
    req = captured[0]
    payload = json.loads(req.content)
    logs = [json.loads(line.removeprefix("proxy_upstream "))
            for line in capsys.readouterr().out.splitlines() if line.startswith("proxy_upstream ")]
    assert [entry["event"] for entry in logs] == ["start", "end"]
    assert logs[0]["request_id"] == logs[1]["request_id"]
    assert logs[0]["request_class"] == (request_class or "main")
    assert logs[0]["model"] == payload["model"]
    assert logs[0]["effort"] == payload["reasoning"]["effort"]
    assert logs[1]["result"] == "completed"
    assert logs[1]["duration_ms"] >= 0
    assert "VERDICT" not in json.dumps(logs)
    assert "Synthetic check." not in json.dumps(logs)
    audit_path = tmp_path / "auxiliary/requests.jsonl"
    if request_class == "auxiliary":
        records = [json.loads(line) for line in audit_path.read_text().splitlines()]
        assert records[0]["event"] == "request"
        assert records[0]["payload"] == payload
        assert records[1]["data"]["delta"] == "VERDICT"
        assert records[-1]["result"] == "completed"
        assert all(r["request_id"] == logs[0]["request_id"] for r in records)
        assert audit_path.stat().st_mode & 0o777 == 0o600
        assert audit_path.parent.stat().st_mode & 0o777 == 0o700
        assert "client_metadata" not in records[0]["payload"]
        assert "prompt_cache_key" not in records[0]["payload"]
    else:
        assert not audit_path.exists()
    assert logs[0]["backend"] == ("auxiliary" if request_class == "auxiliary" else "codex")
    if request_class == "auxiliary":
        assert logs[0]["endpoint_origin"] == "http://localhost:11434"

        assert str(req.url) == "http://localhost:11434/v1/responses"
        assert not auth_calls
        assert "authorization" not in req.headers
        assert "chatgpt-account-id" not in req.headers
        assert not any(k.startswith("x-claude-code-") or k.startswith("x-codex-") for k in req.headers)
        assert "client_metadata" not in payload
        assert "prompt_cache_key" not in payload
        assert "include" not in payload
        assert payload["model"] == model
        assert payload["reasoning"] == {"effort": "low"}
    else:
        assert req.url.host == "chatgpt.com"
        assert auth_calls
        assert req.headers["authorization"] == "Bearer access"


@pytest.mark.parametrize("status, text", [
    (404, '{"error":"model not found"}'),
    (200, 'data: {"type":"response.created"}\n\n'),
    (200, 'data: not-json\n\n'),
])
@pytest.mark.parametrize("stream", [False, True])
async def test_auxiliary_endpoint_errors_do_not_fallback_or_allow(monkeypatch, status, text, stream, capsys):
    monkeypatch.setenv("CLAUDE_CODEX_AUXILIARY_ENDPOINT", "http://localhost:11434/v1/responses")
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(status, text=text)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        app = create_app(auth=FakeAuth(), client=http)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://proxy"
        ) as c:
            r = await c.post(
                "/v1/messages", headers={"x-claude-code-request-class": "auxiliary"},
                json={"model": "claude-codex", "stream": stream,
                      "messages": [{"role": "user", "content": "Check."}]},
            )
    if stream and status == 200:
        assert "event: error" in r.text
    else:
        assert r.status_code >= 400
    assert len(calls) == 1
    assert calls[0].url.host == "localhost"
    logs = [json.loads(line.removeprefix("proxy_upstream "))
            for line in capsys.readouterr().out.splitlines() if line.startswith("proxy_upstream ")]
    assert logs[-1]["event"] == "end"
    assert logs[-1]["result"] == "error"
    assert "error_type" in logs[-1]
    if status != 200:
        assert logs[-1]["http_status"] == status
    assert "model not found" not in json.dumps(logs)


async def test_upstream_log_redacts_endpoint_credentials_and_query(monkeypatch, capsys):
    monkeypatch.setenv(
        "CLAUDE_CODEX_AUXILIARY_ENDPOINT",
        "http://private-user:private-password@localhost:11434/v1/responses?token=private-query",
    )

    def upstream(request):
        return httpx.Response(503, text="private-error-body")

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        app = create_app(auth=FakeAuth(), client=http)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            await c.post(
                "/v1/messages", headers={"x-claude-code-request-class": "auxiliary"},
                json={"model": "claude-codex", "messages": []},
            )
    output = capsys.readouterr().out
    for secret in (
        "private-user", "private-password", "private-query", "private-error-body", "Bearer access",
    ):
        assert secret not in output
    logs = [json.loads(line.removeprefix("proxy_upstream "))
            for line in output.splitlines() if line.startswith("proxy_upstream ")]
    assert logs[0]["endpoint_origin"] == "http://localhost:11434"
    assert logs[-1]["http_status"] == 503

async def test_large_checkpoint_resumes_completed_segments_after_restart(monkeypatch, tmp_path):
    monkeypatch.setattr("claude_codex.proxy.COMPACTION_CHUNK_TOKENS", 512)
    monkeypatch.setattr("claude_codex.proxy.COMPACTION_CONCURRENCY", 1)
    monkeypatch.setattr("claude_codex.proxy.COMPACTION_TIMEOUT_SECONDS", 0.04)
    monkeypatch.setattr("claude_codex.proxy.CHECKPOINT_CACHE_PATH", tmp_path / "checkpoints", raising=False)
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", "1")
    phase = 1
    seen = [[], []]

    async def upstream(request):
        body = json.loads(request.content)
        text = body["input"][0]["content"][0]["text"]
        if text.startswith("Transcript segment "):
            index = int(text.splitlines()[0].split()[-1])
            seen[phase - 1].append(index)
            if phase == 1 and index > 1:
                await asyncio.sleep(10)
            summary = f"summary-{index}"
        elif text.startswith("Checkpoint summaries"):
            summary = "restored checkpoint"
        else:
            summary = "normal answer"
        events = [
            {"type": "response.output_text.delta", "output_index": 0, "delta": summary},
            {"type": "response.completed", "response": {"usage": {"input_tokens": 10, "output_tokens": 2}}},
        ]
        return httpx.Response(200, text="".join("data: " + json.dumps(e) + "\n\n" for e in events))

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        async def send():
            app = create_app(auth=FakeAuth(), client=http)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://proxy"
            ) as c:
                return await c.post(
                    "/v1/messages", headers={"x-claude-code-session-id": "resume-checkpoint"},
                    json={"model": "claude-codex", "messages": [
                        {"role": "user", "content": "x" * 5000},
                    ] + ([{"role": "user", "content": "continue"}] if phase == 2 else [])},
                )

        first = await send()
        assert first.status_code == 400
        assert first.json()["error"]["code"] == "compaction_timeout"
        phase = 2
        second = await send()
    assert second.status_code == 200
    assert seen[0][0] == 1
    assert seen[1]
    assert 1 not in seen[1], "Completed segment was lost across proxy restart"

@pytest.mark.parametrize("override, expected", [(None, "gpt-6-luna"), ("gpt-6.1-sol", "gpt-6.1-sol")])
async def test_checkpoint_model_override_leaves_main_model_unchanged(monkeypatch, override, expected):
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", "1")
    monkeypatch.setenv("CLAUDE_CODEX_MODEL", "gpt-6.1-sol")
    if override is None:
        monkeypatch.delenv("CLAUDE_CODEX_COMPACTION_MODEL", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_CODEX_COMPACTION_MODEL", override)
    models = []

    def upstream(request):
        payload = json.loads(request.content)
        models.append(payload["model"])
        events = [
            {"type": "response.output_text.delta", "output_index": 0, "delta": "checkpoint"},
            {"type": "response.completed"},
        ]
        return httpx.Response(200, text="".join("data: " + json.dumps(e) + "\n\n" for e in events))

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        app = create_app(auth=FakeAuth(), client=http)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
            response = await c.post(
                "/v1/messages",
                json={"model": "claude-codex", "messages": [{"role": "user", "content": "task"}]},
            )
    assert response.status_code == 200
    assert models == [expected, "gpt-6.1-sol"]
