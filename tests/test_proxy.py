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
from claude_codex.proxy import _compact_at_tokens, _remote_compact_enabled, create_app


@pytest.fixture(autouse=True)
def isolated_installation_id(monkeypatch, tmp_path: Path) -> Path:
    path = tmp_path / "installation_id"
    monkeypatch.setattr("claude_codex.proxy.INSTALLATION_ID_PATH", path)
    return path


class FakeAuth:
    async def get(self, *, force_refresh: bool = False, stale_access: str | None = None) -> Tokens:
        del force_refresh, stale_access
        return Tokens("access", "refresh", int(time.time() * 1000) + 60_000, "acc-123", "test")

    def load(self) -> Tokens:
        return Tokens("access", "refresh", int(time.time() * 1000) + 60_000, "acc-123", "test")


@pytest.mark.parametrize("value", [None, "invalid"])
def test_compact_at_tokens_defaults_to_900k(monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv("CLAUDE_CODEX_COMPACT_AT", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", value)

    assert _compact_at_tokens() == 900_000


@pytest.mark.parametrize("value, expected", [("0", 0), ("-1", 0), ("123456", 123_456)])
def test_compact_at_tokens_preserves_overrides(monkeypatch, value, expected) -> None:
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", value)

    assert _compact_at_tokens() == expected


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
    assert captured["body"]["reasoning"] == {"effort": "xhigh", "summary": "auto"}
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
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", "100")
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
        input_tokens = 100 if len(normal_inputs) == 1 else 10
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
    assert "result=started input_tokens=100 threshold=100" in log
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
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", "100")
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
        input_tokens = 100 if initial_user_text in {"a", "b"} else 10
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
    assert capsys.readouterr().out == (
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
    assert capsys.readouterr().out == (
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
    assert response.json()["error"]["message"] == message
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
    "configured, effort, expected", [(None, "high", "high"), (None, "max", "xhigh"), ("low", "high", "low")]
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
            response = await client.post(
                "/v1/messages", json={"messages": [], "output_config": {"effort": effort}}
            )
    assert response.status_code == 200
    assert captured[0]["reasoning"]["effort"] == expected


async def test_startup_probe_and_discovery_report_only_the_configured_backend(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CODEX_MODEL", "gpt-test-backend")
    monkeypatch.setenv("ANTHROPIC_MODEL", "anthropic-codex-alias")

    def upstream(_: httpx.Request) -> httpx.Response:
        pytest.fail("Startup endpoints must not call inference")

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
