from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from claude_codex.auth import Tokens
from claude_codex.proxy import _remote_compact_enabled, create_app


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


def test_remote_compact_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_CODEX_REMOTE_COMPACT", raising=False)
    assert not _remote_compact_enabled()
    monkeypatch.setenv("CLAUDE_CODEX_REMOTE_COMPACT", "true")
    assert _remote_compact_enabled()


async def test_proxy_streams_anthropic_events(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_CODEX_MODEL", raising=False)
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
    assert captured["body"]["model"] == "gpt-5.6-sol"
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
                headers={"x-session-id": "launcher-session", "anthropic-session-id": native_session},
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

    captured: list[dict[str, object]] = []
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
    assert compact_inputs[0]["model"] == "gpt-5.6-sol"
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
            events = [
                {"type": "response.output_text.delta", "output_index": 0, "delta": "checkpoint"},
                {"type": "response.completed", "response": {"usage": {"input_tokens": 100}}},
            ]
        else:
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
                json={"model": "claude-opus", "max_tokens": 10, "stream": True, "messages": messages},
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


async def test_streaming_backend_error_is_sse_error() -> None:
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

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "event: message_stop" not in response.text
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
            text=f"event: response.completed\\ndata: {json.dumps(event)}\\n\\n",
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

    assert response.status_code == 200
    assert '"type":"rate_limit_error"' in response.text
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

    async def upstream(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.1)  # long silent gap, as during model reasoning
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
