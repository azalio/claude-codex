from __future__ import annotations

import asyncio
import json
import time

import httpx

from claude_codex.auth import Tokens
from claude_codex.proxy import create_app


class FakeAuth:
    async def get(self, *, force_refresh: bool = False, stale_access: str | None = None) -> Tokens:
        del force_refresh, stale_access
        return Tokens("access", "refresh", int(time.time() * 1000) + 60_000, "acc-123", "test")

    def load(self) -> Tokens:
        return Tokens("access", "refresh", int(time.time() * 1000) + 60_000, "acc-123", "test")


async def test_proxy_streams_anthropic_events(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_CODEX_MODEL", raising=False)
    captured: dict = {}

    async def upstream(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["account"] = request.headers.get("ChatGPT-Account-Id")
        captured["session"] = request.headers.get("session-id")
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
    assert captured["body"]["model"] == "gpt-5.6-sol"
    assert captured["body"]["prompt_cache_key"] == "session-test"
    assert "max_output_tokens" not in captured["body"]
    assert captured["body"]["input"][0]["content"][0]["text"] == "hi"
    await upstream_client.aclose()


async def test_nonstream_surfaces_cached_input_usage() -> None:
    async def upstream(_: httpx.Request) -> httpx.Response:
        events = [
            {"type": "response.created", "response": {"id": "resp_test"}},
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 4096,
                        "input_tokens_details": {"cached_tokens": 3072},
                        "output_tokens": 2,
                    }
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
        "input_tokens": 1024,
        "cache_read_input_tokens": 3072,
        "output_tokens": 2,
    }
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
