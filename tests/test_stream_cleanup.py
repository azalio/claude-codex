"""Exercise response ownership with real TCP streams and a bounded HTTPX pool."""
from __future__ import annotations

import asyncio
import json
import socket
import time
from contextlib import suppress

import httpx
import pytest
import uvicorn
from starlette.requests import ClientDisconnect, Request

from claude_codex.auth import Tokens
from claude_codex.proxy import BackendError, CodexBackend, CodexRequestIdentity, create_app
from claude_codex.translate import CodexResponseError


class FakeAuth:
    async def get(self, **kwargs) -> Tokens:
        return Tokens("access", "refresh", int(time.time() * 1000) + 60_000, "test", "test")


@pytest.fixture
async def upstream():
    tasks = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                          if line.lower().startswith(b"content-length:"))
            body = await reader.readexactly(length)
            mode = next((mode for mode in ("hold", "eof", "invalid", "failed", "incomplete")
                         if mode.encode() in body), "complete")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Transfer-Encoding: chunked\r\n\r\n")
            events = [{"type": "response.created", "response": {"id": "test"}}]
            if mode == "invalid":
                events += [
                    {"type": "response.output_text.delta", "output_index": 0, "delta": "ok"},
                    {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{}"},
                ]
            elif mode in {"failed", "incomplete"}:
                events.append({
                    "type": f"response.{mode}", "response": {"error": {"message": "test failure"}},
                })
            elif mode == "complete":
                events += [{"type": "response.output_text.delta", "output_index": 0, "delta": "ok"},
                           {"type": "response.completed", "response": {"usage": {}}}]
            for event in events:
                data = f"data: {json.dumps(event)}\n\n".encode()
                writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            await writer.drain()
            if mode != "eof":
                # Even a terminal SSE event must close a response whose HTTP body stays open.
                await reader.read()
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/responses"
    finally:
        server.close()
        await server.wait_closed()
        pending = list(tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.fixture
async def proxy(monkeypatch, tmp_path, upstream):
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", "0")
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1), timeout=httpx.Timeout(1, pool=0.05), trust_env=False,
    ) as backend:
        app = create_app(auth=FakeAuth(), client=backend, endpoint=upstream,
                         installation_id_path=tmp_path / "installation_id")
        yield app, backend


def request(mode="hold", *, stream=True, spec="2.4"):
    scope = {
        "type": "http", "asgi": {"spec_version": spec}, "method": "POST",
        "path": "/v1/messages", "headers": [], "query_string": b"",
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps({
            "model": "claude-opus", "stream": stream,
            "messages": [{"role": "user", "content": mode}],
        }).encode()}

    return Request(scope, receive)


async def response_for(app, req):
    endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/messages")
    return await endpoint(req)


async def assert_pool_recovers(app, backend):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test",
    ) as client:
        probe = await client.post("/v1/messages", json={
            "model": "claude-opus", "messages": [{"role": "user", "content": "probe"}],
        })
    assert probe.status_code == 200, probe.text
    assert probe.json()["content"][0]["text"] == "ok"
    # Check live ownership, not just socket counts or eventual garbage collection.
    assert not backend._transport._pool._requests
    assert not backend._transport._pool.connections


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
@pytest.mark.parametrize("failure", ["disconnect", "cancel"])
@pytest.mark.parametrize("fail_at", ["http.response.start", "http.response.body"])
async def test_send_failure_releases_upstream_pool(proxy, spec, failure, fail_at):
    app, backend = proxy
    req = request(spec=spec)
    response = await response_for(app, req)

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == fail_at:
            if failure == "disconnect":
                raise OSError("downstream disconnected")
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

    try:
        # ASGI 2.3 uses a cancelling task group; 2.4 propagates send failures directly.
        with suppress(OSError, ClientDisconnect, asyncio.CancelledError):
            await asyncio.create_task(response(req.scope, receive, send))
        await assert_pool_recovers(app, backend)
    finally:
        await response.body_iterator.aclose()
        if response.background is not None:
            await response.background()


@pytest.mark.parametrize("phase", ["headers", "body", "backpressure"])
async def test_receive_disconnect_releases_upstream_pool(proxy, phase):
    app, backend = proxy
    req = request(spec="2.3")
    response = await response_for(app, req)
    ready = asyncio.Event()

    async def receive():
        await ready.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        trigger = "http.response.start" if phase == "headers" else "http.response.body"
        if message["type"] == trigger:
            ready.set()
            if phase == "backpressure":
                await asyncio.Event().wait()

    try:
        await asyncio.wait_for(response(req.scope, receive, send), timeout=1)
        await assert_pool_recovers(app, backend)
    finally:
        await response.body_iterator.aclose()
        if response.background is not None:
            await response.background()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["complete", "eof", "invalid", "hold"])
async def test_upstream_completion_and_errors_release_pool(proxy, stream, mode):
    app, backend = proxy
    if mode == "hold":
        backend.timeout = httpx.Timeout(0.05)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test",
    ) as client:
        result = await client.post("/v1/messages", json={
            "model": "claude-opus", "stream": stream,
            "messages": [{"role": "user", "content": mode}],
        })
    if mode == "complete":
        assert result.status_code == 200
    elif stream:
        assert "event: error" in result.text
    else:
        assert result.status_code == 502
    await assert_pool_recovers(app, backend)


async def test_nonstream_translation_error_closes_source_before_unwinding(proxy):
    app, backend = proxy
    # Keep the traceback alive: a response must be closed before a GC/finalizer pass.
    with pytest.raises(RuntimeError, match="incompatible content block") as error:
        await response_for(app, request("invalid", stream=False))
    assert error.value
    assert not backend._transport._pool._requests
    await assert_pool_recovers(app, backend)


@pytest.mark.parametrize("mode", ["failed", "incomplete"])
async def test_checkpoint_error_closes_source_before_unwinding(proxy, upstream, mode):
    app, backend = proxy
    codex = CodexBackend(FakeAuth(), backend, upstream)
    identity = CodexRequestIdentity("install", "session", "test", "thread", "window")
    with pytest.raises((CodexResponseError, BackendError)) as error:
        await codex.summarize({"input": mode}, identity)
    assert error.value
    assert not backend._transport._pool._requests
    await assert_pool_recovers(app, backend)


async def test_parallel_disconnect_batches_leave_shared_client_usable(monkeypatch, tmp_path, upstream):
    monkeypatch.setenv("CLAUDE_CODEX_COMPACT_AT", "0")
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=3), timeout=httpx.Timeout(1, pool=0.05), trust_env=False,
    ) as backend:
        app = create_app(auth=FakeAuth(), client=backend, endpoint=upstream,
                         installation_id_path=tmp_path / "installation_id")
        for _ in range(5):
            requests = [request() for _ in range(3)]
            responses = await asyncio.gather(*(response_for(app, req) for req in requests))
            ready = [asyncio.Event() for _ in responses]
            disconnect = [asyncio.Event() for _ in responses]

            async def run(response, req, ready_event, disconnect_event):
                async def send(message):
                    if message["type"] == "http.response.body":
                        ready_event.set()
                        await disconnect_event.wait()
                        raise OSError("downstream disconnected")

                with suppress(ClientDisconnect):
                    await response(req.scope, req.receive, send)

            tasks = [asyncio.create_task(run(*args))
                     for args in zip(responses, requests, ready, disconnect, strict=True)]
            try:
                await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), timeout=1)
                assert len(backend._transport._pool._requests) == 3
                disconnect[0].set()
                await tasks[0]
                # Closing one request must not reset the shared pool or cancel its siblings.
                assert len(backend._transport._pool._requests) == 2
                assert not tasks[1].done() and not tasks[2].done()
                for event in disconnect[1:]:
                    event.set()
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
                await assert_pool_recovers(app, backend)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for response in responses:
                    await response.body_iterator.aclose()
                    if response.background is not None:
                        await response.background()


async def test_live_proxy_recovers_after_repeated_tcp_disconnects(proxy):
    app, backend = proxy
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, lifespan="off", access_log=False, log_config=None))
        serving = asyncio.create_task(server.serve(sockets=[listener]))

        async def wait_until_started():
            while not server.started:
                if serving.done():
                    await serving
                    raise AssertionError("Proxy exited before startup")
                await asyncio.sleep(0.005)

        async def wait_until_released():
            while backend._transport._pool._requests:
                await asyncio.sleep(0.005)

        try:
            await asyncio.wait_for(wait_until_started(), timeout=2)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
                for _ in range(10):
                    async with client.stream("POST", "/v1/messages", json={
                        "model": "claude-opus", "stream": True,
                        "messages": [{"role": "user", "content": "hold"}],
                    }) as response:
                        assert response.status_code == 200
                        first = await anext(response.aiter_lines())
                        assert first == "event: message_start"
                    await asyncio.wait_for(wait_until_released(), timeout=1)
                    probe = await client.post("/v1/messages", json={
                        "model": "claude-opus", "messages": [{"role": "user", "content": "probe"}],
                    })
                    assert probe.status_code == 200, probe.text
                    assert probe.json()["content"][0]["text"] == "ok"
                    assert not backend._transport._pool._requests
                    assert not backend._transport._pool.connections
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, timeout=2)
