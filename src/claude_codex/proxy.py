from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import AuthError, AuthManager, AuthProvider
from .translate import (
    AnthropicStream,
    encode_sse,
    estimate_tokens,
    to_responses_request,
    validate_messages_request,
)

CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
INSTALLATION_ID_PATH = Path.home() / ".config" / "claude-codex" / "installation_id"

# Emit an SSE `ping` at least this often so a client never sees a silent stream
# (e.g. during long model reasoning) and time the connection out.
PING_INTERVAL_SECONDS = 15.0


def _resolve_installation_id(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            return str(uuid.UUID(path.read_text().strip()))
        except (OSError, ValueError):
            installation_id = str(uuid.uuid4())
            path.write_text(installation_id + "\n")
            path.chmod(0o600)
            return installation_id
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class CodexRequestIdentity:
    """Stable upstream identity for one Claude Code session.

    The ChatGPT Codex backend uses all four IDs for request routing. In
    particular, keeping them stable across turns lets it route repeated prompt
    prefixes to the cache written by the first turn.
    """

    installation_id: str
    session_id: str
    thread_id: str
    window_id: str

    def client_metadata(self) -> dict[str, str]:
        return {
            "x-codex-installation-id": self.installation_id,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "x-codex-window-id": self.window_id,
        }


class BackendError(RuntimeError):
    """A non-success HTTP response from the Codex backend."""

    def __init__(self, status_code: int, body: str, retry_after: str | None = None) -> None:
        super().__init__(f"Codex backend HTTP {status_code}: {body}")
        self.status_code = status_code
        self.retry_after = retry_after


def _error_type(status_code: int) -> str:
    # Preserve the upstream rate-limit / overload semantics so the client backs
    # off instead of treating a 429 as a generic bad-gateway and retry-storming.
    if status_code == 429:
        return "rate_limit_error"
    if status_code == 529:
        return "overloaded_error"
    return "api_error"


def _stream_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, AuthError):
        kind = "authentication_error"
    elif isinstance(exc, BackendError):
        kind = _error_type(exc.status_code)
    else:
        kind = "api_error"
    return {"type": "error", "error": {"type": kind, "message": str(exc)}}


def _log_upstream_cache_usage(event: str, data: dict[str, Any]) -> None:
    """Write cache usage reported by the upstream Responses event to proxy.log."""
    if event not in {"response.completed", "response.incomplete"}:
        return
    response = data.get("response")
    if not isinstance(response, dict):
        return
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return
    details = usage.get("input_tokens_details")
    if not isinstance(details, dict):
        details = {}
    try:
        input_tokens = max(0, int(usage.get("input_tokens") or 0))
        cached_tokens = max(0, int(details.get("cached_tokens") or 0))
    except (TypeError, ValueError):
        return
    cache_write = details.get("cache_write_tokens")
    if cache_write is None:
        cache_write_tokens = "unreported"
    else:
        try:
            cache_write_tokens = str(max(0, int(cache_write)))
        except (TypeError, ValueError):
            cache_write_tokens = "unreported"
    result = "hit" if cached_tokens else "miss"
    print(
        "codex_cache "
        f"source=upstream result={result} input_tokens={input_tokens} "
        f"cached_tokens={cached_tokens} cache_write_tokens={cache_write_tokens}",
        flush=True,
    )


async def _sse(response: httpx.Response) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event = "message"
    data: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                raw = "\n".join(data)
                if raw != "[DONE]":
                    payload = json.loads(raw)
                    yield event, payload
                event, data = "message", []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if data:
        raw = "\n".join(data)
        if raw != "[DONE]":
            yield event, json.loads(raw)


class CodexBackend:
    def __init__(self, auth: AuthProvider, client: httpx.AsyncClient, endpoint: str = CODEX_ENDPOINT) -> None:
        self.auth = auth
        self.client = client
        self.endpoint = endpoint

    async def events(
        self, payload: dict[str, Any], identity: CodexRequestIdentity
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        stale_access: str | None = None
        for attempt in range(2):
            tokens = await self.auth.get(force_refresh=attempt == 1, stale_access=stale_access)
            headers = {
                "Authorization": f"Bearer {tokens.access}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                # Match the first-party Codex client contract. The subscription
                # endpoint uses this identity, not just prompt_cache_key, for
                # stable cache routing between Claude Code turns.
                "originator": "codex_cli_rs",
                "User-Agent": "Codex/0.1.0",
                "x-codex-installation-id": identity.installation_id,
                "session-id": identity.session_id,
                "thread-id": identity.thread_id,
                "x-client-request-id": identity.thread_id,
                "x-codex-window-id": identity.window_id,
            }
            if tokens.account_id:
                headers["ChatGPT-Account-Id"] = tokens.account_id
            async with self.client.stream("POST", self.endpoint, headers=headers, json=payload) as response:
                if response.status_code == 401 and attempt == 0:
                    stale_access = tokens.access
                    continue
                if response.is_error:
                    body = (await response.aread()).decode(errors="replace")
                    raise BackendError(response.status_code, body[:500], response.headers.get("retry-after"))
                async for event, data in _sse(response):
                    _log_upstream_cache_usage(event, data)
                    yield event, data
                return


def create_app(
    *,
    auth: AuthProvider | None = None,
    client: httpx.AsyncClient | None = None,
    endpoint: str | None = None,
    startup_id: str | None = None,
    installation_id_path: Path | None = None,
) -> FastAPI:
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30))
    manager = auth or AuthManager(client=http)
    backend = CodexBackend(
        manager,
        http,
        endpoint or os.environ.get("CLAUDE_CODEX_ENDPOINT", CODEX_ENDPOINT),
    )
    installation_id = _resolve_installation_id(installation_id_path or INSTALLATION_ID_PATH)
    session_identities: dict[str, CodexRequestIdentity] = {}

    def identity_for(session_id: str) -> CodexRequestIdentity:
        identity = session_identities.get(session_id)
        if identity is None:
            identity = CodexRequestIdentity(
                installation_id=installation_id,
                session_id=session_id,
                thread_id=str(uuid.uuid4()),
                window_id=str(uuid.uuid4()),
            )
            session_identities[session_id] = identity
        return identity

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if owns_client:
            await http.aclose()

    app = FastAPI(title="claude-codex", lifespan=lifespan)

    @app.exception_handler(AuthError)
    async def auth_error(_: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"type": "error", "error": {"type": "authentication_error", "message": str(exc)}},
        )

    @app.exception_handler(BackendError)
    async def backend_http_error(_: Request, exc: BackendError) -> JSONResponse:
        status = exc.status_code if exc.status_code in {429, 529} else 502
        headers = {"Retry-After": exc.retry_after} if exc.retry_after else None
        return JSONResponse(
            status_code=status,
            headers=headers,
            content={"type": "error", "error": {"type": _error_type(exc.status_code), "message": str(exc)}},
        )

    @app.exception_handler(RuntimeError)
    async def backend_error(_: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "api_error", "message": str(exc)}},
        )

    def invalid_request(message: str) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": message}},
        )

    async def request_body(
        request: Request, *, require_messages: bool = True
    ) -> dict[str, Any] | JSONResponse:
        try:
            body = await request.json()
            return validate_messages_request(body, require_messages=require_messages)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return invalid_request(str(exc) or "Invalid request body")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        try:
            source = manager.load().source
        except AuthError:
            source = None
        return {
            "status": "ok",
            "auth": bool(source),
            "auth_source": source,
            "startup_id": startup_id,
        }

    @app.post("/v1/messages/count_tokens", response_model=None)
    async def count_tokens(request: Request) -> dict[str, int] | JSONResponse:
        body = await request_body(request)
        if isinstance(body, JSONResponse):
            return body
        return {"input_tokens": estimate_tokens(body)}

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request_body(request)
        if isinstance(body, JSONResponse):
            return body
        requested_model = str(body.get("model") or "claude-codex")
        codex_model = os.environ.get("CLAUDE_CODEX_MODEL", "gpt-5.6-sol")
        reasoning = os.environ.get("CLAUDE_CODEX_REASONING", "medium")
        session_id = (
            request.headers.get("x-session-id")
            or request.headers.get("session-id")
            or request.headers.get("anthropic-session-id", "claude-codex")
        )
        identity = identity_for(session_id)
        upstream = to_responses_request(
            body,
            model=codex_model,
            reasoning_effort=reasoning,
            prompt_cache_key=identity.session_id,
        )
        upstream["client_metadata"] = identity.client_metadata()

        if body.get("stream", False):
            input_tokens = estimate_tokens(body)

            async def stream() -> AsyncIterator[str]:
                translator = AnthropicStream(
                    requested_model,
                    retain_content=False,
                    input_tokens=input_tokens,
                )
                # Consume the upstream in a background task so the response can
                # keep emitting SSE pings during long silent gaps (e.g. model
                # reasoning) instead of letting the client time the stream out.
                queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=256)

                async def pump() -> None:
                    try:
                        async for event, data in backend.events(upstream, identity):
                            await queue.put(("event", (event, data)))
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # forwarded to the consumer as an SSE error
                        await queue.put(("error", exc))
                    else:
                        await queue.put(("done", None))

                task = asyncio.create_task(pump())
                error: Exception | None = None
                completed = False
                cancelled = False
                try:
                    while True:
                        try:
                            kind, payload = await asyncio.wait_for(queue.get(), timeout=PING_INTERVAL_SECONDS)
                        except TimeoutError:
                            yield encode_sse("ping", {"type": "ping"})
                            continue
                        if kind == "error":
                            error = payload
                            break
                        if kind == "done":
                            completed = True
                            break
                        event, data = payload
                        try:
                            translated = translator.feed(event, data)
                        except Exception as exc:  # never drop the stream on a translation fault
                            error = exc
                            break
                        for name, chunk in translated:
                            yield encode_sse(name, chunk)
                except asyncio.CancelledError:
                    # Launcher shutdown и disconnect клиента отменяют response task.
                    # Upstream task очищается ниже; штатное завершение не должно
                    # превращаться в ASGI traceback.
                    cancelled = True
                finally:
                    task.cancel()
                    with suppress(BaseException):
                        await task
                if cancelled:
                    return
                if error is not None:
                    yield encode_sse("error", _stream_error(error))
                elif completed:
                    for name, chunk in translator.finish():
                        yield encode_sse(name, chunk)

            return StreamingResponse(stream(), media_type="text/event-stream")

        translator = AnthropicStream(requested_model)
        async for event, data in backend.events(upstream, identity):
            translator.feed(event, data)
        translator.finish()
        return translator.response()

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic-compatible Codex subscription proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CLAUDE_CODEX_PORT", "8111")))
    parser.add_argument("--fd", type=int)
    parser.add_argument("--startup-id")
    args = parser.parse_args()
    print(f"proxy_started startup_id={args.startup_id or 'unknown'}", flush=True)
    uvicorn.run(
        create_app(startup_id=args.startup_id),
        host=args.host,
        port=args.port,
        fd=args.fd,
        log_level="warning",
        # Don't let an in-flight stream block shutdown when the launcher stops us.
        timeout_graceful_shutdown=2,
    )


if __name__ == "__main__":
    main()
