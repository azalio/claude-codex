from __future__ import annotations

import argparse
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import AuthError, AuthManager
from .translate import AnthropicStream, encode_sse, estimate_tokens, to_responses_request

CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"


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
    def __init__(self, auth: AuthManager, client: httpx.AsyncClient, endpoint: str = CODEX_ENDPOINT) -> None:
        self.auth = auth
        self.client = client
        self.endpoint = endpoint

    async def events(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        for attempt in range(2):
            tokens = await self.auth.get(force_refresh=attempt == 1)
            headers = {
                "Authorization": f"Bearer {tokens.access}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "originator": "claude-codex",
                "User-Agent": "claude-codex/0.1.0",
                "session-id": session_id,
            }
            if tokens.account_id:
                headers["ChatGPT-Account-Id"] = tokens.account_id
            async with self.client.stream("POST", self.endpoint, headers=headers, json=payload) as response:
                if response.status_code == 401 and attempt == 0:
                    continue
                if response.is_error:
                    body = (await response.aread()).decode(errors="replace")
                    raise RuntimeError(f"Codex backend HTTP {response.status_code}: {body[:500]}")
                async for event in _sse(response):
                    yield event
                return


def create_app(
    *,
    auth: AuthManager | None = None,
    client: httpx.AsyncClient | None = None,
    endpoint: str | None = None,
) -> FastAPI:
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30))
    manager = auth or AuthManager()
    backend = CodexBackend(manager, http, endpoint or os.environ.get("CLAUDE_CODEX_ENDPOINT", CODEX_ENDPOINT))

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

    @app.exception_handler(RuntimeError)
    async def backend_error(_: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "api_error", "message": str(exc)}},
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        try:
            source = manager.load().source
        except AuthError:
            source = None
        return {"status": "ok", "auth": bool(source), "auth_source": source}

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> dict[str, int]:
        return {"input_tokens": estimate_tokens(await request.json())}

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        requested_model = str(body.get("model") or "claude-codex")
        codex_model = os.environ.get("CLAUDE_CODEX_MODEL", "gpt-5.6-sol")
        reasoning = os.environ.get("CLAUDE_CODEX_REASONING", "medium")
        upstream = to_responses_request(body, model=codex_model, reasoning_effort=reasoning)
        session_id = (
            request.headers.get("x-session-id")
            or request.headers.get("session-id")
            or request.headers.get("anthropic-session-id", "claude-codex")
        )

        if body.get("stream", False):

            async def stream() -> AsyncIterator[str]:
                translator = AnthropicStream(requested_model)
                async for event, data in backend.events(upstream, session_id):
                    for name, translated in translator.feed(event, data):
                        yield encode_sse(name, translated)
                for name, translated in translator.finish():
                    yield encode_sse(name, translated)

            return StreamingResponse(stream(), media_type="text/event-stream")

        translator = AnthropicStream(requested_model)
        async for event, data in backend.events(upstream, session_id):
            translator.feed(event, data)
        translator.finish()
        return translator.response()

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic-compatible Codex subscription proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CLAUDE_CODEX_PORT", "8111")))
    args = parser.parse_args()
    uvicorn.run("claude_codex.proxy:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
