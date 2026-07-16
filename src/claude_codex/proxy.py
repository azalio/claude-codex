from __future__ import annotations

import argparse
import asyncio
import copy
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
DEFAULT_COMPACT_AT_TOKENS = 200_000
LOCAL_COMPACTION_RETAINED_USER_TOKENS = 20_000
MAX_COMPACTION_BRANCHES_PER_SESSION = 8
LOCAL_COMPACTION_PROMPT = """You are performing a context checkpoint compaction.
Return only a concise handoff summary for another coding agent. Preserve the current task,
user requirements, decisions, files changed, tool results, unresolved errors, and next steps.
Do not call tools and do not answer the user directly."""
LOCAL_SUMMARY_PREFIX = "Context checkpoint summary:"

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
    session_source: str
    thread_id: str
    window_id: str

    def client_metadata(self) -> dict[str, str]:
        return {
            "x-codex-installation-id": self.installation_id,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "x-codex-window-id": self.window_id,
        }


@dataclass(slots=True)
class CompactionState:
    """Локальное соответствие полной Claude-истории и compact-истории Codex."""

    branch_id: str
    identity: CodexRequestIdentity
    last_raw_input: list[dict[str, Any]]
    last_input_tokens: int = 0
    original_prefix: list[dict[str, Any]] | None = None
    replacement_history: list[dict[str, Any]] | None = None
    last_used: int = 0


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


def _log_upstream_cache_usage(event: str, data: dict[str, Any], identity: CodexRequestIdentity) -> None:
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
        f"source=upstream client_id={identity.installation_id} session_source={identity.session_source} "
        f"session_id={identity.session_id} result={result} input_tokens={input_tokens} "
        f"cached_tokens={cached_tokens} cache_write_tokens={cache_write_tokens}",
        flush=True,
    )


def _reported_input_tokens(event: str, data: dict[str, Any]) -> int | None:
    """Возвращает фактический размер контекста, подтверждённый upstream."""
    if event not in {"response.completed", "response.incomplete"}:
        return None
    response = data.get("response")
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        return max(0, int(usage.get("input_tokens") or 0))
    except (TypeError, ValueError):
        return None


def _compact_at_tokens() -> int:
    """Читает порог compact; ноль отключает автоматическое сжатие."""
    raw = os.environ.get("CLAUDE_CODEX_COMPACT_AT")
    if raw is None:
        return DEFAULT_COMPACT_AT_TOKENS
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_COMPACT_AT_TOKENS


def _remote_compact_enabled() -> bool:
    """Возвращает явное согласие использовать нестабильный remote compact API."""
    return os.environ.get("CLAUDE_CODEX_REMOTE_COMPACT", "").strip().lower() in {"1", "true", "yes"}


def _suffix_after_prefix(
    full_input: list[dict[str, Any]], prefix: list[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    if len(full_input) < len(prefix) or full_input[: len(prefix)] != prefix:
        return None
    return full_input[len(prefix) :]


def _compact_payload(upstream: dict[str, Any]) -> dict[str, Any]:
    """Строит документ для `/responses/compact` из обычного Responses-запроса."""
    fields = (
        "model",
        "input",
        "instructions",
        "tools",
        "parallel_tool_calls",
        "reasoning",
        "service_tier",
        "prompt_cache_key",
        "text",
    )
    return {field: copy.deepcopy(upstream[field]) for field in fields if field in upstream}


def _sanitize_remote_replacement_history(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Убирает из remote compact элементы, несовместимые с обычным `/responses`.

    Codex заново добавляет developer/system контекст из текущей сессии. Передавать
    его из replacement history нельзя: ChatGPT backend отвергает system message в
    `input` с HTTP 400.
    """
    return [copy.deepcopy(item) for item in items if item.get("role") not in {"system", "developer"}]


def _local_compact_payload(upstream: dict[str, Any]) -> dict[str, Any]:
    """Формирует fallback-компакт обычным, уже поддерживаемым `/responses`."""
    payload = copy.deepcopy(upstream)
    payload["input"].append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": LOCAL_COMPACTION_PROMPT}],
        }
    )
    payload["tool_choice"] = "none"
    payload["parallel_tool_calls"] = False
    return payload


def _local_replacement_history(input_items: list[dict[str, Any]], summary: str) -> list[dict[str, Any]]:
    """Повторяет local-путь Codex: оставляет до 20k токенов последних user-входов."""
    selected: list[dict[str, Any]] = []
    remaining = LOCAL_COMPACTION_RETAINED_USER_TOKENS
    for item in reversed(input_items):
        if item.get("role") != "user":
            continue
        tokens = max(1, (len(json.dumps(item, ensure_ascii=False)) + 3) // 4)
        if tokens > remaining:
            continue
        selected.append(copy.deepcopy(item))
        remaining -= tokens
        if remaining == 0:
            break
    selected.reverse()
    selected.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": f"{LOCAL_SUMMARY_PREFIX}\n{summary.strip()}"}],
        }
    )
    return selected


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

    def _headers(self, tokens: Any, identity: CodexRequestIdentity, *, accept: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {tokens.access}",
            "Content-Type": "application/json",
            "Accept": accept,
            # ChatGPT backend использует эту идентичность при маршрутизации
            # кэша между ходами одного Claude Code сеанса.
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
        return headers

    async def events(
        self, payload: dict[str, Any], identity: CodexRequestIdentity
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        stale_access: str | None = None
        auth_retried = False
        transport_retried = False
        while True:
            tokens = await self.auth.get(force_refresh=auth_retried, stale_access=stale_access)
            headers = self._headers(tokens, identity, accept="text/event-stream")
            emitted_event = False
            try:
                async with self.client.stream(
                    "POST", self.endpoint, headers=headers, json=payload
                ) as response:
                    if response.status_code == 401 and not auth_retried:
                        stale_access = tokens.access
                        auth_retried = True
                        continue
                    if response.is_error:
                        body = (await response.aread()).decode(errors="replace")
                        raise BackendError(
                            response.status_code, body[:500], response.headers.get("retry-after")
                        )
                    async for event, data in _sse(response):
                        emitted_event = True
                        _log_upstream_cache_usage(event, data, identity)
                        yield event, data
                    return
            except httpx.TransportError as exc:
                # Повторять можно только запрос, с которого ещё не поступил ни один
                # SSE event: иначе клиент получит дублированные tool/text deltas.
                if emitted_event or transport_retried:
                    raise
                transport_retried = True
                print(
                    "codex_upstream_retry "
                    f"client_id={identity.installation_id} session_source={identity.session_source} "
                    f"session_id={identity.session_id} error={type(exc).__name__}",
                    flush=True,
                )

    async def compact(self, payload: dict[str, Any], identity: CodexRequestIdentity) -> list[dict[str, Any]]:
        """Запускает штатный Codex endpoint и возвращает replacement history."""
        stale_access: str | None = None
        endpoint = f"{self.endpoint.rstrip('/')}/compact"
        for attempt in range(2):
            tokens = await self.auth.get(force_refresh=attempt == 1, stale_access=stale_access)
            response = await self.client.post(
                endpoint,
                headers=self._headers(tokens, identity, accept="application/json"),
                json=payload,
            )
            if response.status_code == 401 and attempt == 0:
                stale_access = tokens.access
                continue
            if response.is_error:
                raise BackendError(
                    response.status_code,
                    response.text[:500],
                    response.headers.get("retry-after"),
                )
            try:
                result = response.json()
            except json.JSONDecodeError as exc:
                raise RuntimeError("Codex compact returned invalid JSON") from exc
            output = result.get("output") if isinstance(result, dict) else None
            if not isinstance(output, list) or not all(isinstance(item, dict) for item in output):
                raise RuntimeError("Codex compact response has no valid output history")
            return output
        raise RuntimeError("Codex compact authentication retry was not attempted")

    async def summarize(self, payload: dict[str, Any], identity: CodexRequestIdentity) -> str:
        """Собирает текст local compact из потокового Responses-ответа."""
        parts: list[str] = []
        async for event, data in self.events(payload, identity):
            if event == "response.output_text.delta" and isinstance(data.get("delta"), str):
                parts.append(data["delta"])
            elif event == "response.failed":
                response = data.get("response")
                error = response.get("error") if isinstance(response, dict) else None
                message = error.get("message") if isinstance(error, dict) else None
                raise RuntimeError(message or "Codex local compact failed")
        summary = "".join(parts).strip()
        if not summary:
            raise RuntimeError("Codex local compact returned no summary")
        return summary


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
    session_identities: dict[tuple[str, str], CodexRequestIdentity] = {}
    compaction_states: dict[tuple[str, str], list[CompactionState]] = {}
    remote_compact_available: dict[tuple[str, str], bool] = {}
    branch_counter = 0
    compact_at_tokens = _compact_at_tokens()
    remote_compact_enabled = _remote_compact_enabled()

    def identity_for(session_source: str, session_id: str) -> CodexRequestIdentity:
        key = (session_source, session_id)
        identity = session_identities.get(key)
        if identity is None:
            identity = CodexRequestIdentity(
                installation_id=installation_id,
                session_id=session_id,
                session_source=session_source,
                thread_id=str(uuid.uuid4()),
                window_id=str(uuid.uuid4()),
            )
            session_identities[key] = identity
        return identity

    def compaction_state_for(
        session_source: str,
        session_id: str,
        identity: CodexRequestIdentity,
        raw_input: list[dict[str, Any]],
    ) -> CompactionState:
        """Выбирает ветку по самому длинному ранее отправленному префиксу."""
        nonlocal branch_counter
        key = (session_source, session_id)
        states = compaction_states.setdefault(key, [])
        candidates = [
            state for state in states if _suffix_after_prefix(raw_input, state.last_raw_input) is not None
        ]
        branch_counter += 1
        if candidates:
            state = max(candidates, key=lambda candidate: len(candidate.last_raw_input))
            state.last_raw_input = copy.deepcopy(raw_input)
            state.last_used = branch_counter
            return state

        if len(states) >= MAX_COMPACTION_BRANCHES_PER_SESSION:
            states.remove(min(states, key=lambda candidate: candidate.last_used))
        branch_identity = (
            identity
            if not states
            else CodexRequestIdentity(
                installation_id=identity.installation_id,
                session_id=identity.session_id,
                session_source=identity.session_source,
                thread_id=str(uuid.uuid4()),
                window_id=str(uuid.uuid4()),
            )
        )
        state = CompactionState(
            branch_id=str(uuid.uuid4()),
            identity=branch_identity,
            last_raw_input=copy.deepcopy(raw_input),
            last_used=branch_counter,
        )
        states.append(state)
        return state

    def advance_context_window(identity: CodexRequestIdentity) -> CodexRequestIdentity:
        """Как Codex: compact начинает новое cache/context window того же thread."""
        return CodexRequestIdentity(
            installation_id=identity.installation_id,
            session_id=identity.session_id,
            session_source=identity.session_source,
            thread_id=identity.thread_id,
            window_id=str(uuid.uuid4()),
        )

    def request_session_identity(request: Request) -> tuple[str, str]:
        for header in ("anthropic-session-id", "session-id", "x-session-id"):
            if value := request.headers.get(header):
                return header, value
        return "default", "claude-codex"

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

    @app.exception_handler(httpx.HTTPError)
    async def backend_transport_error(_: Request, exc: httpx.HTTPError) -> JSONResponse:
        """Не допускает ASGI traceback, если upstream разорвал соединение."""
        return JSONResponse(status_code=502, content=_stream_error(exc))

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

    async def compact_if_needed(
        *,
        session_source: str,
        session_id: str,
        identity: CodexRequestIdentity,
        state: CompactionState,
        upstream: dict[str, Any],
    ) -> CodexRequestIdentity:
        """Подменяет полный Claude-префикс результатом `/responses/compact`."""
        session_key = (session_source, session_id)
        raw_input = copy.deepcopy(upstream["input"])
        if state.original_prefix is not None and state.replacement_history is not None:
            suffix = _suffix_after_prefix(raw_input, state.original_prefix)
            if suffix is None:
                # Claude Code прислал историю, не продолжающую сохранённый префикс:
                # не смешиваем её с чужим summary и начинаем отсчёт заново.
                state.last_input_tokens = 0
                state.original_prefix = None
                state.replacement_history = None
                print(
                    "codex_compact "
                    f"client_id={identity.installation_id} session_source={identity.session_source} "
                    f"session_id={identity.session_id} branch_id={state.branch_id} "
                    "result=discarded reason=history_changed",
                    flush=True,
                )
            else:
                upstream["input"] = copy.deepcopy(state.replacement_history) + suffix

        if not compact_at_tokens or state.last_input_tokens < compact_at_tokens:
            return identity

        input_tokens_before = state.last_input_tokens
        print(
            "codex_compact "
            f"client_id={identity.installation_id} session_source={identity.session_source} "
            f"session_id={identity.session_id} branch_id={state.branch_id} "
            f"result=started input_tokens={input_tokens_before} "
            f"threshold={compact_at_tokens}",
            flush=True,
        )
        replacement_history: list[dict[str, Any]]
        implementation = "remote"
        if remote_compact_enabled and remote_compact_available.get(session_key, True):
            try:
                replacement_history = _sanitize_remote_replacement_history(
                    await backend.compact(_compact_payload(upstream), identity)
                )
                if not replacement_history:
                    raise RuntimeError("Codex remote compact returned no usable replacement history")
            except (BackendError, httpx.HTTPError, RuntimeError) as exc:
                # Некоторые ChatGPT backend-кластеры принимают `/responses`, но
                # закрывают `/responses/compact` без HTTP-ответа. Не повторяем
                # заведомо неработающий remote путь в этой сессии.
                remote_compact_available[session_key] = False
                implementation = "local"
                print(
                    "codex_compact "
                    f"client_id={identity.installation_id} session_source={identity.session_source} "
                    f"session_id={identity.session_id} branch_id={state.branch_id} "
                    "result=remote_unavailable "
                    f"error={type(exc).__name__}",
                    flush=True,
                )
            else:
                implementation = "remote"
        else:
            implementation = "local"

        if implementation == "local":
            try:
                summary = await backend.summarize(_local_compact_payload(upstream), identity)
                replacement_history = _local_replacement_history(upstream["input"], summary)
            except (BackendError, httpx.HTTPError, RuntimeError) as exc:
                # Нельзя отдавать Claude Code ASGI traceback из-за внутреннего
                # обслуживания контекста. Пропускаем один compact и позволяем
                # обычному запросу завершиться своим upstream-ответом.
                state.last_input_tokens = 0
                print(
                    "codex_compact "
                    f"client_id={identity.installation_id} session_source={identity.session_source} "
                    f"session_id={identity.session_id} branch_id={state.branch_id} "
                    "result=error implementation=local "
                    f"error={type(exc).__name__}",
                    flush=True,
                )
                return identity

        state.original_prefix = raw_input
        state.replacement_history = replacement_history
        state.last_input_tokens = 0
        upstream["input"] = copy.deepcopy(replacement_history)
        identity = advance_context_window(identity)
        state.identity = identity
        upstream["client_metadata"] = identity.client_metadata()
        print(
            "codex_compact "
            f"client_id={identity.installation_id} session_source={identity.session_source} "
            f"session_id={identity.session_id} branch_id={state.branch_id} "
            f"result=success implementation={implementation} "
            f"input_tokens={input_tokens_before} replacement_items={len(replacement_history)}",
            flush=True,
        )
        return identity

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request_body(request)
        if isinstance(body, JSONResponse):
            return body
        requested_model = str(body.get("model") or "claude-codex")
        codex_model = os.environ.get("CLAUDE_CODEX_MODEL", "gpt-5.6-sol")
        reasoning = os.environ.get("CLAUDE_CODEX_REASONING", "medium")
        session_source, session_id = request_session_identity(request)
        session_identity = identity_for(session_source, session_id)
        upstream = to_responses_request(
            body,
            model=codex_model,
            reasoning_effort=reasoning,
            prompt_cache_key=session_identity.session_id,
        )
        compaction_state = compaction_state_for(
            session_source,
            session_id,
            session_identity,
            upstream["input"],
        )
        identity = compaction_state.identity
        upstream["client_metadata"] = identity.client_metadata()
        identity = await compact_if_needed(
            session_source=session_source,
            session_id=session_id,
            identity=identity,
            state=compaction_state,
            upstream=upstream,
        )

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
                            input_tokens = _reported_input_tokens(event, data)
                            if input_tokens is not None:
                                compaction_state.last_input_tokens = input_tokens
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
            input_tokens = _reported_input_tokens(event, data)
            if input_tokens is not None:
                compaction_state.last_input_tokens = input_tokens
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
