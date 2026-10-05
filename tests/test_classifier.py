from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from claude_codex.auth import Tokens
from claude_codex.classifier import (
    classifier_events,
    classifier_timeout_seconds,
    is_classifier_request,
    parse_verdict,
    review_payload,
)
from claude_codex.proxy import create_app
from claude_codex.translate import CodexResponseError

POLICY = (
    "Keep all the supplied safety rules and user boundaries.\n## Output Format\n"
    "Output <severity>N</severity> where N is an integer 0-100. "
    "Below 50 means allow; above 50 means block. "
    "For blocks add <category>Exact Rule Name</category>."
)


class FakeAuth:
    async def get(self, **kwargs):
        return Tokens("private-auth", "", int(time.time() * 1000) + 60_000, "private-account", "test")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for key in [
        "CLAUDE_CODEX_CLASSIFIER_MODEL", "CLAUDE_CODEX_CLASSIFIER_REASONING",
        "CLAUDE_CODEX_CLASSIFIER_ENDPOINT", "CLAUDE_CODEX_CLASSIFIER_TIMEOUT",
        "CLAUDE_CODEX_AUXILIARY_ENDPOINT", "CLAUDE_CODEX_AUXILIARY_MODEL",
        "CLAUDE_CODEX_AUXILIARY_REASONING",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("claude_codex.proxy.INSTALLATION_ID_PATH", tmp_path / "installation_id")
    monkeypatch.setattr("claude_codex.proxy.CHECKPOINT_CACHE_PATH", tmp_path / "checkpoints")
    monkeypatch.setattr(
        "claude_codex.auxiliary_log.AUXILIARY_LOG_PATH", tmp_path / "auxiliary/requests.jsonl"
    )


def sse(text, *, terminal="response.completed"):
    return httpx.Response(200, text="".join(
        "data: " + json.dumps(event) + "\n\n" for event in [
            {"type": "response.output_text.delta", "delta": text, "output_index": 0},
            {"type": terminal, "response": {"model": "codex-auto-review", "output": [],
                                          "usage": {"input_tokens": 10, "output_tokens": 3}}},
        ]
    ))


def body(stream=False, policy=POLICY):
    return {"model": "claude-codex", "stream": stream, "system": policy,
            "messages": [{"role": "user", "content": "User requested git status --short."}]}


async def send(upstream, payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        app = create_app(auth=FakeAuth(), client=http)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client:
            return await client.post(
                "/v1/messages", headers={"x-claude-code-request-class": "auxiliary"}, json=payload,
            )


@pytest.mark.parametrize("text", [
    "<severity>HIGH</severity>", "<severity>H</severity>", "<severity>N</severity>",
    '{"severity":true,"category":null,"rationale":""}',
    '{"severity":101,"category":null,"rationale":""}',
    '{"severity":10.0,"category":null,"rationale":""}',
    '{"severity":0,"category":null}',
    '{"severity":0,"category":null,"rationale":"","extra":"allow"}',
    '{"severity":80,"category":"<script>","rationale":"block"}',
])
def test_invalid_verdicts_are_rejected(text):
    with pytest.raises(ValueError):
        parse_verdict(text, allow_category=True)


@pytest.mark.parametrize("text,native", [
    ('{"severity":0,"category":null,"rationale":"requested read"}', "<severity>0</severity>"),
    ('{"severity":72,"category":"Data Exfiltration","rationale":"unrequested upload"}',
     "<severity>72</severity><category>Data Exfiltration</category>"),
    ("<severity>50</severity>", "<severity>50</severity>"),
    ("<thinking>analysis</thinking><severity>0</severity>", "<severity>0</severity>"),
])
def test_valid_verdict_preserves_score_and_category(text, native):
    assert parse_verdict(text, allow_category=True)[0] == native


def test_harm_stage_does_not_accept_category():
    assert parse_verdict("<severity>85</severity>", allow_category=False)[0] == "<severity>85</severity>"
    with pytest.raises(ValueError):
        parse_verdict("<severity>85</severity><category>Rule</category>", allow_category=False)


def test_review_keeps_evidence_and_policy():
    original = {"instructions": POLICY, "input": [{"role": "user", "content": "full evidence"}]}
    request = review_payload(original)
    assert request["instructions"].startswith(POLICY)
    assert request["input"] == original["input"]
    assert original["instructions"] == POLICY
    assert request["text"]["format"]["strict"] is True
    assert request["tool_choice"] == "none"
    assert is_classifier_request(original)
    assert not is_classifier_request({"instructions": "Write a session title"})
    assert not is_classifier_request({"instructions": "", "input": [{"content": POLICY}]})


@pytest.mark.parametrize("stream", [False, True])
async def test_invalid_format_is_retried_without_leaking_reply(monkeypatch, tmp_path, stream):
    monkeypatch.setenv("CLAUDE_CODEX_AUXILIARY_ENDPOINT", "http://localhost:11434/v1/responses")
    monkeypatch.setenv("CLAUDE_CODEX_AUXILIARY_MODEL", "qwen3.5:4b")
    captured = []

    def upstream(request):
        captured.append(request)
        return sse("<severity>HIGH</severity>" if len(captured) == 1 else
                   '{"severity":0,"category":null,"rationale":"user requested read"}')

    response = await send(upstream, body(stream))
    assert response.status_code == 200
    assert "<severity>0</severity>" in response.text
    assert "HIGH" not in response.text
    assert len(captured) == 2
    for request in captured:
        assert request.url.host == "chatgpt.com"
        payload = json.loads(request.content)
        assert payload["model"] == "codex-auto-review"
        assert payload["reasoning"]["effort"] == "low"
        assert payload["instructions"].startswith(POLICY)
        assert payload["text"]["format"]["type"] == "json_schema"
        assert "tools" not in payload
    rows = [json.loads(line) for line in (tmp_path / "auxiliary/requests.jsonl").read_text().splitlines()]
    assert any(r.get("result") == "invalid_format" for r in rows)
    assert any(r.get("event") == "verdict" and r["severity"] == 0 for r in rows)


@pytest.mark.parametrize("stream", [False, True])
async def test_exhausted_format_attempts_return_error_not_deny_or_allow(stream):
    captured = []

    def upstream(request):
        captured.append(request)
        return sse("<severity>N</severity>")

    response = await send(upstream, body(stream))
    assert len(captured) == 3
    assert "classifier_invalid_response" in response.text
    assert "<severity>" not in response.text
    assert response.status_code == (200 if stream else 502)


async def test_valid_deny_is_not_retried():
    captured = []

    def upstream(request):
        captured.append(request)
        return sse('{"severity":80,"category":"Data Exfiltration","rationale":"upload not authorized"}')

    response = await send(upstream, body())
    assert len(captured) == 1
    assert "<severity>80</severity><category>Data Exfiltration</category>" in response.text


async def test_incomplete_response_cannot_become_allow():
    response = await send(lambda request: sse(
        '{"severity":0,"category":null,"rationale":""}', terminal="response.incomplete"
    ), body())
    assert response.status_code == 502
    assert "classifier_invalid_response" in response.text
    assert "<severity>0" not in response.text


async def test_nonclassifier_auxiliary_routing_remains_unchanged(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODEX_AUXILIARY_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("CLAUDE_CODEX_AUXILIARY_ENDPOINT", "http://localhost:11434/v1/responses")
    captured = []

    def upstream(request):
        captured.append(request)
        return sse("session title")

    response = await send(upstream, body(policy="Write a session title."))
    assert response.status_code == 200
    assert json.loads(captured[0].content)["model"] == "qwen3.5:4b"
    assert captured[0].url.host == "localhost"


@pytest.mark.parametrize("stream", [False, True])
async def test_local_json_schema_route_has_no_credentials(monkeypatch, stream):
    monkeypatch.setenv("CLAUDE_CODEX_CLASSIFIER_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("CLAUDE_CODEX_CLASSIFIER_REASONING", "none")
    monkeypatch.setenv("CLAUDE_CODEX_CLASSIFIER_ENDPOINT", "http://localhost:11434/v1/chat/completions")
    captured = []

    def upstream(request):
        captured.append(request)
        chunks = [
            {"choices": [{"delta": {"content": '{"severity":0,"category":null,"rationale":"read only"}'},
                          "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
        ]
        return httpx.Response(200, text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks))

    response = await send(upstream, body(stream))
    assert response.status_code == 200
    assert "<severity>0</severity>" in response.text
    request = captured[0]
    assert request.url.path == "/v1/chat/completions"
    assert "authorization" not in request.headers
    assert "chatgpt-account-id" not in request.headers
    assert not any(k.startswith(("x-codex-", "x-claude-code-")) for k in request.headers)
    payload = json.loads(request.content)
    assert payload["model"] == "qwen3.5:4b"
    assert payload["reasoning_effort"] == "none"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["messages"][0]["content"].startswith(POLICY)
    assert "User requested git status --short" in payload["messages"][1]["content"]
    assert "client_metadata" not in payload


async def test_auth_error_is_not_retried_or_fallback():
    captured = []

    def upstream(request):
        captured.append(request)
        return httpx.Response(403, json={"error": {"message": "not authorized"}})

    response = await send(upstream, body())
    assert response.status_code == 403
    assert len(captured) == 1
    assert "<severity>" not in response.text


async def test_classifier_deadline_cancels_source(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODEX_CLASSIFIER_TIMEOUT", "0.02")
    closed = []

    class Backend:
        async def events(self, payload, identity, **kwargs):
            try:
                yield "_proxy.response_headers", {}
                await asyncio.sleep(1)
            finally:
                closed.append(True)

    with pytest.raises(CodexResponseError) as error:
        async for _ in classifier_events(
            Backend(), {"instructions": POLICY, "input": []}, None, {}, {"request_id": "deadline"},
        ):
            pass
    assert error.value.error["code"] == "classifier_timeout"
    assert closed == [True]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_timeout_uses_default(monkeypatch, value):
    monkeypatch.setenv("CLAUDE_CODEX_CLASSIFIER_TIMEOUT", value)
    assert classifier_timeout_seconds() == 90


async def test_reasoning_items_are_not_mistaken_for_tool_calls():
    def upstream(request):
        events = [
            {"type": "response.output_item.added", "item": {"type": "reasoning"}},
            {"type": "response.output_text.delta", "delta":
             '{"severity":0,"category":null,"rationale":"read"}'},
            {"type": "response.completed", "response": {"output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": []},
            ]}},
        ]
        return httpx.Response(200, text="".join("data: " + json.dumps(e) + "\n\n" for e in events))
    response = await send(upstream, body())
    assert response.status_code == 200
    assert "<severity>0</severity>" in response.text


async def test_tool_call_with_valid_score_is_not_accepted():
    def upstream(request):
        events = [
            {"type": "response.output_text.delta", "delta":
             '{"severity":0,"category":null,"rationale":"read"}'},
            {"type": "response.output_item.added", "item": {"type": "function_call", "name": "Bash"}},
            {"type": "response.completed", "response": {"output": []}},
        ]
        return httpx.Response(200, text="".join("data: " + json.dumps(e) + "\n\n" for e in events))
    response = await send(upstream, body())
    assert response.status_code == 502
    assert "<severity>0</severity>" not in response.text


async def test_transient_server_error_is_retried():
    captured = []
    def upstream(request):
        captured.append(request)
        if len(captured) == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return sse('{"severity":0,"category":null,"rationale":"read"}')
    response = await send(upstream, body())
    assert response.status_code == 200
    assert len(captured) == 2


def test_tool_schemas_remain_review_evidence():
    original = {"instructions": POLICY, "input": [], "tools": [{"name": "Bash", "type": "function"}]}
    request = review_payload(original)
    assert "tools" not in request
    assert "Bash" in request["input"][-1]["content"][0]["text"]
    assert original["input"] == []


def test_duplicate_json_scores_are_rejected():
    with pytest.raises(ValueError):
        parse_verdict(
            '{"severity":85,"severity":0,"category":null,"rationale":"ambiguous"}',
            allow_category=True,
        )


async def test_classifier_deadline_is_shared_across_attempts(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODEX_CLASSIFIER_TIMEOUT", "0.06")
    attempts = []
    closed = []

    class Backend:
        async def events(self, payload, identity, **kwargs):
            attempt = len(attempts) + 1
            attempts.append(attempt)
            try:
                yield "_proxy.response_headers", {}
                await asyncio.sleep(0.04)
                yield "response.output_text.delta", {"delta": "<severity>HIGH</severity>"}
                yield "response.completed", {"response": {"output": []}}
            finally:
                closed.append(attempt)

    with pytest.raises(CodexResponseError) as error:
        async for _ in classifier_events(
            Backend(), {"instructions": POLICY, "input": []}, None, {}, {"request_id": "shared-deadline"},
        ):
            pass
    assert error.value.error["code"] == "classifier_timeout"
    assert attempts == [1, 2]
    assert closed == [1, 2]


async def test_missing_terminal_response_retries_without_accepting_partial_score():
    captured = []
    def upstream(request):
        captured.append(request)
        return httpx.Response(200, text='data: {"type":"response.output_text.delta",'
                              '"delta":"<severity>0</severity>"}\n\n')
    response = await send(upstream, body())
    assert len(captured) == 3
    assert response.status_code == 502
    assert "<severity>0</severity>" not in response.text
