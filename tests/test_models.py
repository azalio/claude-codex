from __future__ import annotations

import json

import httpx
import pytest
from test_proxy import FakeAuth, isolated_installation_id  # noqa: F401

from claude_codex.models import ModelCatalog
from claude_codex.proxy import create_app


def catalog_response():
    return {"models": [
        {"slug": "gpt-first", "display_name": "First", "visibility": "list"},
        {"slug": "gpt-second", "display_name": "Second", "visibility": "list"},
        {"slug": "codex-auto-review", "visibility": "hide"},
        {"slug": "bad/slug", "visibility": "list"},
        {"slug": "gpt-first", "visibility": "list"},
        {"slug": "invisible"},
    ]}


async def test_catalog_uses_subscription_auth_and_filters_hidden_models():
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, json=catalog_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        catalog = ModelCatalog(FakeAuth(), client, "https://codex.test/responses")
        entries = await catalog.list()
        assert await catalog.resolve("claude-codex/gpt-second") == "gpt-second"
        assert await catalog.resolve("claude-opus") is None
        assert await catalog.list() == entries
    assert len(calls) == 1
    assert calls[0].url.path == "/models"
    assert calls[0].url.params["client_version"] == "0.160.0"
    assert calls[0].headers["authorization"] == "Bearer access"
    assert calls[0].headers["chatgpt-account-id"] == "acc-123"
    assert [entry["id"] for entry in entries] == [
        "claude-codex/gpt-first", "claude-codex/gpt-second",
    ]


async def test_catalog_refreshes_auth_once_on_401():
    refreshes = []

    class Auth(FakeAuth):
        async def get(self, **kwargs):
            refreshes.append(kwargs)
            return await super().get(**kwargs)

    def upstream(request):
        return httpx.Response(
            401 if len(refreshes) == 1 else 200,
            json=catalog_response(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        catalog = ModelCatalog(Auth(), client, "https://codex.test/responses")
        assert len(await catalog.list()) == 2
    assert refreshes == [
        {"force_refresh": False, "stale_access": None},
        {"force_refresh": True, "stale_access": "access"},
    ]


@pytest.mark.parametrize("failure", ["503", "redirect", "bad_json", "timeout"])
async def test_catalog_preserves_last_good_list_on_failure(failure):
    calls = []

    def upstream(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, json=catalog_response())
        if failure == "timeout":
            raise httpx.ReadTimeout("no data")
        if failure == "bad_json":
            return httpx.Response(200, text="not json")
        return httpx.Response(
            302 if failure == "redirect" else 503,
            headers={"location": "https://untrusted.test/models"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        catalog = ModelCatalog(FakeAuth(), client, "https://codex.test/responses")
        good = await catalog.list()
        catalog.expires = 0
        assert await catalog.list() == good
        assert await catalog.list() == good
    assert len(calls) == 2


async def test_picker_selection_routes_main_and_preserves_auxiliary_model(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODEX_MODEL", "gpt-default")
    monkeypatch.setenv("CLAUDE_CODEX_AUXILIARY_MODEL", "gpt-auxiliary")
    calls = []

    def upstream(request):
        if request.method == "GET":
            assert "x-api-key" not in request.headers
            return httpx.Response(200, json=catalog_response())
        calls.append(json.loads(request.content))
        return httpx.Response(200, text=(
            'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
            'data: {"type":"response.completed","response":{"usage":'
            '{"input_tokens":1,"output_tokens":1}}}\n\n'
        ))

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:
        app = create_app(auth=FakeAuth(), client=backend, endpoint="https://codex.test/responses")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            models = await client.get("/v1/models", headers={"x-api-key": "local-secret"})
            assert len(models.json()["data"]) == 2
            body = {"model": "claude-codex/gpt-second", "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}]}
            for request_class in ("main", "subagent", "auxiliary"):
                response = await client.post(
                    "/v1/messages", json=body,
                    headers={"x-claude-code-request-class": request_class},
                )
                assert response.status_code == 200
            body["model"] = "claude-codex/codex-auto-review"
            assert (await client.post("/v1/messages", json=body)).status_code == 400
    assert [call["model"] for call in calls] == ["gpt-second", "gpt-second", "gpt-auxiliary"]


async def test_configured_backend_remains_usable_when_catalog_is_down(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODEX_MODEL", "gpt-configured")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))
    ) as client:
        catalog = ModelCatalog(FakeAuth(), client, "https://codex.test/responses")
        assert await catalog.resolve("claude-codex/gpt-configured") == "gpt-configured"
        with pytest.raises(ValueError):
            await catalog.resolve("claude-codex/gpt-other")
