from __future__ import annotations

import ssl
from unittest.mock import AsyncMock, Mock

import httpx

from claude_codex.auth import AuthManager, Tokens
from claude_codex.proxy import create_app


def assert_system_trust(client_factory: Mock) -> None:
    context = client_factory.call_args.kwargs["verify"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    assert set(context.get_ca_certs(binary_form=True)) == set(
        ssl.create_default_context().get_ca_certs(binary_form=True)
    )


async def test_proxy_uses_system_trust(monkeypatch, tmp_path) -> None:
    client = AsyncMock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    app = create_app(installation_id_path=tmp_path / "installation_id")
    assert_system_trust(factory)
    async with app.router.lifespan_context(app):
        pass
    client.aclose.assert_awaited_once()


async def test_standalone_refresh_uses_system_trust(monkeypatch, tmp_path) -> None:
    client = AsyncMock()
    client.post.return_value = httpx.Response(200, json={"access_token": "new-access"})
    factory = Mock(return_value=client)
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    manager = AuthManager(cache_path=tmp_path / "auth.json")
    result = await manager._refresh(Tokens("old-access", "refresh", 0, None, "test"))
    assert result.access == "new-access"
    assert_system_trust(factory)
    client.aclose.assert_awaited_once()
