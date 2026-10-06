from __future__ import annotations

import ssl
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import certifi
import httpx
import pytest
import truststore

from claude_codex import tls
from claude_codex.auth import AuthManager, Tokens
from claude_codex.proxy import create_app
from claude_codex.tls import upstream_ssl_context

# Сертификат и ключ созданы только для локального тестового сервера.
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def clear_ca_overrides(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)


def assert_native_trust(client_factory: Mock) -> None:
    context = client_factory.call_args.kwargs["verify"]
    assert isinstance(context, truststore.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


@pytest.mark.parametrize("empty_system_store", [False, True])
async def test_proxy_uses_system_trust(monkeypatch, tmp_path, empty_system_store) -> None:
    if empty_system_store:
        monkeypatch.setattr(ssl.SSLContext, "load_default_certs", lambda *args, **kwargs: None)
    client = AsyncMock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    app = create_app(installation_id_path=tmp_path / "installation_id")
    assert_native_trust(factory)
    async with app.router.lifespan_context(app):
        pass
    client.aclose.assert_awaited_once()


@pytest.mark.parametrize("empty_system_store", [False, True])
async def test_standalone_refresh_uses_system_trust(monkeypatch, tmp_path, empty_system_store) -> None:
    if empty_system_store:
        monkeypatch.setattr(ssl.SSLContext, "load_default_certs", lambda *args, **kwargs: None)
    client = AsyncMock()
    client.post.return_value = httpx.Response(200, json={"access_token": "new-access"})
    factory = Mock(return_value=client)
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    manager = AuthManager(cache_path=tmp_path / "auth.json")
    result = await manager._refresh(Tokens("old-access", "refresh", 0, None, "test"))
    assert result.access == "new-access"
    assert_native_trust(factory)
    client.aclose.assert_awaited_once()


@pytest.fixture
def tls_endpoint():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"trusted")

        def log_message(self, *args):
            pass

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(FIXTURES / "tls-localhost.pem", FIXTURES / "tls-localhost.key")
    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://localhost:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def native_context_factory(monkeypatch):
    factory = Mock(side_effect=ssl.SSLContext)
    monkeypatch.setattr(tls.truststore, "SSLContext", factory)
    monkeypatch.setattr(ssl.SSLContext, "load_default_certs", lambda *args, **kwargs: None)
    return factory


@pytest.mark.parametrize("source", ["native", "filesystem", "public", "file", "directory"])
async def test_tls_accepts_configured_organization_ca(
    monkeypatch,
    tmp_path,
    tls_endpoint,
    native_context_factory,
    source,
):
    certificate = FIXTURES / "tls-localhost.pem"
    if source == "native":

        def native_context(protocol):
            context = ssl.SSLContext(protocol)
            context.load_verify_locations(cafile=certificate)
            return context

        native_context_factory.side_effect = native_context
    elif source == "filesystem":
        monkeypatch.setattr(
            ssl.SSLContext,
            "load_default_certs",
            lambda context, *args, **kwargs: context.load_verify_locations(cafile=certificate),
        )
    elif source == "public":
        monkeypatch.setattr(certifi, "where", lambda: str(certificate))
    elif source == "file":
        monkeypatch.setenv("SSL_CERT_FILE", str(certificate))
        monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "unused-directory"))
    else:
        # Хеш subject тестового сертификата для OpenSSL capath.
        (tmp_path / "ce275665.0").write_bytes(certificate.read_bytes())
        monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    if source in {"file", "directory"}:
        public_bundle = Mock(side_effect=AssertionError("Explicit trust must not load public roots"))
        monkeypatch.setattr(certifi, "where", public_bundle)
    context = upstream_ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    async with httpx.AsyncClient(verify=context, trust_env=False) as client:
        response = await client.get(tls_endpoint)
    assert response.status_code == 200
    assert response.text == "trusted"
    if source in {"native", "filesystem", "public"}:
        native_context_factory.assert_called_once_with(ssl.PROTOCOL_TLS_CLIENT)
    else:
        native_context_factory.assert_not_called()
        public_bundle.assert_not_called()
        assert context.get_ca_certs(binary_form=True) == [
            ssl.PEM_cert_to_DER_cert(certificate.read_text()),
        ]


@pytest.mark.parametrize("trust_source", ["untrusted", "file"])
async def test_tls_rejects_untrusted_ca_and_wrong_hostname(monkeypatch, tls_endpoint, trust_source):
    if trust_source == "file":
        monkeypatch.setenv("SSL_CERT_FILE", str(FIXTURES / "tls-localhost.pem"))
    async with httpx.AsyncClient(verify=upstream_ssl_context(), trust_env=False) as client:
        if trust_source != "untrusted":
            response = await client.get(tls_endpoint)
            assert response.status_code == 200
            tls_endpoint = tls_endpoint.replace("localhost", "127.0.0.1")
        with pytest.raises(httpx.ConnectError) as error:
            await client.get(tls_endpoint)
    cause = error.value
    while cause is not None and not isinstance(cause, ssl.SSLCertVerificationError):
        cause = cause.__cause__ or cause.__context__
    assert isinstance(cause, ssl.SSLCertVerificationError)


def test_invalid_explicit_bundle_is_not_replaced_with_public_trust(
    monkeypatch,
    tmp_path,
    native_context_factory,
):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    with pytest.raises(FileNotFoundError):
        upstream_ssl_context()
    native_context_factory.assert_not_called()
