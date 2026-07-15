from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

import httpx

from claude_codex.auth import AuthManager, _from_cache


def jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def test_loads_opencode_oauth(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path
    auth_file = home / ".local" / "share" / "opencode" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    access = jwt({"exp": int(time.time()) + 3600, "chatgpt_account_id": "acc-123"})
    auth_file.write_text(
        json.dumps(
            {
                "openai": {
                    "type": "oauth",
                    "access": access,
                    "refresh": "refresh-token",
                    "expires": int(time.time() * 1000) + 3_600_000,
                }
            }
        )
    )
    monkeypatch.setattr(Path, "home", lambda: home)

    tokens = AuthManager().load()

    assert tokens.account_id == "acc-123"
    assert tokens.source == "opencode:openai"


def test_explicit_file_accepts_codex_schema(monkeypatch, tmp_path: Path) -> None:
    auth_file = tmp_path / "codex.json"
    access = jwt({"exp": int(time.time()) + 3600})
    auth_file.write_text(json.dumps({"tokens": {"access_token": access, "refresh_token": "refresh"}}))
    monkeypatch.setenv("CLAUDE_CODEX_AUTH_FILE", str(auth_file))

    assert AuthManager(cache_path=tmp_path / "cache.json").load().access == access


def test_skips_malformed_and_unusable_sources(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]")
    expired = tmp_path / "expired.json"
    expired.write_text(json.dumps({"access": jwt({"exp": int(time.time()) - 60})}))
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            {
                "access": jwt({"exp": int(time.time()) + 3600}),
                "refresh": "refresh",
                "expires": int(time.time() * 1000) + 3_600_000,
            }
        )
    )
    manager = AuthManager(cache_path=tmp_path / "cache.json")
    manager._sources = lambda: [
        (malformed, _from_cache),
        (expired, _from_cache),
        (valid, _from_cache),
    ]

    assert manager.load().access == json.loads(valid.read_text())["access"]


async def test_concurrent_refresh_is_deduplicated(tmp_path: Path) -> None:
    requests = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={
                "access_token": jwt({"exp": int(time.time()) + 3600, "chatgpt_account_id": "acc-new"}),
                "refresh_token": "refresh-new",
                "expires_in": 3600,
            },
        )

    auth_file = tmp_path / "source.json"
    auth_file.write_text(
        json.dumps(
            {
                "access": jwt({"exp": int(time.time()) - 60}),
                "refresh": "refresh-old",
                "expires": 0,
                "account_id": "acc-old",
            }
        )
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    manager = AuthManager(cache_path=tmp_path / "cache.json", issuer="https://auth.test", client=client)
    manager._sources = lambda: [(auth_file, _from_cache)]

    first, second = await asyncio.gather(manager.get(), manager.get())

    assert requests == 1
    assert first.access == second.access
    assert first.account_id == "acc-new"
    assert (tmp_path / "cache.json").stat().st_mode & 0o777 == 0o600
    await client.aclose()


async def test_independent_managers_share_one_refresh(tmp_path: Path) -> None:
    requests = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={
                "access_token": jwt({"exp": int(time.time()) + 3600}),
                "refresh_token": "refresh-new",
                "expires_in": 3600,
            },
        )

    cache = tmp_path / "cache.json"
    cache.write_text(
        json.dumps(
            {
                "access": jwt({"exp": int(time.time()) - 60}),
                "refresh": "refresh-old",
                "expires": 0,
            }
        )
    )
    clients = [httpx.AsyncClient(transport=httpx.MockTransport(handler)) for _ in range(2)]
    managers = [
        AuthManager(cache_path=cache, issuer="https://auth.test", client=client) for client in clients
    ]

    first, second = await asyncio.gather(*(manager.get() for manager in managers))

    assert requests == 1
    assert first.access == second.access
    assert json.loads(cache.read_text())["access"] == first.access
    await asyncio.gather(*(client.aclose() for client in clients))
