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


async def test_concurrent_refresh_is_deduplicated(tmp_path: Path) -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
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
