from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any

import httpx

from .auth import AuthError, AuthProvider

MODEL_PREFIX = "claude-codex/"
CATALOG_CLIENT_VERSION = "0.160.0"
MODEL_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class ModelCatalog:
    """Account-scoped catalog, cached only in this proxy process."""

    def __init__(self, auth: AuthProvider, client: httpx.AsyncClient, endpoint: str) -> None:
        self.auth = auth
        self.client = client
        self.endpoint = endpoint.rstrip("/").rsplit("/", 1)[0] + "/models"
        self.entries: list[dict[str, Any]] = []
        self.expires = 0.0
        self.lock = asyncio.Lock()

    async def list(self) -> list[dict[str, Any]]:
        if time.monotonic() < self.expires:
            return self.entries
        async with self.lock:
            if time.monotonic() < self.expires:
                return self.entries
            try:
                async with asyncio.timeout(8):
                    stale_access = None
                    for attempt in range(2):
                        tokens = await self.auth.get(
                            force_refresh=attempt == 1,
                            stale_access=stale_access,
                        )
                        headers = {
                            "Authorization": f"Bearer {tokens.access}",
                            "originator": "codex_cli_rs",
                            "User-Agent": f"Codex/{CATALOG_CLIENT_VERSION}",
                            "Accept": "application/json",
                        }
                        if tokens.account_id:
                            headers["ChatGPT-Account-Id"] = tokens.account_id
                        response = await self.client.get(
                            self.endpoint, headers=headers,
                            params={"client_version": os.environ.get(
                                "CLAUDE_CODEX_CATALOG_CLIENT_VERSION", CATALOG_CLIENT_VERSION
                            )},
                            follow_redirects=False, timeout=8,
                        )
                        if response.status_code == 401 and attempt == 0:
                            stale_access = tokens.access
                            continue
                        response.raise_for_status()
                        raw = response.json().get("models")
                        if not isinstance(raw, list):
                            raise ValueError("Invalid catalog")
                        entries = []
                        seen = set()
                        for model in raw:
                            if not isinstance(model, dict) or model.get("visibility") != "list":
                                continue
                            slug = model.get("slug")
                            if not isinstance(slug, str) or not MODEL_SLUG.fullmatch(slug) or slug in seen:
                                continue
                            seen.add(slug)
                            name = model.get("display_name")
                            description = model.get("description")
                            entries.append({
                                "type": "model", "id": MODEL_PREFIX + slug,
                                "display_name": name if isinstance(name, str) and name else slug,
                                "description": description if isinstance(description, str) else (
                                    f"ChatGPT subscription: {slug}"
                                ),
                            })
                        self.entries = entries
                        self.expires = time.monotonic() + 300
                        return self.entries
            except (AuthError, httpx.HTTPError, TimeoutError, ValueError, TypeError, AttributeError):
                # No credentials, response bodies, or task content in diagnostics.
                print("model_catalog result=unavailable", flush=True)
                self.expires = time.monotonic() + 30
            return self.entries

    async def resolve(self, requested: str) -> str | None:
        if not requested.startswith(MODEL_PREFIX):
            return None
        entries = await self.list()
        if not any(entry["id"] == requested for entry in entries):
            raise ValueError("Selected model is not in the subscription catalog; restart to refresh")
        return requested.removeprefix(MODEL_PREFIX)
