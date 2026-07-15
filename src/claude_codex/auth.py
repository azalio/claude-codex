from __future__ import annotations

import asyncio
import base64
import fcntl
import json
import os
import tempfile
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
ISSUER = "https://auth.openai.com"


class AuthError(RuntimeError):
    """Raised when no usable ChatGPT OAuth credentials are available."""


@dataclass(frozen=True, slots=True)
class Tokens:
    access: str
    refresh: str
    expires: int
    account_id: str | None
    source: str

    @property
    def fresh(self) -> bool:
        return bool(self.access) and self.expires > int(time.time() * 1000) + 60_000


def _jwt_claims(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    try:
        padding = "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + padding))
        return claims if isinstance(claims, dict) else {}
    except (ValueError, json.JSONDecodeError):
        return {}


def _account_id(*tokens: str) -> str | None:
    for token in tokens:
        claims = _jwt_claims(token)
        auth_claims = claims.get("https://api.openai.com/auth") or {}
        if not isinstance(auth_claims, Mapping):
            auth_claims = {}
        value = claims.get("chatgpt_account_id") or auth_claims.get("chatgpt_account_id")
        if value:
            return str(value)
        organizations = claims.get("organizations") or []
        if (
            isinstance(organizations, list)
            and organizations
            and isinstance(organizations[0], Mapping)
            and organizations[0].get("id")
        ):
            return str(organizations[0]["id"])
    return None


def _expiry_ms(access: str, explicit: Any = None) -> int:
    if explicit is not None:
        try:
            value = int(explicit)
            return value if value > 10_000_000_000 else value * 1000
        except (TypeError, ValueError):
            pass
    exp = _jwt_claims(access).get("exp")
    if exp:
        return int(exp) * 1000
    return 0


def _from_opencode(path: Path) -> Tokens | None:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("credential file must contain a JSON object")
    for provider in ("openai", "codex"):
        entry = data.get(provider) or {}
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "oauth" or not entry.get("access"):
            continue
        access = str(entry["access"])
        return Tokens(
            access=access,
            refresh=str(entry.get("refresh") or ""),
            expires=_expiry_ms(access, entry.get("expires")),
            account_id=entry.get("accountId") or _account_id(access),
            source=f"opencode:{provider}",
        )
    return None


def _from_codex(path: Path) -> Tokens | None:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("credential file must contain a JSON object")
    nested = data.get("tokens") or data
    if not isinstance(nested, dict):
        raise ValueError("Codex tokens must be a JSON object")
    access = nested.get("access_token") or nested.get("access")
    if not access:
        return None
    access = str(access)
    id_token = str(nested.get("id_token") or "")
    return Tokens(
        access=access,
        refresh=str(nested.get("refresh_token") or nested.get("refresh") or ""),
        expires=_expiry_ms(access, nested.get("expires")),
        account_id=nested.get("account_id") or nested.get("accountId") or _account_id(id_token, access),
        source="codex",
    )


def _from_cache(path: Path) -> Tokens | None:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("credential file must contain a JSON object")
    access = str(data.get("access") or "")
    if not access:
        return None
    return Tokens(
        access=access,
        refresh=str(data.get("refresh") or ""),
        expires=_expiry_ms(access, data.get("expires")),
        account_id=data.get("account_id") or _account_id(access),
        source="claude-codex-cache",
    )


class AuthProvider(Protocol):
    def load(self) -> Tokens: ...

    async def get(self, *, force_refresh: bool = False, stale_access: str | None = None) -> Tokens: ...


class AuthManager:
    def __init__(
        self,
        *,
        cache_path: Path | None = None,
        issuer: str = ISSUER,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.cache_path = cache_path or Path.home() / ".config" / "claude-codex" / "auth.json"
        self.issuer = issuer.rstrip("/")
        self._client = client
        self._tokens: Tokens | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _from_explicit(path: Path) -> Tokens | None:
        errors: list[Exception] = []
        for loader in (_from_cache, _from_codex, _from_opencode):
            try:
                tokens = loader(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(exc)
                continue
            if tokens:
                return tokens
        if errors:
            raise ValueError("unsupported credential file format") from errors[-1]
        return None

    def _sources(self) -> list[tuple[Path, Any]]:
        sources: list[tuple[Path, Any]] = []
        explicit = os.environ.get("CLAUDE_CODEX_AUTH_FILE")
        if explicit:
            sources.append((Path(explicit).expanduser(), self._from_explicit))
        sources.append((self.cache_path, _from_cache))
        sources.append((Path.home() / ".local" / "share" / "opencode" / "auth.json", _from_opencode))
        codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        sources.append((codex_home / "auth.json", _from_codex))
        return sources

    def load(self) -> Tokens:
        errors: list[str] = []
        for path, loader in self._sources():
            if not path.is_file():
                continue
            try:
                tokens = loader(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{path}: {exc}")
                continue
            if tokens and (tokens.fresh or tokens.refresh):
                return tokens
            if tokens:
                errors.append(f"{path}: credentials are expired and contain no refresh token")
        detail = f" Unreadable sources: {'; '.join(errors)}" if errors else ""
        raise AuthError(
            "No ChatGPT OAuth credentials found. Connect OpenCode to ChatGPT with `opencode auth login`, "
            "or set CLAUDE_CODEX_AUTH_FILE to an OAuth credential file." + detail
        )

    @asynccontextmanager
    async def _cache_lock(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_path.with_suffix(self.cache_path.suffix + ".lock")
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            await asyncio.to_thread(fcntl.flock, descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    async def get(self, *, force_refresh: bool = False, stale_access: str | None = None) -> Tokens:
        async with self._lock:
            if self._tokens is None:
                self._tokens = self.load()
            if self._tokens.fresh and (
                not force_refresh or (stale_access is not None and self._tokens.access != stale_access)
            ):
                return self._tokens
            previous_access = self._tokens.access
            async with self._cache_lock():
                reloaded = None
                if self.cache_path.is_file():
                    with suppress(OSError, ValueError, json.JSONDecodeError):
                        reloaded = _from_cache(self.cache_path)
                if reloaded is None:
                    try:
                        reloaded = self.load()
                    except AuthError:
                        reloaded = self._tokens
                if reloaded.fresh and (
                    not force_refresh
                    or reloaded.access != (stale_access if stale_access is not None else previous_access)
                ):
                    self._tokens = reloaded
                    return reloaded
                self._tokens = await self._refresh(reloaded)
                await asyncio.to_thread(self._save, self._tokens)
                return self._tokens

    async def _refresh(self, current: Tokens) -> Tokens:
        if not current.refresh:
            raise AuthError(f"OAuth credentials from {current.source} expired and contain no refresh token")
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=30)
        try:
            response = await client.post(
                f"{self.issuer}/oauth/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": current.refresh,
                    "client_id": CLIENT_ID,
                },
            )
            if response.is_error:
                raise AuthError(f"ChatGPT OAuth refresh failed: HTTP {response.status_code}")
            data = response.json()
        finally:
            if owns_client:
                await client.aclose()
        access = str(data["access_token"])
        refresh = str(data.get("refresh_token") or current.refresh)
        expires = int(time.time() * 1000) + int(data.get("expires_in") or 3600) * 1000
        return Tokens(
            access=access,
            refresh=refresh,
            expires=expires,
            account_id=_account_id(str(data.get("id_token") or ""), access) or current.account_id,
            source="claude-codex-cache",
        )

    def _save(self, tokens: Tokens) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "oauth",
            "access": tokens.access,
            "refresh": tokens.refresh,
            "expires": tokens.expires,
            "account_id": tokens.account_id,
        }
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.cache_path.name}.", suffix=".tmp", dir=self.cache_path.parent
        )
        temporary = Path(name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w") as output:
                json.dump(payload, output)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self.cache_path)
            self.cache_path.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)
