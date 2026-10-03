"""Private, bounded checkpoint summaries; never stores the source transcript."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

MAX_ENTRY_BYTES = 262_144
MAX_AGE_SECONDS = 86_400


class CheckpointCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.enabled = os.environ.get("CLAUDE_CODEX_CHECKPOINT_CACHE", "1") != "0"

    def entry(self, scope: Any, request: dict[str, Any]) -> tuple[Path, str]:
        namespace = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
        content = {
            k: request[k] for k in
            ("model", "instructions", "input", "reasoning", "tools", "tool_choice", "text")
            if k in request
        }
        key = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return self.root / namespace / (key + ".json"), key

    def get(self, scope: Any, request: dict[str, Any]) -> str | None:
        if not self.enabled:
            return None
        path, key = self.entry(scope, request)
        try:
            for directory in (self.root, path.parent):
                info = directory.lstat()
                if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                    return None
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as f:
                info = os.fstat(f.fileno())
                if (
                    not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or info.st_size > MAX_ENTRY_BYTES
                    or time.time() - info.st_mtime > MAX_AGE_SECONDS
                ):
                    return None
                data = json.load(f)
            if data.get("version") == 1 and data.get("key") == key:
                summary = data.get("summary")
                if isinstance(summary, str) and summary.strip():
                    return summary
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return None

    def put(self, scope: Any, request: dict[str, Any], summary: str) -> None:
        if not self.enabled or not summary.strip():
            return
        path, key = self.entry(scope, request)
        encoded = json.dumps({"version": 1, "key": key, "summary": summary}, ensure_ascii=False).encode()
        if len(encoded) > MAX_ENTRY_BYTES:
            return
        temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
        created = False
        try:
            for directory in (self.root, path.parent):
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                info = directory.lstat()
                if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                    return
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            created = True
            with os.fdopen(fd, "wb") as f:
                f.write(encoded)
            os.replace(temporary, path)
        except OSError:
            pass
        finally:
            if created:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
