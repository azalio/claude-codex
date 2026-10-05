"""Private request/response diagnostics for auxiliary inference."""
from __future__ import annotations

import fcntl
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AUXILIARY_LOG_PATH = Path.home() / ".local/state/claude-codex/auxiliary/requests.jsonl"
MAX_LOG_BYTES = 10 * 1024 * 1024
REQUEST_FIELDS = {
    "model", "instructions", "input", "stream", "store", "reasoning",
    "tools", "tool_choice", "parallel_tool_calls", "text", "max_output_tokens",
}


def request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    request = {key: value for key, value in payload.items() if key in REQUEST_FIELDS}
    if "reasoning" in request:
        request["reasoning"] = {"effort": request["reasoning"]["effort"]}
    return request


def _private_file(path: Path) -> int:
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        os.close(fd)
        raise OSError("Auxiliary log must be a private regular file")
    return fd


def write_record(metadata: dict[str, Any], event: str, **fields: Any) -> None:
    if os.environ.get("CLAUDE_CODEX_AUXILIARY_LOG", "1") == "0":
        return
    record = {"time": datetime.now(UTC).isoformat(), "event": event, **metadata, **fields}
    encoded = (json.dumps(record, ensure_ascii=True) + "\n").encode()
    path = AUXILIARY_LOG_PATH
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = path.parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise OSError("Auxiliary log directory must be private")
        with os.fdopen(_private_file(path.with_suffix(".lock")), "ab") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            fd = _private_file(path)
            if os.fstat(fd).st_size + len(encoded) > MAX_LOG_BYTES:
                os.close(fd)
                os.replace(path, path.with_suffix(".jsonl.1"))
                fd = _private_file(path)
            with os.fdopen(fd, "ab") as out:
                out.write(encoded)
    except OSError:
        # Diagnostic failures must not alter classifier decisions or inference.
        print("auxiliary_log error=write_failed", flush=True)
