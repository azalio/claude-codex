from __future__ import annotations

import uuid
from pathlib import Path

from claude_codex.proxy import _resolve_installation_id


def test_installation_id_is_persistent_and_private(tmp_path: Path) -> None:
    path = tmp_path / "installation_id"

    first = _resolve_installation_id(path)
    second = _resolve_installation_id(path)

    assert first == second
    assert uuid.UUID(first)
    assert path.read_text().strip() == first
    assert path.stat().st_mode & 0o777 == 0o600


def test_installation_id_replaces_invalid_value(tmp_path: Path) -> None:
    path = tmp_path / "installation_id"
    path.write_text("invalid")

    installation_id = _resolve_installation_id(path)

    assert uuid.UUID(installation_id)
    assert path.read_text().strip() == installation_id
