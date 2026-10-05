from __future__ import annotations

import json

from claude_codex import auxiliary_log


def test_disabled_log_creates_no_files(monkeypatch, tmp_path):
    path = tmp_path / "private/requests.jsonl"
    monkeypatch.setattr(auxiliary_log, "AUXILIARY_LOG_PATH", path)
    monkeypatch.setenv("CLAUDE_CODEX_AUXILIARY_LOG", "0")
    auxiliary_log.write_record({"request_id": "one"}, "request", payload={"input": []})
    assert not path.parent.exists()


def test_log_rotation_retains_complete_records(monkeypatch, tmp_path):
    path = tmp_path / "private/requests.jsonl"
    monkeypatch.setattr(auxiliary_log, "AUXILIARY_LOG_PATH", path)
    monkeypatch.setattr(auxiliary_log, "MAX_LOG_BYTES", 1)
    monkeypatch.delenv("CLAUDE_CODEX_AUXILIARY_LOG", raising=False)
    auxiliary_log.write_record({"request_id": "one"}, "request", payload={"input": ["prompt"]})
    auxiliary_log.write_record({"request_id": "one"}, "response_event", data={"delta": "deny"})
    assert json.loads(path.read_text())["data"]["delta"] == "deny"
    backup = path.with_suffix(".jsonl.1")
    assert json.loads(backup.read_text())["payload"]["input"] == ["prompt"]
    assert backup.stat().st_mode & 0o777 == 0o600


def test_symlink_log_is_not_followed(monkeypatch, tmp_path, capsys):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    path = root / "requests.jsonl"
    target = tmp_path / "other"
    target.write_text("untouched")
    path.symlink_to(target)
    monkeypatch.setattr(auxiliary_log, "AUXILIARY_LOG_PATH", path)
    monkeypatch.delenv("CLAUDE_CODEX_AUXILIARY_LOG", raising=False)
    auxiliary_log.write_record({"request_id": "one"}, "request", payload={"input": ["prompt"]})
    assert target.read_text() == "untouched"
    assert "error=write_failed" in capsys.readouterr().out
