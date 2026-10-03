from __future__ import annotations

import os
import time

from claude_codex.checkpoint import MAX_AGE_SECONDS, CheckpointCache


def test_cache_is_private_and_invalidated_by_scope_source_or_model(tmp_path):
    cache = CheckpointCache(tmp_path / "cache")
    scope = ("endpoint", "installation", "session", "main")
    request = {"model": "gpt-6-luna", "instructions": "summarize", "input": ["private-source"]}
    cache.put(scope, request, "finished summary")
    assert cache.get(scope, request) == "finished summary"
    assert cache.get((*scope[:-1], "auxiliary"), request) is None
    assert cache.get(scope, {**request, "model": "gpt-6.1-sol"}) is None
    assert cache.get(scope, {**request, "input": ["changed source"]}) is None
    path, _ = cache.entry(scope, request)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert "private-source" not in path.read_text()
    os.utime(path, (time.time() - MAX_AGE_SECONDS - 1,) * 2)
    assert cache.get(scope, request) is None


def test_cache_rejects_symlinks_corruption_and_inaccessible_directory(tmp_path):
    cache = CheckpointCache(tmp_path / "cache")
    request = {"model": "model", "input": ["source"]}
    cache.put("scope", request, "summary")
    path, _ = cache.entry("scope", request)
    path.write_text("corrupt")
    assert cache.get("scope", request) is None
    path.unlink()
    target = tmp_path / "untrusted"
    target.write_text("untrusted summary")
    path.symlink_to(target)
    assert cache.get("scope", request) is None
    broken = tmp_path / "not-a-directory"
    broken.write_text("keep")
    CheckpointCache(broken).put("scope", request, "summary")
    assert broken.read_text() == "keep"


def test_cache_kill_switch_disables_reads_and_writes(tmp_path, monkeypatch):
    cache = CheckpointCache(tmp_path / "cache")
    request = {"model": "model", "input": ["source"]}
    cache.put("scope", request, "summary")
    monkeypatch.setenv("CLAUDE_CODEX_CHECKPOINT_CACHE", "0")
    disabled = CheckpointCache(cache.root)
    assert disabled.get("scope", request) is None
    disabled.put("scope", request, "replacement")
    assert cache.get("scope", request) == "summary"
