from __future__ import annotations

import json

import pytest

from claude_codex import proxy


def test_multimodal_parts_preserve_images_text_and_source_order(monkeypatch):
    monkeypatch.setattr(proxy, "COMPACTION_CHUNK_TOKENS", 512)
    images = [
        {"type": "input_image", "image_url": "data:image/png;base64," + "A" * 4000},
        {"type": "input_image", "image_url": "https://example.invalid/screenshot.png"},
    ]
    source = {"model": "test", "instructions": "system evidence", "input": [
        {"type": "function_call_output", "call_id": "capture-1", "output": [
            {"type": "input_text", "text": "before截图" * 1000}, images[0],
            {"type": "input_text", "text": "between"}, images[1],
            {"type": "input_text", "text": "after"},
        ]},
    ], "tools": [{"type": "function", "name": "capture"}]}
    parts = proxy._multimodal_checkpoint_parts(source)
    observed = [block for part in parts for block in part["input"][0]["content"]
                if block["type"] == "input_image"]
    assert observed == images
    text = "".join(block["text"] for part in parts for block in part["input"][0]["content"]
                   if block["type"] == "input_text")
    assert "A" * 100 not in text
    assert "capture-1" in text and "system evidence" in text and "capture" in text
    assert "between" in text and "after" in text
    for part in parts:
        assert len(json.dumps(part, ensure_ascii=False).encode()) <= proxy.COMPACTION_IMAGE_WIRE_BYTES
        assert sum(block["type"] == "input_image" for block in part["input"][0]["content"]) <= 1


def test_multimodal_oversized_single_image_fails_before_dispatch(monkeypatch):
    monkeypatch.setattr(proxy, "COMPACTION_IMAGE_WIRE_BYTES", 3000)
    with pytest.raises(proxy.BackendError) as error:
        proxy._multimodal_checkpoint_parts({"model": "test", "input": [
            {"role": "user", "content": [{"type": "input_image", "image_url": "A" * 4000}]}]})
    assert "compaction_image_budget" in error.value.body


def test_visual_checkpoint_cache_tracks_image_without_persisting_source(tmp_path):
    from claude_codex.checkpoint import CheckpointCache

    cache = CheckpointCache(tmp_path / "cache")
    request = {"model": "vision", "input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64,PRIVATE_IMAGE"}]}]}
    cache.put("scope", request, "Observed deployment table")
    assert cache.get("scope", request) == "Observed deployment table"
    path, _ = cache.entry("scope", request)
    assert "PRIVATE_IMAGE" not in path.read_text()
    request["input"][0]["content"][0]["image_url"] += "CHANGED"
    assert cache.get("scope", request) is None


def test_multimodal_segment_ceiling_and_metadata_budget(monkeypatch):
    image = {"type": "input_image", "image_url": "https://example.invalid/image"}
    with pytest.raises(proxy.BackendError, match="128 checkpoint"):
        proxy._multimodal_checkpoint_parts({"input": [{"role": "user", "content": [image] * 129}]})
    monkeypatch.setattr(proxy, "COMPACTION_CHUNK_TOKENS", 512)
    with pytest.raises(proxy.BackendError, match="metadata exceeds text budget"):
        proxy._multimodal_checkpoint_parts({"input": [{"type": "function_call_output",
            "call_id": "x" * 3000, "output": [image]}]})


def test_image_checkpoint_has_adjacent_legend_and_global_segment_labels(monkeypatch):
    monkeypatch.setattr(proxy, "COMPACTION_CHUNK_TOKENS", 1024)
    parts = proxy._multimodal_checkpoint_parts({"input": [{"type": "function_call_output",
        "call_id": "chart", "output": [
            {"type": "input_text", "text": "Red means failed; blue means healthy. " + "x" * 6000},
            {"type": "input_image", "image_url": "https://example.invalid/chart"},
            {"type": "input_text", "text": "Units: thousands per minute."},
        ]}]})
    for index, request in enumerate(parts):
        assert request["input"][0]["content"][0]["text"].startswith(f"Transcript segment {index + 1}\n")
    visual, = [request for request in parts if len(request["input"][0]["content"]) == 2]
    caption = visual["input"][0]["content"][0]["text"]
    assert "Red means failed; blue means healthy." in caption
    assert "Units: thousands per minute." in caption
