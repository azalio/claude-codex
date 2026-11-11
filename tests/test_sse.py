from __future__ import annotations

import asyncio
import json
from contextlib import aclosing

import httpx
import pytest

from claude_codex.proxy import _sse


class ChunkedBody(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


async def parse(chunks: list[bytes]):
    response = httpx.Response(200, stream=ChunkedBody(chunks))
    try:
        return [event async for event in _sse(response)]
    finally:
        await response.aclose()


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("eol", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
async def test_sse_preserves_unicode_separators(separator, eol, chunk_size):
    payload = {"type": "response.output_text.delta", "delta": f"лево{separator}право"}
    wire = ("data: " + json.dumps(payload, ensure_ascii=False) + eol * 2).encode()
    chunks = [wire[start:start + chunk_size] for start in range(0, len(wire), chunk_size)]

    assert await parse(chunks) == [(payload["type"], payload)]


@pytest.mark.parametrize("eol", ["\n", "\r\n", "\r"])
async def test_sse_preserves_events_at_every_byte_boundary(eol):
    payloads = [
        {"type": "response.created", "response": {"id": "test"}},
        {"type": "response.output_text.delta", "delta": "a\u0085б\u2028中\u2029z"},
        {"type": "response.completed", "response": {}},
    ]
    wire = "".join("data: " + json.dumps(item, ensure_ascii=False) + eol * 2 for item in payloads).encode()
    expected = [(item["type"], item) for item in payloads]
    for split in range(len(wire) + 1):
        assert await parse([wire[:split], b"", wire[split:]]) == expected, split


@pytest.mark.parametrize("chunk_size", [1, 4096])
async def test_sse_keeps_multiline_data_comments_done_and_eof_behavior(chunk_size):
    wire = (
        ': keepalive\r\nevent: chunk\r\ndata: {"delta":\r\ndata: "a\u2028b\u2029c"}\r\n\r\n'
        'data: [DONE]\r\n\r\nevent: tail\ndata: {"delta":"last"}'
    ).encode()
    chunks = [wire[start:start + chunk_size] for start in range(0, len(wire), chunk_size)]

    assert await parse(chunks) == [("chunk", {"delta": "a\u2028b\u2029c"}), ("tail", {"delta": "last"})]


@pytest.mark.parametrize("eol", [b"\n", b"\r\n", b"\r"])
async def test_sse_emits_complete_event_without_waiting_for_next_chunk(eol):
    class PausedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"type":"response.created"}' + eol * 2
            await asyncio.Event().wait()

    response = httpx.Response(200, stream=PausedBody())
    try:
        async with aclosing(_sse(response)) as events:
            assert await asyncio.wait_for(anext(events), timeout=0.2) == (
                "response.created", {"type": "response.created"},
            )
    finally:
        await response.aclose()


@pytest.mark.parametrize("data, message", [
    ('{"delta":"unterminated', "invalid JSON event"),
    ('{"delta":"literal\x0bcontrol"}', "invalid JSON event"),
    ('{"delta":"literal\x1econtrol"}', "invalid JSON event"),
    ('[]', "must be a JSON object"),
])
async def test_sse_still_rejects_invalid_payloads(data, message):
    with pytest.raises(RuntimeError, match=message):
        await parse([f"data: {data}\n\n".encode()])
