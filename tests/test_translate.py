from __future__ import annotations

from claude_codex.translate import AnthropicStream, to_responses_request


def test_lowers_anthropic_tool_loop() -> None:
    payload = {
        "system": [{"type": "text", "text": "Be precise", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "README.md"}],
            },
        ],
        "tools": [
            {
                "name": "Bash",
                "description": "Run a command",
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        ],
    }

    result = to_responses_request(payload, model="gpt-5.4", reasoning_effort="medium")

    assert result["instructions"] == "Be precise"
    assert result["input"][1] == {
        "type": "function_call",
        "call_id": "toolu_1",
        "name": "Bash",
        "arguments": '{"command":"ls"}',
    }
    assert result["input"][2] == {
        "type": "function_call_output",
        "call_id": "toolu_1",
        "output": "README.md",
    }
    assert result["tools"][0]["name"] == "Bash"
    assert result["store"] is False


def test_translates_text_stream() -> None:
    stream = AnthropicStream("claude-sonnet")
    events = []
    events += stream.feed("response.created", {"type": "response.created", "response": {"id": "resp_123"}})
    events += stream.feed(
        "response.output_text.delta",
        {"type": "response.output_text.delta", "output_index": 0, "delta": "hello"},
    )
    events += stream.feed(
        "response.completed",
        {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 10, "output_tokens": 2}},
        },
    )

    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert stream.response()["content"] == [{"type": "text", "text": "hello"}]


def test_translates_function_call_stream() -> None:
    stream = AnthropicStream("claude-sonnet")
    stream.feed(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "call_id": "call_1", "name": "Read"},
        },
    )
    delta = stream.feed(
        "response.function_call_arguments.delta",
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"file_path":"x"}'},
    )
    stream.feed(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "call_id": "call_1", "name": "Read"},
        },
    )
    stream.finish()

    assert delta[0][1]["delta"]["type"] == "input_json_delta"
    assert stream.response()["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "x"}}
    ]
    assert stream.response()["stop_reason"] == "tool_use"
