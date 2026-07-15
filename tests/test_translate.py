from __future__ import annotations

from claude_codex.translate import (
    AnthropicStream,
    to_responses_request,
    validate_messages_request,
)


def test_accepts_system_role_message() -> None:
    # Claude Code sends a `system` turn inside `messages`; it must not be rejected.
    payload = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "be brief"},
        ]
    }
    assert validate_messages_request(payload) is payload
    lowered = to_responses_request(payload, model="gpt-5.4", reasoning_effort="medium")
    system_items = [item for item in lowered["input"] if item.get("role") == "system"]
    assert system_items and system_items[0]["content"][0]["text"] == "be brief"


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


def test_maps_output_limit_and_parallel_tool_choice() -> None:
    result = to_responses_request(
        {
            "max_tokens": 17,
            "messages": [],
            "tools": [{"name": "Bash"}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        },
        model="gpt-5.4",
        reasoning_effort="medium",
    )

    # the Codex backend rejects max_output_tokens, so it must not be forwarded
    assert "max_output_tokens" not in result
    assert result["parallel_tool_calls"] is False


def test_prompt_cache_key_is_stable_and_bounded() -> None:
    long_key = "session-" + "x" * 100

    first = to_responses_request(
        {"messages": []},
        model="gpt-5.4",
        reasoning_effort="medium",
        prompt_cache_key=long_key,
    )
    second = to_responses_request(
        {"messages": []},
        model="gpt-5.4",
        reasoning_effort="medium",
        prompt_cache_key=long_key,
    )

    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    assert len(first["prompt_cache_key"]) == 64


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


def test_translates_cached_input_usage() -> None:
    stream = AnthropicStream("claude-sonnet")

    events = stream.feed(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 4096,
                    "input_tokens_details": {"cached_tokens": 3072},
                    "output_tokens": 2,
                }
            },
        },
    )

    message_delta = next(data for name, data in events if name == "message_delta")
    expected = {
        "input_tokens": 1024,
        "cache_read_input_tokens": 3072,
        "output_tokens": 2,
    }
    assert message_delta["usage"] == expected
    assert stream.response()["usage"] == expected


def test_failed_stream_is_terminal() -> None:
    stream = AnthropicStream("claude-sonnet")

    events = stream.feed(
        "response.failed",
        {"type": "response.failed", "response": {"error": {"message": "boom"}}},
    )

    assert [name for name, _ in events] == ["error"]
    assert stream.finish() == []


def test_nonstream_preserves_max_tokens_stop_reason() -> None:
    stream = AnthropicStream("claude-sonnet")

    stream.feed(
        "response.incomplete",
        {
            "type": "response.incomplete",
            "response": {
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
        },
    )

    assert stream.response()["stop_reason"] == "max_tokens"


def test_streaming_mode_does_not_retain_content() -> None:
    stream = AnthropicStream("claude-sonnet", retain_content=False)

    stream.feed(
        "response.output_text.delta",
        {"type": "response.output_text.delta", "output_index": 0, "delta": "hello"},
    )
    stream.feed(
        "response.function_call_arguments.delta",
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "delta": '{"path":"x"}',
        },
    )

    assert stream.blocks[0].text == ""
    assert stream.blocks[1].arguments == ""
    assert stream.blocks[1].arguments_seen is True


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
