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


def test_translates_structured_output_and_strict_function_tools() -> None:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    payload = {
        "messages": [],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
        "tools": [{"name": "Check", "input_schema": schema, "strict": True}],
    }
    validate_messages_request(payload)
    result = to_responses_request(payload, model="gpt-6.1-sol", reasoning_effort="high")
    assert result["text"]["format"] == {
        "type": "json_schema",
        "name": "response",
        "schema": schema,
        "strict": True,
    }
    assert result["tools"][0]["strict"] is True


def test_native_web_search_request() -> None:
    payload = {"messages": [], "tools": [{
        "type": "web_search_20250305", "name": "web_search", "max_uses": 8,
        "allowed_domains": ["example.com"],
        "user_location": {"type": "approximate", "country": "US"},
    }], "tool_choice": {"type": "tool", "name": "web_search"}}
    validate_messages_request(payload)
    request = to_responses_request(payload, model="gpt-5.4", reasoning_effort="medium")
    assert request["tools"] == [{"type": "web_search", "external_web_access": True,
                                "filters": {"allowed_domains": ["example.com"]},
                                "user_location": {"type": "approximate", "country": "US"}}]
    assert request["tool_choice"] == "required"
    assert "web_search_call.action.sources" in request["include"]


def test_web_search_rejects_unenforceable_options() -> None:
    import pytest

    for option in ({"blocked_domains": ["example.com"]},
                   {"allowed_domains": "example.com"}, {"user_location": {"type": "exact"}}):
        with pytest.raises(ValueError):
            validate_messages_request({"messages": [], "tools": [
                {"type": "web_search_20250305", "name": "web_search", **option}]})
    with pytest.raises(ValueError):
        validate_messages_request({"messages": [], "tools": [
            {"type": "web_search_20250305", "name": "web_search"}, {"name": "Bash"}],
            "tool_choice": {"type": "tool", "name": "web_search"}})


def test_search_results_citations_and_replay() -> None:
    for retain in (True, False):
        stream = AnthropicStream("test", retain_content=retain)
        search = {"type": "web_search_call", "id": "ws_1", "status": "completed",
                  "action": {"type": "search", "query": "example",
                             "sources": [{"type": "url", "url": "https://example.com"}]}}
        message = {"type": "message", "id": "m_1", "content": [{
            "type": "output_text", "text": "Answer", "annotations": [{
                "type": "url_citation", "url": "https://example.com", "title": "Example"}]}]}
        events = stream.feed("response.output_item.added", {"output_index": 0, "item": search})
        events += stream.feed("response.output_text.delta", {"output_index": 1, "delta": "Answer"})
        events += stream.feed("response.completed", {"response": {"output": [search, message]}})
        starts = [data["content_block"] for name, data in events if name == "content_block_start"]
        assert starts[0]["type"] == "server_tool_use"
        assert starts[0]["input"] == {"query": "example"}
        assert starts[1]["type"] == "web_search_tool_result"
        assert starts[1]["tool_use_id"] == starts[0]["id"]
        assert starts[1]["content"] == [{"type": "web_search_result", "url": "https://example.com",
                                        "title": "Example"}]
        assert stream.stop_reason == "end_turn"
        assert "https://example.com" in str(events)
        if retain:
            response = stream.response()
            lowered = to_responses_request({"messages": [{"role": "assistant",
                "content": response["content"]}]}, model="test", reasoning_effort="medium")
            assert all(item.get("type") != "function_call" for item in lowered["input"])
            assert "https://example.com" in str(lowered["input"])
            assert "example" in str(lowered["input"])


def test_multiple_search_sources_are_not_guessed_from_citations() -> None:
    stream = AnthropicStream("test")
    items = [{"type": "web_search_call", "id": f"ws_{index}", "status": "completed",
              "action": {"type": "search", "query": f"query {index}",
                         "sources": [{"url": f"https://example.com/{index}"}]}}
             for index in range(2)]
    items.append({"type": "message", "content": [{"type": "output_text", "text": "answer",
        "annotations": [{"type": "url_citation", "url": "https://other.com", "title": "Other"},
                        {"type": "url_citation", "url": "https://example.com/0", "title": "Zero"}]}]})
    stream.feed("response.completed", {"response": {"output": items}})
    content = stream.response()["content"]
    assert content[1]["content"] == [{"type": "web_search_result", "url": "https://example.com/0",
                                      "title": "Zero"}]
    assert content[3]["content"] == [{"type": "web_search_result", "url": "https://example.com/1",
                                      "title": "https://example.com/1"}]
    assert "https://other.com" in content[-1]["text"]


def test_search_empty_failed_and_page_actions() -> None:
    for status, action in (("completed", {"type": "search", "query": "empty"}),
                           ("failed", {"type": "search", "query": "bad"}),
                           ("completed", {"type": "open_page", "url": "https://example.com"}),
                           ("completed", {"type": "find_in_page", "url": "https://example.com",
                                          "pattern": "needle"})):
        stream = AnthropicStream("test")
        stream.feed("response.completed", {"response": {"output": [{
            "type": "web_search_call", "id": "ws", "status": status, "action": action}]}})
        content = stream.response()["content"]
        if status == "failed":
            assert content[1]["content"]["error_code"] == "unavailable"
        else:
            assert content[1]["content"] == []
        if action["type"] != "search":
            assert content[0]["input"] == action
        assert stream.stop_reason == "end_turn"


def test_search_mixed_function_and_late_annotations() -> None:
    for retain in (True, False):
        stream = AnthropicStream("test", retain_content=retain)
        search = {"type": "web_search_call", "id": "ws", "status": "completed",
                  "action": {"type": "search", "queries": ["one", "two"],
                             "sources": [{"url": "https://example.com"}]}}
        tool = {"type": "function_call", "id": "fc", "call_id": "call", "name": "Bash",
                "arguments": '{"command":"ls"}'}
        events = stream.feed("response.output_item.added", {"item": search})
        events += stream.feed("response.output_item.done", {"item": search})
        events += stream.feed("response.output_text.delta", {"output_index": 1, "delta": "Answer"})
        events += stream.feed("response.output_text.annotation.added", {"output_index": 1,
            "annotation": {"type": "url_citation", "url": "https://example.com", "title": "Late"}})
        events += stream.feed("response.output_item.added", {"output_index": 2, "item": tool})
        events += stream.feed("response.output_item.done", {"output_index": 2, "item": tool})
        events += stream.feed("response.completed", {"response": {"output": [search,
            {"type": "message", "content": [{"type": "output_text", "text": "Answer"}]}, tool]}})
        starts = [data["content_block"] for name, data in events if name == "content_block_start"]
        assert sum(block["type"] == "server_tool_use" for block in starts) == 1
        assert starts[0]["input"] == {"query": "one", "queries": ["one", "two"]}
        assert starts[1]["content"][0]["title"] == "Late"
        assert stream.stop_reason == "tool_use"
        deltas = [data["delta"] for name, data in events if name == "content_block_delta"]
        assert sum(delta.get("text") == "Answer" for delta in deltas) == 1
        assert any(delta.get("partial_json") == '{"command":"ls"}' for delta in deltas)


def test_search_terminal_failure_does_not_emit_results() -> None:
    for terminal in ("response.failed", "response.incomplete"):
        stream = AnthropicStream("test")
        stream.feed("response.output_item.added", {"item": {"type": "web_search_call", "id": "ws"}})
        events = stream.feed(terminal, {"response": {"error": {"message": "failed"}}})
        assert events[0][0] == "error"
        assert stream.failed
        assert not stream.blocks


def test_search_versions_names_and_duplicate_definitions_rejected() -> None:
    import pytest

    search = {"type": "web_search_20250305", "name": "web_search"}
    for tools in ([{**search, "type": "web_search_20260209"}],
                  [{**search, "name": "other"}], [search, search], [search, {"name": "web_search"}]):
        with pytest.raises(ValueError):
            validate_messages_request({"messages": [], "tools": tools})


def test_search_with_forced_client_function_keeps_function_choice() -> None:
    payload = {"messages": [], "tools": [
        {"type": "web_search_20250305", "name": "web_search"}, {"name": "Bash"}],
        "tool_choice": {"type": "tool", "name": "Bash", "disable_parallel_tool_use": True}}
    validate_messages_request(payload)
    request = to_responses_request(payload, model="test", reasoning_effort="medium")
    assert request["tool_choice"] == {"type": "function", "name": "Bash"}
    assert request["parallel_tool_calls"] is False


def test_search_after_preamble_and_existing_function_reconciles_terminal() -> None:
    for retain in (True, False):
        stream = AnthropicStream("test", retain_content=retain)
        preamble = {"type": "message", "content": [{"type": "output_text", "text": "Looking"}]}
        stream.feed("response.output_text.delta", {"output_index": 0, "delta": "Looking"})
        stream.feed("response.output_item.done", {"output_index": 0, "item": preamble})
        tool = {"type": "function_call", "call_id": "call", "name": "Read"}
        stream.feed("response.output_item.added", {"output_index": 1, "item": tool})
        search = {"type": "web_search_call", "id": "ws", "status": "completed",
                  "action": {"type": "search", "query": "query"}}
        stream.feed("response.output_item.added", {"output_index": 2, "item": search})
        events = stream.feed("response.completed", {"response": {"output": [
            preamble, {**tool, "arguments": '{"file_path":"x"}'}, search]}})
        assert not any(data.get("delta", {}).get("text") == "Looking" for _, data in events)
        assert any(data.get("delta", {}).get("partial_json") == '{"file_path":"x"}' for _, data in events)


def test_search_without_terminal_fails_instead_of_empty_success() -> None:
    stream = AnthropicStream("test")
    stream.feed("response.output_item.added", {"item": {"type": "web_search_call", "id": "ws"}})
    events = stream.finish()
    assert events[0][0] == "error"
    assert stream.failed


def test_web_search_max_uses_validates_positive_integers() -> None:
    import pytest

    for value in (True, False, None, "8", 8.0, 0, -1):
        with pytest.raises(ValueError, match="max_uses must be a positive integer"):
            validate_messages_request({"messages": [], "tools": [{
                "type": "web_search_20250305", "name": "web_search", "max_uses": value}]})


def test_web_search_max_uses_is_explicitly_unenforced(caplog) -> None:
    payload = {"messages": [], "tools": [{
        "type": "web_search_20250305", "name": "web_search", "max_uses": 1}]}
    validate_messages_request(payload)
    request = to_responses_request(payload, model="test", reasoning_effort="medium")
    assert "max_tool_calls" not in request
    assert "max_uses" not in request["tools"][0]
    assert "web_search_max_uses_unenforced requested=1" in caplog.text
    stream = AnthropicStream("test")
    stream.feed("response.completed", {"response": {"output": [{
        "type": "web_search_call", "id": f"ws_{index}", "status": "completed",
        "action": {"type": "search", "query": f"query {index}"}} for index in range(2)]}})
    assert sum(block["type"] == "server_tool_use" for block in stream.response()["content"]) == 2
    caplog.clear()
    payload["tools"][0].pop("max_uses")
    to_responses_request(payload, model="test", reasoning_effort="medium")
    assert "web_search_max_uses_unenforced" not in caplog.text
