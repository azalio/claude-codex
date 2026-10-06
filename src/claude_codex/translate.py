from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

DEFAULT_INSTRUCTIONS = """You are a coding agent running as the model backend for Claude Code.
Follow the supplied system instructions and use the supplied tools when appropriate.
Do not invent tool results. Continue until the user's request is genuinely handled."""


class CodexResponseError(RuntimeError):
    """Сохраняет детали SSE-ошибки и обозначает переполнение для Claude recovery."""

    def __init__(self, error: Any) -> None:
        self.error = dict(error) if isinstance(error, dict) else {}
        message = str(self.error.get("message") or "Codex request failed")
        overflow = self.error.get("code") == "context_length_exceeded" or message.startswith(
            "Your input exceeds the context window"
        )
        self.status_code = 400 if overflow else 502
        if overflow and not message.lower().startswith("prompt is too long"):
            message = f"prompt is too long: {message}"
        self.error.update(type="invalid_request_error" if overflow else "api_error", message=message)
        super().__init__(message)

    def error_body(self) -> dict[str, Any]:
        return {"type": "error", "error": self.error.copy()}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "\n".join(str(block.get("text", "")) for block in value if block.get("type") == "text")


def _image(block: dict[str, Any]) -> dict[str, str] | None:
    source = block.get("source") or {}
    if source.get("type") == "base64" and source.get("data"):
        media = source.get("media_type") or "image/png"
        return {"type": "input_image", "image_url": f"data:{media};base64,{source['data']}"}
    if source.get("type") == "url" and source.get("url"):
        return {"type": "input_image", "image_url": str(source["url"])}
    return None


def _tool_output(content: Any) -> str | list[dict[str, str]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    output: list[dict[str, str]] = []
    for block in content:
        if block.get("type") == "text":
            output.append({"type": "input_text", "text": str(block.get("text", ""))})
        elif block.get("type") == "image":
            image = _image(block)
            if image:
                output.append(image)
    if len(output) == 1 and output[0]["type"] == "input_text":
        return output[0]["text"]
    return output or ""


def _flush_pending(result: list[dict[str, Any]], role: str, pending: list[dict[str, str]]) -> None:
    if not pending:
        return
    result.append({"role": role, "content": list(pending)})
    pending.clear()


def _lower_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        if isinstance(content, str):
            item_type = "output_text" if role == "assistant" else "input_text"
            result.append({"role": role, "content": [{"type": item_type, "text": content}]})
            continue

        pending: list[dict[str, str]] = []

        for block in content:
            kind = block.get("type")
            if kind == "text":
                pending.append(
                    {
                        "type": "output_text" if role == "assistant" else "input_text",
                        "text": str(block.get("text", "")),
                    }
                )
                if block.get("citations"):
                    pending.append({
                        "type": "output_text" if role == "assistant" else "input_text",
                        "text": "Previous citation metadata:\n"
                                + json.dumps(block["citations"], ensure_ascii=False),
                    })
            elif kind in {"server_tool_use", "web_search_tool_result"}:
                pending.append({
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": "Previous server search evidence (not an executable tool):\n"
                            + json.dumps(block, ensure_ascii=False),
                })
            elif kind == "image" and role == "user":
                image = _image(block)
                if image:
                    pending.append(image)
            elif kind == "tool_use" and role == "assistant":
                _flush_pending(result, role, pending)
                result.append(
                    {
                        "type": "function_call",
                        "call_id": str(block.get("id") or f"call_{uuid.uuid4().hex}"),
                        "name": str(block.get("name") or "unknown_tool"),
                        "arguments": json.dumps(
                            block.get("input") or {}, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
            elif kind == "tool_result":
                _flush_pending(result, role, pending)
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(block.get("tool_use_id") or ""),
                        "output": _tool_output(block.get("content", "")),
                    }
                )
        _flush_pending(result, role, pending)
    return result


def _tool_choice(value: Any) -> Any:
    if not isinstance(value, dict):
        return "auto"
    kind = value.get("type")
    if kind in {"auto", "none"}:
        return kind
    if kind == "any":
        return "required"
    if kind == "tool":
        return {"type": "function", "name": value.get("name")}
    return "auto"


def validate_messages_request(payload: Any, *, require_messages: bool = True) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    if "safeguards" in payload:
        raise ValueError(
            "Codex does not implement Anthropic safeguards or safeguard_results; "
            "start Claude Code with CLAUDE_CODE_AUTO_MODE_SERVER=0 to use client-side checks"
        )
    if "context_management" in payload:
        raise ValueError(
            "Codex does not implement Anthropic context_management; "
            "set CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 (proxy compaction remains available)"
        )
    system = payload.get("system")
    if system is not None:
        if not isinstance(system, (str, list)):
            raise ValueError("system must be a string or an array")
        if isinstance(system, list) and not all(isinstance(block, dict) for block in system):
            raise ValueError("each system block must be an object")
    messages = payload.get("messages")
    if require_messages and not isinstance(messages, list):
        raise ValueError("messages must be an array")
    if messages is not None:
        if not isinstance(messages, list):
            raise ValueError("messages must be an array")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("each message must be an object")
            # Claude Code emits `system`/`developer` turns inside `messages`
            # (not just top-level `system`), and the Codex Responses API
            # accepts them, so mirror that set instead of only user/assistant.
            if message.get("role") not in {"user", "assistant", "system", "developer"}:
                raise ValueError("message role must be user, assistant, system, or developer")
            content = message.get("content", "")
            if not isinstance(content, (str, list)):
                raise ValueError("message content must be a string or an array")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        raise ValueError("each content block must be an object")
                    if block.get("type") == "tool_reference":
                        raise ValueError(
                            "Codex does not implement tool_reference; set ENABLE_TOOL_SEARCH=false"
                        )
                    if block.get("type") == "image" and not isinstance(block.get("source"), dict):
                        raise ValueError("image source must be an object")
                    if block.get("type") == "tool_result":
                        result_content = block.get("content", "")
                        if not isinstance(result_content, (str, list)):
                            raise ValueError("tool result content must be a string or an array")
                        if isinstance(result_content, list) and not all(
                            isinstance(item, dict) for item in result_content
                        ):
                            raise ValueError("each tool result content block must be an object")
                        if isinstance(result_content, list) and any(
                            item.get("type") == "tool_reference" for item in result_content
                        ):
                            raise ValueError(
                                "Codex does not implement tool_reference; set ENABLE_TOOL_SEARCH=false"
                            )
    tools = payload.get("tools")
    if tools is not None and (
        not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools)
    ):
        raise ValueError("tools must be an array of objects")
    for tool in tools or []:
        kind = tool.get("type")
        if kind == "web_search_20250305":
            if tool.get("name") != "web_search":
                raise ValueError("web_search_20250305 must be named web_search")
            unsupported = set(tool) - {"type", "name", "allowed_domains", "user_location", "cache_control"}
            if unsupported:
                raise ValueError("Unsupported web_search options: " + ", ".join(sorted(unsupported)))
            domains = tool.get("allowed_domains")
            if domains is not None and (
                not isinstance(domains, list)
                or not all(isinstance(domain, str) and domain.strip() for domain in domains)
            ):
                raise ValueError("web_search allowed_domains must be an array of nonempty strings")
            location = tool.get("user_location")
            if location is not None and (
                not isinstance(location, dict) or location.get("type") != "approximate"
                or set(location) - {"type", "country", "region", "city", "timezone"}
                or not all(isinstance(value, str) for value in location.values())
            ):
                raise ValueError("web_search user_location must use supported approximate fields")
            if sum(t.get("name") == "web_search" for t in tools or []) != 1:
                raise ValueError("web_search must have a unique tool name")
            choice = payload.get("tool_choice") or {}
            if (isinstance(choice, dict) and choice.get("type") == "tool"
                and choice.get("name") == "web_search" and len(tools or []) != 1):
                raise ValueError("Forced web_search requires it to be the only tool")
            continue
        if kind not in {None, "custom"}:
            raise ValueError(f"Input tag '{kind}' in tools is not supported by the Codex bridge")
        if tool.get("defer_loading"):
            raise ValueError("Codex does not implement defer_loading; set ENABLE_TOOL_SEARCH=false")
        if "strict" in tool and not isinstance(tool["strict"], bool):
            raise ValueError("tool strict must be a boolean")
    output_config = payload.get("output_config")
    if output_config is not None:
        if not isinstance(output_config, dict):
            raise ValueError("output_config must be an object")
        if "effort" in output_config and output_config["effort"] not in {
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("output_config.effort must be low, medium, high, xhigh, or max")
        if "format" in output_config:
            output_format = output_config["format"]
            if not isinstance(output_format, dict) or output_format.get("type") != "json_schema":
                raise ValueError("output_config.format must use json_schema")
            if not isinstance(output_format.get("schema"), dict):
                raise ValueError("output_config.format.schema must be an object")
        if "task_budget" in output_config:
            raise ValueError("Codex does not implement output_config.task_budget")
    max_tokens = payload.get("max_tokens")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer")
    if "stream" in payload and not isinstance(payload["stream"], bool):
        raise ValueError("stream must be a boolean")
    return payload


def _bounded_cache_key(value: str) -> str:
    if len(value) <= 64:
        return value
    return hashlib.sha256(value.encode()).hexdigest()


def to_responses_request(
    payload: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    prompt_cache_key: str | None = None,
) -> dict[str, Any]:
    system = _text(payload.get("system"))
    tools = [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description") or "",
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            "strict": tool.get("strict", False),
        }
        for tool in payload.get("tools") or []
        if tool.get("name") and tool.get("type") != "web_search_20250305"
    ]
    search_tools = [tool for tool in payload.get("tools") or [] if tool.get("type") == "web_search_20250305"]
    for tool in search_tools:
        native: dict[str, Any] = {"type": "web_search", "external_web_access": True}
        if tool.get("allowed_domains") is not None:
            native["filters"] = {"allowed_domains": tool["allowed_domains"]}
        if tool.get("user_location") is not None:
            native["user_location"] = tool["user_location"]
        tools.append(native)
    request: dict[str, Any] = {
        "model": model,
        "instructions": system or DEFAULT_INSTRUCTIONS,
        "input": _lower_messages(payload.get("messages") or []),
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": reasoning_effort, "summary": "auto"},
    }
    if prompt_cache_key:
        request["prompt_cache_key"] = _bounded_cache_key(prompt_cache_key)
    output_format = (payload.get("output_config") or {}).get("format")
    if output_format:
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": "response",
                "schema": output_format["schema"],
                "strict": True,
            }
        }
    # NB: the Codex `/responses` backend rejects `max_output_tokens`
    # ("Unsupported parameter") and manages output length itself, so the
    # Anthropic `max_tokens` is validated on the way in but not forwarded.
    if tools:
        tool_choice = payload.get("tool_choice")
        request["tools"] = tools
        request["tool_choice"] = _tool_choice(tool_choice)
        if search_tools:
            request["include"].append("web_search_call.action.sources")
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool" and (
                tool_choice.get("name") == "web_search"
            ):
                if len(tools) != 1:
                    raise ValueError("Forced web_search requires it to be the only tool")
                request["tool_choice"] = "required"
        disable_parallel = (
            bool(tool_choice.get("disable_parallel_tool_use")) if isinstance(tool_choice, dict) else False
        )
        request["parallel_tool_calls"] = not disable_parallel
    return request


def encode_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


@dataclass(slots=True)
class Block:
    index: int
    kind: str
    item_id: str
    name: str | None = None
    text: str = ""
    arguments: str = ""
    arguments_seen: bool = False
    open: bool = True
    content: dict[str, Any] | None = None


@dataclass(slots=True)
class AnthropicStream:
    requested_model: str
    retain_content: bool = True
    response_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    started: bool = False
    completed: bool = False
    failed: bool = False
    failure_message: str | None = None
    failure: CodexResponseError | None = None
    stop_reason: str | None = None
    has_tool: bool = False
    search_pending: bool = False
    search_events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    by_output: dict[int, Block] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def _usage(self) -> dict[str, int]:
        cache_read = max(0, int(getattr(self, "cache_read_input_tokens", 0)))
        cache_creation = max(0, int(getattr(self, "cache_creation_input_tokens", 0)))
        usage = {
            "input_tokens": max(0, self.input_tokens - cache_read - cache_creation),
            "output_tokens": self.output_tokens,
        }
        if cache_read:
            usage["cache_read_input_tokens"] = cache_read
        if cache_creation:
            usage["cache_creation_input_tokens"] = cache_creation
        return usage

    def _start(self) -> list[tuple[str, dict[str, Any]]]:
        if self.started:
            return []
        self.started = True
        return [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self.response_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.requested_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                    },
                },
            )
        ]

    def _block(
        self, output_index: int, kind: str, item: dict[str, Any] | None = None
    ) -> tuple[Block, list[tuple[str, dict[str, Any]]]]:
        existing = self.by_output.get(output_index)
        if existing:
            if not existing.open or existing.kind != kind:
                raise RuntimeError("Codex stream references a closed or incompatible content block")
            return existing, []
        item = item or {}
        block = Block(
            index=len(self.blocks),
            kind=kind,
            item_id=str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"),
            name=item.get("name"),
        )
        self.blocks.append(block)
        self.by_output[output_index] = block
        if kind == "tool":
            self.has_tool = True
            content = {
                "type": "tool_use",
                "id": block.item_id,
                "name": block.name or "unknown_tool",
                "input": {},
            }
        else:
            content = {"type": "text", "text": ""}
        return block, [
            (
                "content_block_start",
                {"type": "content_block_start", "index": block.index, "content_block": content},
            )
        ]

    def _server_block(self, content: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        index = len(self.blocks)
        self.blocks.append(Block(index, "server", str(content.get("id") or ""), open=False, content=content))
        return [
            ("content_block_start", {
                "type": "content_block_start", "index": index, "content_block": content,
            }),
            ("content_block_stop", {"type": "content_block_stop", "index": index}),
        ]

    def _search_output(self, response: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        items: dict[int, dict[str, Any]] = {}
        citations: dict[str, str] = {}
        for _name, data in self.search_events:
            item = data.get("item")
            if isinstance(item, dict):
                index = int(data.get("output_index") or 0)
                items[index] = {**items.get(index, {}), **item}
            annotation = data.get("annotation") or {}
            if annotation.get("type") == "url_citation" and annotation.get("url"):
                citations[annotation["url"]] = annotation.get("title") or annotation["url"]
        for index, item in enumerate(response.get("output") or []):
            items[index] = {**items.get(index, {}), **item}
        for item in items.values():
            for part in item.get("content") or []:
                for annotation in part.get("annotations") or []:
                    if annotation.get("type") == "url_citation" and annotation.get("url"):
                        citations[annotation["url"]] = annotation.get("title") or annotation["url"]
        searches = [item for item in items.values() if item.get("type") == "web_search_call"]
        output: list[tuple[str, dict[str, Any]]] = []
        for item in searches:
            action = item.get("action") or {}
            if action.get("type") == "search":
                query = action.get("query")
                queries = action.get("queries") or []
                arguments = {"query": query or (queries[0] if queries else "")}
                if len(queries) > 1:
                    arguments["queries"] = queries
            else:
                # Preserve open_page/find_in_page honestly, rather than inventing a query.
                arguments = dict(action)
                arguments.pop("sources", None)
            call_id = str(item.get("id") or "srvtoolu_" + uuid.uuid4().hex)
            output.extend(self._server_block({
                "type": "server_tool_use", "id": call_id, "name": "web_search", "input": arguments,
            }))
            sources: dict[str, str] = {}
            for source in action.get("sources") or []:
                if isinstance(source, dict) and source.get("url"):
                    url = source["url"]
                    sources[url] = citations.get(url) or source.get("title") or url
            if len(searches) == 1:
                sources.update(citations)
            results: Any = [{"type": "web_search_result", "url": url, "title": title}
                            for url, title in sources.items()]
            if item.get("status") != "completed":
                results = {"type": "web_search_tool_result_error", "error_code": "unavailable"}
            output.extend(self._server_block({
                "type": "web_search_tool_result", "tool_use_id": call_id, "content": results,
            }))
        # Reuse ordinary delta handling after reconciling hosted results and citations.
        text_indices: set[int] = set()
        for name, data in self.search_events:
            if name == "response.output_text.delta":
                text_indices.add(int(data.get("output_index") or 0))
            if name in {"response.output_text.delta", "response.function_call_arguments.delta"} or (
                name in {"response.output_item.added", "response.output_item.done"}
                and (data.get("item") or {}).get("type") == "function_call"
            ):
                output.extend(self._feed(name, data))
        for index, item in items.items():
            if item.get("type") == "message" and index not in text_indices and index not in self.by_output:
                text = "".join(part.get("text", "") for part in item.get("content") or []
                               if part.get("type") == "output_text")
                if text:
                    output.extend(self._feed("response.output_text.delta", {
                        "output_index": index, "delta": text,
                    }))
            elif item.get("type") == "function_call":
                block = self.by_output.get(index)
                if block is None or block.open:
                    output.extend(self._feed("response.output_item.done", {
                        "output_index": index, "item": item,
                    }))
        if citations:
            index = max([*items, *self.by_output], default=-1) + 1
            links = "\n\nSources:\n" + "\n".join(f"- [{title}]({url})" for url, title in citations.items())
            output.extend(self._feed("response.output_text.delta", {"output_index": index, "delta": links}))
        return output

    def feed(self, event: str, data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        event = str(data.get("type") or event)
        if self.completed:
            return []
        if (data.get("item") or {}).get("type") == "web_search_call":
            self.search_pending = True
        terminal = event in {"response.completed", "response.incomplete"}
        if terminal and any(item.get("type") == "web_search_call"
                            for item in (data.get("response") or {}).get("output") or []):
            self.search_pending = True
        if event in {"error", "response.failed"}:
            self.search_events.clear()
            return self._feed(event, data)
        if self.search_pending:
            if terminal:
                if event == "response.incomplete":
                    self.search_events.clear()
                    return self._feed("error", {
                        "code": "incomplete_search", "message": "Search response incomplete",
                    })
                output = self._start() + self._search_output(data.get("response") or {})
                self.search_events.clear()
                self.search_pending = False
                output.extend(self._feed(event, data))
                return output
            self.search_events.append((event, data))
            return self._start()
        return self._feed(event, data)

    def _feed(self, event: str, data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        event = str(data.get("type") or event)
        output: list[tuple[str, dict[str, Any]]] = []
        if event == "response.created":
            response = data.get("response") or {}
            self.response_id = str(response.get("id") or self.response_id).replace("resp_", "msg_", 1)
            output.extend(self._start())
            return output
        if event in {"error", "response.failed"}:
            response = data.get("response")
            error = data.get("error") or (
                response.get("error") if isinstance(response, dict) else None
            )
            # Responses error event может содержать code/message прямо на верхнем уровне.
            if not error and event == "error":
                error = data
            self.failed = True
            self.completed = True
            self.failure = CodexResponseError(error)
            self.failure_message = str(self.failure)
            return [("error", self.failure.error_body())]
        if self.completed:
            return []
        output.extend(self._start())
        output_index = int(data.get("output_index") or 0)
        if event == "response.output_item.added":
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                _, opened = self._block(output_index, "tool", item)
                output.extend(opened)
            return output
        if event == "response.output_text.delta":
            block, opened = self._block(output_index, "text")
            output.extend(opened)
            delta = str(data.get("delta") or "")
            if self.retain_content:
                block.text += delta
            if delta:
                output.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block.index,
                            "delta": {"type": "text_delta", "text": delta},
                        },
                    )
                )
            return output
        if event == "response.function_call_arguments.delta":
            item = data.get("item") or {}
            block, opened = self._block(output_index, "tool", item)
            output.extend(opened)
            delta = str(data.get("delta") or "")
            if delta:
                block.arguments_seen = True
                if self.retain_content:
                    block.arguments += delta
                output.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block.index,
                            "delta": {"type": "input_json_delta", "partial_json": delta},
                        },
                    )
                )
            return output
        if event == "response.output_item.done":
            item = data.get("item") or {}
            block = self.by_output.get(output_index)
            if item.get("type") == "function_call":
                block, opened = self._block(output_index, "tool", item)
                output.extend(opened)
                arguments = str(item.get("arguments") or "")
                if arguments and not block.arguments_seen:
                    block.arguments_seen = True
                    if self.retain_content:
                        block.arguments = arguments
                    output.append(
                        (
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block.index,
                                "delta": {"type": "input_json_delta", "partial_json": arguments},
                            },
                        )
                    )
            if block and block.open:
                block.open = False
                output.append(("content_block_stop", {"type": "content_block_stop", "index": block.index}))
            return output
        if event in {"response.completed", "response.incomplete"}:
            response = data.get("response") or {}
            usage = response.get("usage") or {}
            self.input_tokens = int(usage.get("input_tokens") or 0)
            self.output_tokens = int(usage.get("output_tokens") or 0)
            details = usage.get("input_tokens_details") or {}
            if isinstance(details, dict):
                self.cache_read_input_tokens = int(details.get("cached_tokens") or 0)
                self.cache_creation_input_tokens = int(details.get("cache_write_tokens") or 0)
            incomplete = response.get("incomplete_details")
            if event == "response.incomplete" and not incomplete:
                incomplete = {"reason": "max_output_tokens"}
            output.extend(self.finish(incomplete))
        return output

    def finish(self, incomplete: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
        if self.completed:
            return []
        if self.search_pending:
            self.search_events.clear()
            return self._feed("error", {
                "code": "incomplete_search",
                "message": "Search stream ended without a terminal response event",
            })
        output = self._start()
        for block in self.blocks:
            if block.open:
                block.open = False
                output.append(("content_block_stop", {"type": "content_block_stop", "index": block.index}))
        reason = "tool_use" if self.has_tool else "end_turn"
        if incomplete and incomplete.get("reason") == "max_output_tokens":
            reason = "max_tokens"
        self.stop_reason = reason
        output.extend(
            [
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": reason, "stop_sequence": None},
                        "usage": self._usage(),
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        )
        self.completed = True
        return output

    def response(self) -> dict[str, Any]:
        if self.failed:
            raise self.failure or RuntimeError(self.failure_message or "Codex request failed")
        content: list[dict[str, Any]] = []
        for block in self.blocks:
            if block.content is not None:
                content.append(block.content)
            elif block.kind == "text":
                content.append({"type": "text", "text": block.text})
            else:
                try:
                    arguments = json.loads(block.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {"_raw": block.arguments}
                content.append(
                    {"type": "tool_use", "id": block.item_id, "name": block.name, "input": arguments}
                )
        return {
            "id": self.response_id,
            "type": "message",
            "role": "assistant",
            "model": self.requested_model,
            "content": content,
            "stop_reason": self.stop_reason or ("tool_use" if self.has_tool else "end_turn"),
            "stop_sequence": None,
            "usage": self._usage(),
        }


def estimate_tokens(payload: dict[str, Any]) -> int:
    serialized = json.dumps(payload.get("system", ""), ensure_ascii=False) + json.dumps(
        payload.get("messages", []), ensure_ascii=False
    )
    return max(1, (len(serialized) + 3) // 4)
