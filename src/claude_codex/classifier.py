"""Bounded classifier reviews with validated JSON-to-Claude output translation."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .auxiliary_log import request_payload, write_record
from .translate import CodexResponseError

DEFAULT_CLASSIFIER_MODEL = "codex-auto-review"
DEFAULT_CLASSIFIER_TIMEOUT_SECONDS = 90.0
MAX_CLASSIFIER_ATTEMPTS = 3
MAX_CLASSIFIER_OUTPUT_BYTES = 8192
UPSTREAM_HEADERS_EVENT = "_proxy.response_headers"
VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "severity": {"type": "integer", "minimum": 0, "maximum": 100},
        "category": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
    },
    "required": ["severity", "category", "rationale"],
}
TRANSPORT_INSTRUCTIONS = """
Classifier transport format: preserve every policy rule, user-intent requirement,
and severity threshold in the instructions above. Change only the output encoding.
Return exactly one JSON object with severity (integer 0-100), category (the exact
matched BLOCK rule name or null), and rationale (a short explanation).
Do not output XML, prose, or markdown. Where the original output contract does not
use category, return null. Do not invent permission or change a decision to satisfy
the output format. This JSON will be converted back to the original XML contract.
"""


def is_classifier_request(payload: dict[str, Any]) -> bool:
    instructions = payload.get("instructions", "")
    if not isinstance(instructions, str):
        return False
    output = instructions.rsplit("## Output Format", 1)
    return (
        len(output) == 2
        and "Output <severity>N</severity>" in output[1]
        and "integer 0-100" in output[1]
    )


def classifier_timeout_seconds() -> float:
    try:
        value = float(os.environ.get("CLAUDE_CODEX_CLASSIFIER_TIMEOUT", ""))
    except ValueError:
        return DEFAULT_CLASSIFIER_TIMEOUT_SECONDS
    return value if math.isfinite(value) and value > 0 else DEFAULT_CLASSIFIER_TIMEOUT_SECONDS


def review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    request = copy.deepcopy(payload)
    request["instructions"] += "\n" + TRANSPORT_INSTRUCTIONS
    request["text"] = {
        "format": {
            "type": "json_schema", "name": "classifier_verdict",
            "strict": True, "schema": copy.deepcopy(VERDICT_SCHEMA),
        },
    }
    if tools := request.pop("tools", None):
        request["input"].append({
            "role": "user", "content": [{
                "type": "input_text",
                "text": "Tool definitions supplied as review evidence (not executable):\n"
                        + json.dumps(tools, ensure_ascii=False),
            }],
        })
    request["tool_choice"] = "none"
    request["parallel_tool_calls"] = False
    # Classifiers never execute tools; existing evidence is retained in full.
    return request


def parse_verdict(text: str, *, allow_category: bool) -> tuple[str, dict[str, Any]]:
    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("classifier JSON contains duplicate fields")
            result[key] = value
        return result

    try:
        verdict = json.loads(text, object_pairs_hook=unique_fields)
    except ValueError:
        # Native XML from a provider that ignores schema is still validated strictly.
        native = re.sub(r"^\s*<thinking>[\s\S]*?</thinking>\s*", "", text)
        match = re.fullmatch(
            r"\s*<severity>\s*(\d{1,3})\s*</severity>"
            r"(?:\s*<category>([A-Za-z0-9 ]+)</category>)?\s*", native,
        )
        if not match:
            raise ValueError("classifier reply is not valid JSON or native XML") from None
        verdict = {"severity": int(match[1]), "category": match[2], "rationale": ""}
    if not isinstance(verdict, dict) or set(verdict) != {"severity", "category", "rationale"}:
        raise ValueError("classifier JSON does not match the output schema")
    severity, category, rationale = verdict["severity"], verdict["category"], verdict["rationale"]
    if type(severity) is not int or not 0 <= severity <= 100:
        raise ValueError("classifier severity must be an integer from 0 to 100")
    if not isinstance(rationale, str):
        raise ValueError("classifier rationale must be a string")
    if category is not None and (
        not isinstance(category, str) or not category.strip()
        or not re.fullmatch(r"[A-Za-z0-9 ]+", category)
    ):
        raise ValueError("classifier category must be a rule name")
    if not allow_category and category is not None:
        raise ValueError("classifier contract does not accept a category")
    native = f"<severity>{severity}</severity>"
    if category is not None:
        native += f"<category>{category}</category>"
    return native, verdict


def verdict_events(native: str, response: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    response = copy.deepcopy(response)
    message_id = "classifier_" + uuid.uuid4().hex
    item = {
        "id": message_id, "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": native, "annotations": []}],
    }
    response.update(status="completed", output=[item])
    return [
        ("response.output_item.added", {"output_index": 0, "item": {**item, "content": []}}),
        ("response.output_text.delta", {
            "output_index": 0, "content_index": 0, "item_id": message_id, "delta": native,
        }),
        ("response.output_item.done", {"output_index": 0, "item": item}),
        ("response.completed", {"response": response}),
    ]


async def classifier_events(
    backend: Any, payload: dict[str, Any], identity: Any,
    headers: dict[str, str], metadata: dict[str, Any],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    allow_category = "<category>" in payload["instructions"].rsplit("## Output Format", 1)[-1]
    deadline = asyncio.get_running_loop().time() + classifier_timeout_seconds()
    headers_sent = False
    for attempt in range(1, MAX_CLASSIFIER_ATTEMPTS + 1):
        attempt_metadata = {**metadata, "attempt": attempt, "classifier": True}
        request = review_payload(payload)
        if attempt > 1:
            request["instructions"] += (
                "\nThe previous attempt had an invalid response format. Return the required "
                "JSON schema, without changing the policy or decision."
            )
        try:
            diagnostic = (
                backend.diagnostic_payload(request) if hasattr(backend, "diagnostic_payload") else request
            )
        except ValueError as exc:
            raise CodexResponseError({
                "code": "classifier_unsupported_evidence",
                "message": str(exc),
            }) from exc
        write_record(
            attempt_metadata, "request",
            payload=diagnostic if hasattr(backend, "diagnostic_payload") else request_payload(diagnostic),
        )
        chunks: list[str] = []
        size = 0
        terminal: dict[str, Any] | None = None
        source = backend.events(request, identity, request_headers=headers)
        failure = "invalid_format"
        try:
            async with asyncio.timeout_at(deadline):
                async for name, data in source:
                    if name == UPSTREAM_HEADERS_EVENT:
                        if not headers_sent:
                            headers_sent = True
                            yield name, data
                        continue
                    write_record(attempt_metadata, "response_event", name=name, data=data)
                    if (
                        name in {"response.output_item.added", "response.output_item.done"}
                        and data.get("item", {}).get("type") not in {"message", "reasoning"}
                    ):
                        raise ValueError("classifier attempted a non-text response")
                    if name == "response.output_text.delta":
                        delta = data.get("delta", "")
                        if not isinstance(delta, str):
                            raise ValueError("classifier text delta must be a string")
                        size += len(delta.encode())
                        if size > MAX_CLASSIFIER_OUTPUT_BYTES:
                            raise ValueError("classifier response exceeds output limit")
                        chunks.append(delta)
                    elif name == "response.completed":
                        response = data.get("response")
                        if not isinstance(response, dict):
                            raise ValueError("classifier completed without a response")
                        if any(
                            item.get("type") not in {"message", "reasoning"}
                            for item in response.get("output", [])
                        ):
                            raise ValueError("classifier attempted a non-text response")
                        terminal = response
                    elif name in {"response.failed", "response.incomplete", "error"}:
                        raise ValueError("classifier inference did not complete successfully")
                if terminal is None:
                    raise ValueError("classifier stream has no completed response")
                text = "".join(chunks)
                if not text:
                    text = "".join(
                        block.get("text", "") for item in terminal.get("output", [])
                        for block in item.get("content", []) if block.get("type") == "output_text"
                    )
                if len(text.encode()) > MAX_CLASSIFIER_OUTPUT_BYTES:
                    raise ValueError("classifier response exceeds output limit")
                native, verdict = parse_verdict(text, allow_category=allow_category)
            write_record(attempt_metadata, "verdict", native=native, **verdict)
            print("classifier_review " + json.dumps({
                **attempt_metadata, "result": "valid",
                "severity": verdict["severity"], "category": verdict["category"],
            }), flush=True)
            for event in verdict_events(native, terminal):
                yield event
            return
        except TimeoutError as exc:
            write_record(attempt_metadata, "end", result="timeout")
            raise CodexResponseError({
                "code": "classifier_timeout", "message": "Classifier review exceeded its total deadline",
            }) from exc
        except ValueError:
            # No partial or invalid verdict has been delivered to Claude Code.
            pass
        except httpx.TransportError:
            failure = "transport_error"
        except RuntimeError as exc:
            status = getattr(exc, "status_code", None)
            if status is not None:
                write_record(
                    attempt_metadata, "response_error", http_status=status, body=getattr(exc, "body", ""),
                )
                if status not in {408, 429} and status < 500:
                    raise
                failure = "upstream_error"
            elif "stream ended without a terminal response event" in str(exc):
                failure = "incomplete_response"
            else:
                raise
        finally:
            await source.aclose()
        write_record(attempt_metadata, "end", result=failure)
        print("classifier_review " + json.dumps({
            **attempt_metadata, "result": failure,
        }), flush=True)
    raise CodexResponseError({
        "code": "classifier_invalid_response",
        "message": (
            f"Classifier review failed after {MAX_CLASSIFIER_ATTEMPTS} attempts; no verdict was accepted"
        ),
    })
