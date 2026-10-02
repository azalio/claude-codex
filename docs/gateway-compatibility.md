# Claude Code gateway compatibility

This bridge exposes Anthropic Messages to Claude Code and translates requests to a non-Anthropic
Codex Responses upstream. The gateway guide assigns schema bridging to gateways with a
non-Anthropic upstream; copying Anthropic fields into Responses would not implement their semantics.

References, checked on 2026-10-01:

- [Claude Code gateway compatibility guide](https://code.claude.com/docs/en/llm-gateway-protocol)
- [Auto mode classifier request charges](https://code.claude.com/docs/en/auto-mode-classifier-billing)
- [Claude Code environment variables](https://code.claude.com/docs/en/env-vars)

## Auto mode

Codex does not implement Anthropic's `safeguards` / `safeguard_results` contract. The launcher pins
`CLAUDE_CODE_AUTO_MODE_SERVER=0` in both the subprocess environment and its private session settings
file. This is the documented fallback when a gateway cannot supply server-side classifier checks.
It does not disable auto mode or grant permission for any action: Claude Code's client-side
classifier still checks actions when the user chooses auto mode. Classifier calls are ordinary
Codex inference and use normal subscription quota, rather than Anthropic's no-charge server checks.

Direct clients bypassing the launcher receive HTTP 400 when they send `safeguards`. The bridge
never invents a safety verdict or silently removes a requested safety check. The fallback variable
is temporary upstream API and may require adjustment in future Claude Code releases. The launcher
does not change persistent permission settings or force a permission mode.

## Compatibility matrix

| Gateway behavior | Bridge implementation | Limit |
| --- | --- | --- |
| Messages endpoint | `POST /v1/messages`, including `?beta=true` | Only Anthropic Messages ingress; provider-specific formats are not exposed |
| Token counts | `/v1/messages/count_tokens` returns a character-based estimate | Not an exact Codex tokenizer count; includes system/messages, not tool schemas |
| Startup warming | `HEAD /api/hello` returns 204 without inference | Best-effort probe only |
| Session/agent routing | Native `x-claude-code-session-id` takes priority; separate thread/cache/compaction state per agent and request class | Legacy session headers remain fallbacks; absent request-class hints default to `main`; agent IDs do not identify users |
| Gateway hints | Open `x-claude-code-*` family reaches Codex unchanged | Not every scheduling hint necessarily has Codex semantics |
| Credentials and Anthropic headers | Local credentials are replaced by ChatGPT OAuth; Anthropic version/beta headers are consumed at ingress | Arbitrary custom headers, local keys, and Anthropic beta semantics are not copied into Responses |
| System attribution | Launcher pins `CLAUDE_CODE_ATTRIBUTION_HEADER=0` before reshaping system blocks | Direct clients should set the same variable; the proxy does not strip prompt text |
| Streaming | Text/tool deltas stream as Anthropic SSE, with body-gap pings and final message events | Missing terminal events and malformed SSE are errors, never synthetic success |
| HTTP errors and control headers | Status, complete error message/details (context overflow gets Claude recovery wording), `x-should-retry`, and open `anthropic-ratelimit-unified-*` family are retained; retry dates become integer seconds | Codex errors need an Anthropic envelope; cookies/hop-by-hop headers are excluded; Anthropic plan limits are never fabricated |
| Function tools | Names, call IDs, arguments, tool choice, parallel-call policy, and `strict` are translated | Provider tools are rejected; advisor errors retain `Input tag` wording for Claude Code recovery |
| Tool search | Launcher pins `ENABLE_TOOL_SEARCH=false`; ordinary MCP function tools remain available | `defer_loading`, `tool_reference`, and provider tool-search tools return explicit errors |
| Effort and thinking | The launcher pins Codex effort to `medium` by default; `CLAUDE_CODEX_REASONING` selects another level | The gateway override wins over client effort, including implicit `max`. Standalone proxy requests without an override honor explicit client effort (`max` maps to `xhigh`) and default to `medium` when omitted; Anthropic thinking budgets/signatures are not forwarded or fabricated |
| Structured outputs | JSON-schema `output_config.format` maps to Responses `text.format`, with strict validation requested | Other formats and Anthropic task budgets are unsupported |
| Context management | Proxy compaction checks previous usage and current input/instructions/tool estimates at 180k by default; auxiliary/classifier requests are excluded; large native compaction histories use bounded segment summaries; Anthropic `context_management` returns HTTP 400 | Native recovery has a four-minute total deadline and aborts on failed/incomplete segments; native final requests omit checkpointed tool schemas and use compact-only instructions if needed; oversized image histories and remaining checkpoint/format text return explicit errors; checkpoint summaries are lossy and do not inherit user JSON-output schemas |
| Prompt caching | Stable identities and prefix text feed Codex caching; reported usage maps back to Anthropic usage | Anthropic cache markers/TTLs have no Codex equivalent; cache creation is never invented |
| Model discovery | `/v1/models?limit=1000` directly reports one configured backend with a compatible alias | Discovery remains client opt-in; selection does not change `CLAUDE_CODEX_MODEL` |

New headers in the routing-hint family require no allowlist update. New Anthropic body capabilities
still require a Responses translation or explicit rejection: this bridge cannot promise transparent
support for future Anthropic features. Unsupported known safety, context-management, and provider
tool contracts are rejected before inference.

## Verification

Tests exercise launcher invocation and the HTTP APIs. Regression coverage includes settings
privacy/cleanup, native header precedence, agent isolation, future hint/limit headers, HTTP error
preservation in JSON and streaming modes, retry dates, structured outputs, effort overrides,
capability rejection, body-gap pings, malformed/truncated streams, bounded native history recovery,
compaction deadlines/cancellation, and existing compaction/cache behavior. Controlled upstream
transports establish local protocol behavior, not no-charge classifier eligibility, model availability, subscription context limits, or Anthropic safety review
on Codex.
