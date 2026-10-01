# claude-codex

Run the normal Claude Code CLI while using a ChatGPT Codex subscription as the model backend.

The command starts a local Anthropic-compatible proxy, points `ANTHROPIC_BASE_URL` at it, and then
executes the regular `claude` binary with the supplied arguments. Claude Code keeps its UI, slash
commands, skills, hooks, MCP servers, and tools. The proxy translates Anthropic Messages and SSE to
the Codex Responses protocol.

The launcher also pins proxy routing in session-level `--settings`, because Claude Code's
`settings.json` environment values can override the inherited process environment. Existing
user/project settings remain enabled; explicit `--settings` JSON or files are merged with the
proxy overrides without modifying the source files. Merged settings are passed through a private
temporary file (mode `0600`), keeping their contents out of process arguments; the file is removed
when Claude exits, including on launch errors or Ctrl+C. Backend selectors for Bedrock, Vertex, and
Foundry and Mantle are disabled for this session.

## Gateway compatibility and auto mode

The launcher uses the documented fallback for a gateway whose upstream cannot perform Anthropic
server-side classifier review: it pins `CLAUDE_CODE_AUTO_MODE_SERVER=0` for the Claude session.
Auto mode keeps its client-side classifier checks. Those checks become ordinary Codex inference
requests and use the subscription's normal quota; this bridge does not provide Anthropic's
no-charge server checks. See [classifier request charges](https://code.claude.com/docs/en/auto-mode-classifier-billing).

Because the bridge converts system blocks to Responses instructions, it also pins
`CLAUDE_CODE_ATTRIBUTION_HEADER=0` at the client. Gateway hint headers are enabled, experimental
Anthropic capabilities are disabled, and `ENABLE_TOOL_SEARCH=false` keeps MCP tools in ordinary
function-tool form. Existing permissions, hooks, and persistent settings files remain untouched.

The proxy supports native session/agent routing, streaming with keep-alive pings, HTTP error and
retry controls, JSON-schema structured outputs, strict function tools, effort translation, and
`/v1/models` discovery. Discovery remains opt-in with `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`
and reports the configured Codex backend. Token counts remain character-based estimates.

See the [compatibility matrix](docs/gateway-compatibility.md) for supported behavior and limits,
mapped to the [Claude Code gateway guide](https://code.claude.com/docs/en/llm-gateway-protocol).

## Authentication

No OpenAI API key is used. Credentials are loaded in this order:

1. `CLAUDE_CODEX_AUTH_FILE`
2. `~/.config/claude-codex/auth.json`
3. OpenCode OAuth credentials in `~/.local/share/opencode/auth.json`
4. Codex CLI credentials in `$CODEX_HOME/auth.json` or `~/.codex/auth.json`

If OpenCode is already connected to ChatGPT, the third source works immediately. Refreshed
credentials are copied to the private `claude-codex` cache with mode `0600`; OpenCode and Codex
credential files are never modified.

## TLS certificates

Upstream HTTPS requests and OAuth token refreshes trust certifi's public CA bundle plus the
Python/OpenSSL default CA store, including organization certificates installed there. The public
bundle remains available when Python has no default CA paths. Certificate and hostname
verification remain enabled. To replace the default trust with a custom CA bundle or OpenSSL
certificate directory, set `SSL_CERT_FILE` or `SSL_CERT_DIR` (`SSL_CERT_FILE` takes precedence):

```bash
SSL_CERT_FILE=/path/to/ca-bundle.pem claude-codex
```

## Install

```bash
git clone https://github.com/azalio/claude-codex.git
cd claude-codex
./install.sh
```

The installer creates or repairs the repo-local `.venv`, reinstalls the locked environment from the
current repository path, and atomically updates `~/bin/claude-codex`. Reinstalling the whole
environment rewrites absolute shebangs for every generated command after a repository move. The
installed command points to a repo-owned shell wrapper rather than a generated virtualenv
entrypoint. If the repository is moved again, rerun `./install.sh`.

## Use

```bash
claude-codex
claude-codex -p "Explain this repository"
claude-codex --continue
```

Configuration:

```bash
CLAUDE_CODEX_MODEL=gpt-6.1-sol claude-codex
CLAUDE_CODEX_REASONING=xhigh claude-codex
CLAUDE_CODEX_LOG_MAX_BYTES=10485760 claude-codex
CLAUDE_CODEX_COMPACT_AT=900000 claude-codex
# Optional experimental native endpoint; local compact is the safe default.
CLAUDE_CODEX_REMOTE_COMPACT=1 claude-codex
```

For GPT-6.1 backends, the launcher uses `ANTHROPIC_MODEL=claude-opus-5-5[1m]` to give
Claude Code a 1,000,000-token client context window. It also upgrades an inherited bare
`claude-opus-5-5` identity; other explicit model identities are preserved. This changes the
client's context accounting, not the actual Codex backend model. A forwarded `--model` option
or an in-session `/model` selection overrides this identity. `CLAUDE_CODE_DISABLE_1M_CONTEXT=1`
limits the client to 200,000 tokens even with `[1m]`; leave it unset for the larger window.

`proxy.log` is rotated to one `proxy.log.1` backup when it reaches 10 MiB. Set
`CLAUDE_CODEX_LOG_MAX_BYTES` to a positive byte limit to override that threshold.

The default compaction threshold is 900,000 input tokens, leaving 150,000 tokens below the
[OpenRouter-listed 1,050,000-token context window](https://openrouter.ai/openai/gpt-6.1-sol).
The Codex subscription endpoint's serving limit has not been verified separately.
When a completed upstream turn reports at least 900,000 input tokens, the next turn in that
native Claude session is compacted locally: the proxy creates a concise handoff summary through
ordinary `/responses`, retains recent user messages, and continues without exposing an internal
error to Claude Code. The proxy stores replacement history only in memory, advances that branch's
Codex context window, and forwards the following turn with the compacted prefix. Parallel histories
that share one launcher session are tracked separately by their message-prefix branch. Set
`CLAUDE_CODEX_COMPACT_AT=0` to disable this behavior or a positive token threshold to change it.
`/responses/compact` is available only as the opt-in experiment `CLAUDE_CODEX_REMOTE_COMPACT=1`:
its output contract currently differs from the proxy's Anthropic translation. Compaction events are
written to `proxy.log` as `codex_compact` lines.

The backend URL is an internal ChatGPT Codex contract also used by OpenCode. It can change without
the compatibility guarantees of the public OpenAI Platform API.
