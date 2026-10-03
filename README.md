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

Reasoning defaults to `medium`. The launcher pins this default in the gateway so
Claude Code's implicit effort does not raise it. Set `CLAUDE_CODEX_REASONING`
to choose another level; it takes precedence over the client effort. Restart
`claude-codex` to apply a changed default or override to an existing session.

Auxiliary requests default to `gpt-6-luna` with `low` reasoning when launched
with `claude-codex`. This includes auto-mode classifiers, session titles, and auxiliary
summaries carrying `x-claude-code-request-class: auxiliary`. The main model,
subagents, workflows, and compaction keep their existing routing. Set
`CLAUDE_CODEX_AUXILIARY_MODEL` and `CLAUDE_CODEX_AUXILIARY_REASONING` to
override these defaults. Nonempty explicit values take precedence; restart the launcher
to apply them. The launcher enables the required gateway hint headers. A lighter model
is experimental for Claude Code's classifier; it does not guarantee fewer denials or
equivalent safety.

For example, to use the main model and medium effort for auxiliary requests too:

```bash
CLAUDE_CODEX_AUXILIARY_MODEL=gpt-6.1-sol CLAUDE_CODEX_AUXILIARY_REASONING=medium claude-codex
```

To route auxiliary requests through Ollama instead of ChatGPT, set a complete
Responses endpoint URL and the model name independently. For example:

```bash
ollama pull deepseek-v4.1-flash:cloud
CLAUDE_CODEX_AUXILIARY_ENDPOINT=http://127.0.0.1:11434/v1/responses \
CLAUDE_CODEX_AUXILIARY_MODEL=deepseek-v4.1-flash:cloud \
claude-codex --continue
```

Change `CLAUDE_CODEX_AUXILIARY_MODEL` to try another model on the same server;
change `CLAUDE_CODEX_AUXILIARY_ENDPOINT` to use another compatible server.
Restart the launcher after changing either. The endpoint must support streaming
Responses events and the selected model must fit the full classifier prompt.
ChatGPT OAuth credentials, account IDs, session hints, and Codex cache metadata
are not sent to this endpoint. No automatic fallback or fabricated verdict is used
on errors. The `:cloud` model above runs in Ollama's cloud, despite the localhost
endpoint; use an installed local model for on-device inference. Other request classes
keep the ChatGPT route. Unset the auxiliary endpoint to return to the Codex backend.

Forwarded requests write structured lines prefixed with `proxy_upstream` to
`~/.local/state/claude-codex/proxy.log`: request start and backend result,
model, effort, request class, session/request IDs, endpoint origin, and elapsed time.
A completed backend response confirms inference, not that Claude Code accepted the
classifier verdict. Prompts, generated text, OAuth tokens, endpoint credentials, and
URL queries are excluded. Restart existing launcher processes to enable new logging.

```bash
tail -f ~/.local/state/claude-codex/proxy.log | rg 'proxy_upstream'
```


Compare classifier models without executing the command under evaluation:

```bash
.venv/bin/python -m claude_codex.benchmark \
  --command 'sl-review sourcecraft list-comments yc/quantum/pr/4427 --json --insecure'
```

The test captures the installed Claude Code's full classifier policy using a harmless
synthetic tool proposal. The evaluated command is substituted only into the captured
request; its verdict is never sent to a tool executor. Five cloud models are tested
by default, three times each, at low effort with concurrency two. Use repeated
`--model NAME` arguments to select other models, `--repeats` to change the number
of trials, or `--base-url` to select another Ollama server. Cloud requests use the
Ollama account's quota. Model aliases are pulled before testing; local models may
download weights. JSON results and a Markdown summary are saved privately under
`~/.local/state/claude-codex/benchmarks/classifier-comparison/`.
This synthetic test does not replay your session history or trust settings.
The [measured comparison](docs/research/classifier-model-comparison-2026-10-03.md)
contains results for the command above.


Configuration:

```bash
CLAUDE_CODEX_MODEL=gpt-6.1-sol claude-codex
CLAUDE_CODEX_REASONING=xhigh claude-codex
CLAUDE_CODEX_LOG_MAX_BYTES=10485760 claude-codex
CLAUDE_CODEX_COMPACT_AT=180000 claude-codex
# Optional experimental native endpoint; local compact is the safe default.
CLAUDE_CODEX_REMOTE_COMPACT=1 claude-codex
```

For GPT-6.1 backends, the launcher defaults to `ANTHROPIC_MODEL=claude-opus-5-5`,
giving Claude Code a conservative 200,000-token client context window. Explicit model identities,
a forwarded `--model` option, and an in-session `/model` selection remain user overrides.
The Codex subscription models endpoint verified on 2026-10-02 reported `context_window=272000`
and `max_context_window=872000` for `gpt-6.1-sol`. A public catalog's context window is not
proof of the subscription serving limit; the launcher no longer automatically advertises `[1m]`.

`proxy.log` is rotated to one `proxy.log.1` backup when it reaches 10 MiB. Set
`CLAUDE_CODEX_LOG_MAX_BYTES` to a positive byte limit to override that threshold.

The default compaction threshold is 180,000 input tokens, leaving room below the client's
200,000-token window and the subscription's advertised default context. Before each eligible turn,
the proxy checks both the previous upstream token usage and a character-based estimate of the
current effective input, instructions, and tool schemas. This also covers restored histories and
new tool outputs; the estimate is not an exact tokenizer count. When either value reaches the
threshold, the proxy creates a concise handoff summary through
ordinary `/responses`, retains recent user messages, and continues with the reduced history.
The proxy stores replacement history only in memory, advances that branch's
Codex context window, and forwards the following turn with the compacted prefix. Parallel histories
that share one launcher session are tracked separately by their message-prefix branch. Set
`CLAUDE_CODEX_COMPACT_AT=0` to disable this behavior or a positive token threshold to change it.
`/responses/compact` is available only as the opt-in experiment `CLAUDE_CODEX_REMOTE_COMPACT=1`:
its output contract currently differs from the proxy's Anthropic translation. Compaction events are
written to `proxy.log` as `codex_compact` lines.

A Codex `context_length_exceeded` error is translated to `invalid_request_error` with
`prompt is too long` wording so Claude Code can recognize its context recovery path. Non-streaming
context failures return HTTP 400 with `x-should-retry: false`; an already-open stream receives an
error event.

Large text histories, including native Claude Code `/compact` requests, are summarized in bounded
segments through ordinary `/responses`, with up to four segment requests in parallel and `low`
reasoning effort. Segment summaries are then merged in order. Each request's serialized
instructions, input, and tools have a 120,000-byte budget; this is a conservative size bound, not
an exact tokenizer count. Tool calls, results, instructions, and tool schemas are included in the
checkpoint source. The final native compaction request disables tool execution and omits tool
schemas already covered by the checkpoint. If its original system text still exceeds the budget,
it uses a compact-only instruction and preserves the last bounded user formatting request.
Ordinary agent requests keep their system instructions and function tools. A checkpoint is a
model-generated summary, so it cannot preserve every detail.

Local summary preparation has a four-minute deadline. A native compaction request has one
four-minute deadline covering preparation and the final response; upstream keepalives do not reset
it. Failed or incomplete segments abort recovery and cancel outstanding segments. The proxy
returns an explicit error instead of forwarding the same oversized inference. Oversized image
histories and checkpoint summaries or retained format instructions that still exceed the budget
require a different checkpoint; their contents are not silently dropped. These limits apply to proxy recovery, not
Claude Code's total UI wait time or retries.

The backend URL is an internal ChatGPT Codex contract also used by OpenCode. It can change without
the compatibility guarantees of the public OpenAI Platform API.
