# Configuration and usage

[Back to the project homepage](../README.md)

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
`/v1/models` discovery. The launcher enables discovery by default. Run `/model` to choose from the visible
models returned by your ChatGPT subscription catalog. Selection changes the main and subagent
backend; auxiliary summaries and the auto mode classifier retain their own model settings.
Claude Code requires discovered IDs to contain `claude` or `anthropic`, so the proxy uses
`claude-codex/<model>` IDs while displaying the catalog's model names. Hidden service models
such as `codex-auto-review` are omitted.

Catalog results are cached in memory for five minutes. If discovery fails, the proxy retains
its last successful list, or reports the configured backend when no list has been fetched.
Unknown catalog aliases are rejected rather than silently routed to the default model.
The launcher gives discovery 10 seconds; the catalog fetch has an 8-second total timeout.
Set `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=0` to disable discovery, or override
`CLAUDE_CODE_GATEWAY_MODEL_DISCOVERY_TIMEOUT_MS` if needed. The catalog compatibility
version defaults to `0.160.0` and can be changed with
`CLAUDE_CODEX_CATALOG_CLIENT_VERSION`. Token counts remain character-based estimates.

See the [compatibility matrix](gateway-compatibility.md) for supported behavior and limits,
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
with `claude-codex`. This includes session titles and auxiliary summaries carrying
`x-claude-code-request-class: auxiliary`. Recognized auto-mode classifiers use the
separate reviewer route described below. The main model,
subagents, workflows, and compaction keep their existing routing. Set
`CLAUDE_CODEX_AUXILIARY_MODEL` and `CLAUDE_CODEX_AUXILIARY_REASONING` to
override these defaults. Nonempty explicit values take precedence; restart the launcher
to apply them. The launcher enables the required gateway hint headers.

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
Responses events and the selected model must fit the complete auxiliary prompt.
These settings do not change the separate classifier route.
ChatGPT OAuth credentials, account IDs, session hints, and Codex cache metadata
are not sent to this endpoint. No automatic fallback or fabricated verdict is used
on errors. The `:cloud` model above runs in Ollama's cloud, despite the localhost
endpoint; use an installed local model for on-device inference. Other request classes
keep the ChatGPT route. Unset the auxiliary endpoint to return to the Codex backend.

### Auto-mode classifier

Recognized Claude Code classifier requests use `codex-auto-review` with `low`
reasoning through the ChatGPT subscription by default. This is the preferred reviewer
model in Codex's ChatGPT provider. Main answers, compaction, titles, and ordinary
auxiliary summaries retain their own routes.

The reviewer retains the complete classifier policy and evidence, requests JSON Schema,
validates the score and category, and converts a valid result to Claude Code's native
`<severity>...</severity>` / optional `<category>...</category>` format.
The policy, numeric scores, and allow/block thresholds are preserved. Valid native XML
is also accepted from providers that ignore the schema. Partial or malformed responses
never become verdicts. Formatting/incomplete-response and transient transport/server
errors get up to three attempts within one 90-second deadline. A valid deny is not retried.
Exhaustion returns a review error, without fabricating an allow or deny or switching models.

Classifier recognition uses the trusted system instruction's native numeric severity
output contract together with the auxiliary request-class hint. Other auxiliary prompts
are unaffected. A future Claude Code contract change may require updating recognition.

- `CLAUDE_CODEX_CLASSIFIER_MODEL`: reviewer model; default `codex-auto-review`.
- `CLAUDE_CODEX_CLASSIFIER_REASONING`: effort; default `low`.
- `CLAUDE_CODEX_CLASSIFIER_ENDPOINT`: optional full independent endpoint URL.
  Unset it to use ChatGPT, regardless of `CLAUDE_CODEX_AUXILIARY_ENDPOINT`.
- `CLAUDE_CODEX_CLASSIFIER_TIMEOUT`: positive seconds for the total deadline;
  default 90. Invalid, nonpositive, and nonfinite values use the default.

For a local Ollama classifier with constrained JSON generation, use Chat Completions:

```bash
CLAUDE_CODEX_CLASSIFIER_ENDPOINT=http://127.0.0.1:11434/v1/chat/completions \
CLAUDE_CODEX_CLASSIFIER_MODEL=auto-mode-bench-qwen3-5-4b \
CLAUDE_CODEX_CLASSIFIER_REASONING=none \
claude-codex --continue
```

The example alias was created by the local benchmark with a 65,536-token context.
On another machine, create an alias with enough context for the full policy and evidence.
Replace the model name to try another installed model. The Chat Completions adapter
sends the complete text evidence with its original roles serialized as JSON; image
evidence is rejected explicitly rather than flattened or dropped.
A `/v1/responses` endpoint can also be used if it supports the requested schema.
Independent endpoints receive no ChatGPT credentials or routing metadata.

Private diagnostic records include each attempt, raw response events, `invalid_format`
errors and the accepted `verdict` with its score, category and rationale.
The payload-free `classifier_review` lines in `proxy.log` report validation outcomes.
A format-valid answer is not a guarantee of correct policy judgment.

Forwarded requests write structured lines prefixed with `proxy_upstream` to
`~/.local/state/claude-codex/proxy.log`: request start and backend result,
model, effort, request class, session/request IDs, endpoint origin, and elapsed time.
A completed backend response confirms inference, not that Claude Code accepted the
classifier verdict. Prompts, generated text, OAuth tokens, endpoint credentials, and
URL queries are excluded. Restart existing launcher processes to enable new logging.

```bash
tail -f ~/.local/state/claude-codex/proxy.log | rg 'proxy_upstream'
```


Auxiliary request bodies and raw Responses SSE events are also recorded by default in
`~/.local/state/claude-codex/auxiliary/requests.jsonl`. Request and response entries
share the `request_id` from `proxy_upstream`, so you can inspect the actual classifier
reply. This includes auxiliary titles and summaries as well as classifier checks.
The private directory uses mode 0700 and files use mode 0600. Logs rotate at 10 MiB
with one backup; a single larger record is retained intact. Authorization headers,
OAuth/account credentials, endpoint queries, and Codex routing/cache metadata are excluded.
Prompt and response content is stored verbatim and can contain sensitive task data.
Set `CLAUDE_CODEX_AUXILIARY_LOG=0` to disable this recording.
Restart the launcher to enable the updated code; earlier replies cannot be recovered.

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
of trials, `--stage 2` to test the user-intent review, `--effort none` to disable
thinking where supported, or `--base-url` to select another Ollama server.
Cloud requests use the Ollama account's quota. Installed models and custom aliases are reused; missing
models are pulled and local models may download weights. JSON results and a
Markdown summary are saved privately under
`~/.local/state/claude-codex/benchmarks/classifier-comparison/`.
This synthetic test does not replay your session history or trust settings.
The [measured comparison](research/classifier-model-comparison-2026-10-03.md)
contains results for the command above. Local classifiers need a sufficiently large
active context for the full native policy; use a separate Modelfile alias with
`PARAMETER num_ctx 65536` and verify it with `ollama ps`. A short-prompt speed
test does not establish compatibility with auto mode. See the
[local model measurements](research/local-ollama-classifiers-2026-10-03.md)
for both stages, latency, protocol failures, and false approvals.


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
segments through ordinary `/responses`, with up to four segment requests in parallel.
Internal checkpoint summaries default to `gpt-6-luna` with `low` reasoning. Set
`CLAUDE_CODEX_COMPACTION_MODEL` to choose another Codex subscription model; this does
not change normal answers or the auxiliary endpoint. Segment summaries are then merged in order. Each request's serialized
instructions, input, and tools have a 120,000-byte budget; this is a conservative size bound, not
an exact tokenizer count. Tool calls, results, instructions, and tool schemas are included in the
checkpoint source. The final native compaction request disables tool execution and omits tool
schemas already covered by the checkpoint. If its original system text still exceeds the budget,
it uses a compact-only instruction and preserves the last bounded user formatting request.
Ordinary agent requests keep their system instructions and function tools. A checkpoint is a
model-generated summary, so it cannot preserve every detail.

Completed checkpoint summaries are cached privately under
`~/.local/state/claude-codex/checkpoints/`. Cache keys bind the backend, session,
agent, request class, selected checkpoint model, instructions, and exact segment
content. After a timeout or launcher restart, matching completed segments are reused;
unfinished or failed segments are retried. Appending a new user message invalidates
changed segments, not earlier matching ones. Merge summaries are cached too.
Source transcripts are not stored in this cache; summary files use mode 0600 inside
mode-0700 directories. Entries older than 24 hours are ignored. Set
`CLAUDE_CODEX_CHECKPOINT_CACHE=0` to disable cache reads and writes.
Replacement conversation history still remains in memory.

Progress is logged as `codex_checkpoint`, including segment/total, selected model,
and started/completed/cached/cancelled/error outcomes, without transcript or summary text.

```bash
tail -f ~/.local/state/claude-codex/proxy.log | rg 'codex_checkpoint|codex_compact'
```

Local summary preparation has a 15-minute deadline. A native compaction request has one
15-minute deadline covering preparation and the final response; upstream keepalives do not reset
it. Set `CLAUDE_CODEX_COMPACTION_TIMEOUT` to a positive number of seconds to override
the deadline (for example, `CLAUDE_CODEX_COMPACTION_TIMEOUT=1800 claude-codex --continue`).
Invalid, nonpositive, or nonfinite values use the default. Restart the launcher to apply
a changed timeout; running proxy processes retain their existing settings. Failed or incomplete segments abort recovery and cancel outstanding segments. The proxy
returns an explicit error instead of forwarding the same oversized inference. Oversized image
histories and checkpoint summaries or retained format instructions that still exceed the budget
require a different checkpoint; their contents are not silently dropped. These limits apply to proxy recovery, not
Claude Code's total UI wait time or retries.

The backend URL is an internal ChatGPT Codex contract also used by OpenCode. It can change without
the compatibility guarantees of the public OpenAI Platform API.
