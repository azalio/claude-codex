# Claude Code auto mode: Stage 2 errors and a lighter GPT

Checked 2026-10-03 against primary documentation, local code, and isolated runtime probes.
No persistent permissions, CLAUDE.md, or user settings were changed.

## Finding

Saved local tool results contain the signature `Stage 2 classifier error`, including
`blocking based on stage 1 assessment`. This establishes that classifier failures occur,
but does not identify the missing command or the cause of its failed request.
Counts from transcripts include quotations and must not be treated as a failure rate.

A [first-hand Claude Code report, #74248](https://github.com/anthropics/claude-code/issues/74248),
documents the same signature on CLI 2.1.199 in July 2026 and says a retry often succeeded.
It is closed as stale/not planned, so it does not establish a fix in the installed 2.1.284.

The [Anthropic engineering article](https://www.anthropic.com/engineering/claude-code-auto-mode)
describes a cautious first pass and a reasoning second pass that reduces false positives.
If the second pass fails, retaining the first pass's block is different from a successful
policy denial. Its native evaluation is not a GPT-through-gateway evaluation.

## Local implementation

The launcher pins `CLAUDE_CODE_AUTO_MODE_SERVER=0`:
[launcher.py](../../src/claude_codex/launcher.py#L213). This selects client classifier
requests rather than unsupported Anthropic server safeguards.

[The proxy route](../../src/claude_codex/proxy.py#L990) normally uses the configured
main Codex model and effort for every client model alias. The opt-in auxiliary route
now provides:

- `CLAUDE_CODEX_AUXILIARY_MODEL`: model for the `auxiliary` request class.
- `CLAUDE_CODEX_AUXILIARY_REASONING`: its reasoning override.
- A model override with no auxiliary reasoning override selects `low`.
- Without either override, the existing main model and effort stay effective.
- Main turns, subagents, workflows, and compaction retain their original route.

The [gateway protocol](https://code.claude.com/docs/en/llm-gateway-protocol#gateway-hint-headers)
defines `auxiliary` as a broader class: classifiers, titles, and summaries.
The launcher already enables these headers. A client that omits them stays on the main route.

Anthropic [explicitly excludes support for routing Claude Code to non-Claude models](https://code.claude.com/docs/en/llm-gateway).
This route is an experimental bridge, not an Anthropic-supported classifier replacement.

## Candidate and verification

The authenticated Codex models endpoint returned HTTP 200 for installed client version
0.160.0 and listed `gpt-6-luna` with `low` effort. A catalog request for an older client
version filters out newer models. Public API model limits are not evidence of subscription limits.

OpenAI describes [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)
as an efficient model for focused, high-volume tasks. This supports testing it as a
faster candidate; it does not establish permission-classifier accuracy or fewer denials.

Local verification:

- 167 tests pass; Ruff and `git diff --check` pass.
- Actual Claude Code 2.1.284, `--bare`, auto mode, a temporary test directory, and a
  synthetic command that only prints a marker.
- The actual classifier request reached GPT-6 Luna / low, returned HTTP 200 and
  `response.completed`, and Claude Code accepted the verdict and ran the marker.
  The measured backend call took 2.697 seconds.
- A separate probe supplied a controlled above-threshold first-stage result to invoke
  the real second stage. GPT-6 Luna / low returned HTTP 200 and completed in 1.915 seconds;
  Claude Code accepted it and ran the same harmless marker.
- Main requests were mocked to keep the test deterministic; their outgoing payloads
  remained GPT-6.1 Sol / medium. These are protocol smoke tests, not a representative
  latency benchmark or a safety evaluation.

Private metadata artifacts:
`~/.local/state/claude-codex/benchmarks/20261003-auxiliary-routing/results.json` and
`stage2-results.json`. No real user commands or conversation payloads were included.
There is no evidence yet that this change fixes historical Stage 2 failures.

## Policy denials need a different remedy

The read-only `claude auto-mode config` matched `defaults` in this repository:
17 allow, 70 soft-deny, 1 hard-deny, and 21 environment entries. No personalized
environment entries were present in that effective configuration; another running
session can use different settings.

[Configure auto mode](https://code.claude.com/docs/en/auto-mode-config) documents these remedies:

- Add concrete trusted repositories, domains, and services to `autoMode.environment`.
- Preserve `"$defaults"` when extending a list.
- User, managed, and invocation settings supply `autoMode`; project/local files do not.
- Use `/permissions → Recently denied → r` for a deliberate manual retry.
- Inspect policy with `claude auto-mode defaults` and `claude auto-mode config`.

[Permission rules](https://code.claude.com/docs/en/permissions) resolve deny and ask
before allow. Narrow command allowances can avoid routine review, subject to protected
paths and other checks; an allow entry cannot override a deny entry.

[Permission modes](https://code.claude.com/docs/en/permission-modes) documents
`acceptEdits` for automatic working-directory edits with ordinary command approvals.
It is a practical temporary alternative when classifier errors persist.

## Using the separate route

```bash
CLAUDE_CODEX_AUXILIARY_MODEL=gpt-6-luna claude-codex --continue
```

The selected auxiliary model defaults to low; the main launcher default remains medium.
See [README](../../README.md) for overrides and scope. Relaunch to apply it to an existing session.
