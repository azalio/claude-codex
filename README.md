# claude-codex

Run the normal Claude Code CLI while using a ChatGPT Codex subscription as the model backend.

The command starts a local Anthropic-compatible proxy, points `ANTHROPIC_BASE_URL` at it, and then
executes the regular `claude` binary with every original argument. Claude Code keeps its UI, slash
commands, skills, hooks, MCP servers, and tools. The proxy translates Anthropic Messages and SSE to
the Codex Responses protocol.

## Authentication

No OpenAI API key is used. Credentials are loaded in this order:

1. `CLAUDE_CODEX_AUTH_FILE`
2. `~/.config/claude-codex/auth.json`
3. OpenCode OAuth credentials in `~/.local/share/opencode/auth.json`
4. Codex CLI credentials in `$CODEX_HOME/auth.json` or `~/.codex/auth.json`

If OpenCode is already connected to ChatGPT, the third source works immediately. Refreshed
credentials are copied to the private `claude-codex` cache with mode `0600`; OpenCode and Codex
credential files are never modified.

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
CLAUDE_CODEX_MODEL=gpt-5.6-sol claude-codex
CLAUDE_CODEX_REASONING=high claude-codex
CLAUDE_CODEX_LOG_MAX_BYTES=10485760 claude-codex
```

`proxy.log` is rotated to one `proxy.log.1` backup when it reaches 10 MiB. Set
`CLAUDE_CODEX_LOG_MAX_BYTES` to a positive byte limit to override that threshold.

The backend URL is an internal ChatGPT Codex contract also used by OpenCode. It can change without
the compatibility guarantees of the public OpenAI Platform API.
