#!/usr/bin/env bash
set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [[ -L "$SOURCE" ]]; do
  SOURCE_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ "$SOURCE" = /* ]] || SOURCE="$SOURCE_DIR/$SOURCE"
done

REPO_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
BIN_DIR="${CLAUDE_CODEX_BIN_DIR:-$HOME/bin}"
TARGET="$BIN_DIR/claude-codex"
WRAPPER="$REPO_DIR/bin/claude-codex"
TEMP_LINK="$BIN_DIR/.claude-codex.$$.tmp"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "error: claude executable is not available in PATH" >&2
  exit 1
fi

cleanup() {
  rm -f "$TEMP_LINK"
}
trap cleanup EXIT

echo "Installing claude-codex from $REPO_DIR"
# A virtualenv is not relocatable: every generated console script embeds the
# absolute path to its Python interpreter. Reinstall all packages so moving
# the repository cannot leave stale shebangs in pytest, uvicorn, or our CLI.
uv sync --project "$REPO_DIR" --frozen --reinstall

"$REPO_DIR/.venv/bin/python" -c \
  'import claude_codex; print(f"claude-codex package {claude_codex.__version__} ready")'

mkdir -p "$BIN_DIR"
ln -s "$WRAPPER" "$TEMP_LINK"
mv -f "$TEMP_LINK" "$TARGET"

echo "Installed command: $TARGET -> $WRAPPER"
echo "Run: claude-codex"
