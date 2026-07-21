#!/usr/bin/env bash
# Write-time Python quality gate for xpcsjax (Plankton-inspired, ruff/uv-native).
#
# Real Plankton (github.com/alxfazio/plankton) needs its own repo clone plus
# brew-installed linters per language and isn't installed here. This is the
# Python-only equivalent built on tooling xpcsjax already ships with:
#   Phase 1 (silent): ruff format + ruff check --fix
#   Phase 2: whatever ruff can't auto-fix is surfaced back to the agent
# No subprocess model-delegation phase (Plankton's Phase 3) — spawning a
# nested `claude -p` from a PostToolUse hook risks colliding with this
# project's other PostToolUse/PreToolUse hooks (CCG, context-mode).
set -uo pipefail

input=$(cat)
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)

[[ -z "$file_path" ]] && exit 0
[[ "$file_path" != *.py ]] && exit 0
[[ ! -f "$file_path" ]] && exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
command -v uv >/dev/null 2>&1 || exit 0

uv run ruff format --quiet "$file_path" >/dev/null 2>&1
uv run ruff check --quiet --fix "$file_path" >/dev/null 2>&1

if ! remaining=$(uv run ruff check "$file_path" 2>&1); then
  echo "[quality-gate] ruff violations remain in $file_path:" >&2
  echo "$remaining" >&2
  exit 2
fi

exit 0
