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
if ! jq_err=$(echo "$input" | jq -r '.tool_input.file_path // empty' 2>&1); then
  echo "[quality-gate] jq failed to parse hook input, skipping quality gate: $jq_err" >&2
  exit 0
fi
file_path=$jq_err

[[ -z "$file_path" ]] && exit 0
[[ "$file_path" != *.py ]] && exit 0
[[ ! -f "$file_path" ]] && exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
if ! command -v uv >/dev/null 2>&1; then
  echo "[quality-gate] uv not found on PATH, skipping quality gate for $file_path" >&2
  exit 0
fi

# ruff format failing (nonzero) almost always means a real problem (e.g. a
# syntax error making the file unparseable), unlike ruff check --fix below,
# whose nonzero exit is the routine "found violations it can't auto-fix" case
# -- that's already surfaced faithfully by the re-check a few lines down.
if ! format_out=$(uv run ruff format --quiet "$file_path" 2>&1); then
  echo "[quality-gate] ruff format failed on $file_path:" >&2
  echo "$format_out" >&2
fi
uv run ruff check --quiet --fix "$file_path" >/dev/null 2>&1

if ! remaining=$(uv run ruff check "$file_path" 2>&1); then
  echo "[quality-gate] ruff violations remain in $file_path:" >&2
  echo "$remaining" >&2
  exit 2
fi

exit 0
