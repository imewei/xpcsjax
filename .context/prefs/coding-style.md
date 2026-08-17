# Coding Style Guide

> 此文件定义团队编码规范，所有 LLM 工具在修改代码时必须遵守。
> 提交到 Git，团队共享。

## General
- Prefer small, reviewable changes; avoid unrelated refactors.
- Keep functions short (<50 lines); avoid deep nesting (≤3 levels).
- Name things explicitly; no single-letter variables except loop counters.
- Handle errors explicitly; never swallow errors silently.

## Language-Specific

### Python
- Python 3.12+, managed via `uv`; never bare `pip install`.
- `JAX_ENABLE_X64=1` is mandatory (parameters span 6+ orders of magnitude).
- No `from module import *` (enforced by ruff `F` rule and user CLAUDE.md).
- Strict type hints at API boundaries and config objects; mypy runs advisory
  locally (`make verify`) but is a hard CI gate — run `uv run mypy xpcsjax`
  before pushing.
- New physics parameters register in `xpcsjax/config/parameter_registry.py`
  first — it is the single source of truth for names/bounds.
- JIT-safe interpolation only (`interpax`, never `jax.numpy.interp` in JIT'd
  paths).

## Git Commits
- Conventional Commits, imperative mood.
- Atomic commits: one logical change per commit.

## Testing
- Every feat/fix MUST include corresponding tests.
- Coverage must not decrease.
- Fix flow: write failing test FIRST, then fix code.

## Security
- Never log secrets (tokens/keys/cookies/JWT).
- Validate inputs at trust boundaries.
