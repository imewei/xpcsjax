# Automated structural doc-coverage check, content accuracy stays manual

**Status**: accepted

## Context

A drift audit (2026-07-27) found `xpcsjax.device`/`io`/`utils` had no
`docs/source/api/*.rst` page — only member-suppressed cross-reference stubs.
Everything else audited clean. The gap existed because doc structure was
never mechanically checked; it relied on someone remembering.

## Decision

Add an automated test (`tests/test_docs_structure.py`, part of the default
`pytest`/`make test`/`make verify` run — no advisory carve-out) that, for
every top-level `xpcsjax` submodule except an explicit hardcoded exclusion
list (currently `{"gui"}`, a PySide6 app with no library surface), asserts
`docs/source/api/{name}.rst` exists. That's the entire check — existence
only, nothing about page content.

The root `xpcsjax` package (`__init__.py`'s lazy `_LAZY_EXPORTS`) is out of
scope — it already has its own runtime-assert enforcement mechanism, and
`public.rst` (not a `{name}.rst`-per-package file) documents it separately.

Content accuracy — and, critically, *symbol-level completeness* — is **not**
automated. It stays a manual/agent-driven audit, run occasionally.

## Considered options

- **Assert every `__all__` name appears (as literal text, or via a
  page-level `automodule`/`:members:` directive) in its package's `.rst`
  page.** Explored in depth and rejected on empirical grounds, not
  hypothetically: `xpcsjax.cli.__all__` has 12 names, `cli.rst` mentions 2;
  `xpcsjax.optimization.__all__` has 19, `optimization.rst` mentions 8;
  `core.rst` misses 2/7, `data.rst` misses 1/8. These pages are already
  audit-confirmed accurate and complete in the sense that matters to a
  reader — they deliberately curate the *user-facing* subset of `__all__`,
  which also holds internal/implementation-detail exports (`NLSQ_AVAILABLE`,
  `OPTIMIZATION_STATUS`, `create_angle_stratified_indices`) that were never
  meant to get a doc entry. `__all__` in this codebase means "import-star
  safe," not "doc-worthy" — see [CONTEXT.md](../../CONTEXT.md)'s `__all__`
  member vs documented symbol` entry. Any name-coverage check built on
  `__all__` would immediately red-flag 7 of 11 correctly-documented pages.
  There is no cheaper substitute (a curated "doc-worthy names" list would
  have to be maintained by hand per package, which is real ongoing work and
  duplicates the page's own content) — so this stays a manual-audit-only
  concern.
- **Advisory-only** (like the project's local `mypy` gate). Rejected: page
  existence is a binary, deterministic fact with no false-positive risk,
  unlike non-strict `mypy` — softening it just reintroduces the "trust
  someone to remember" failure mode this exists to close.
- **Automating content accuracy too** (architecture claims, command
  examples, config keys). Rejected as infeasible — truth-checking needs
  judgment a text-based test can't do; the earlier audit needed an agent,
  not a grep.

## Consequences

- Adding a new top-level `xpcsjax` package without a matching
  `docs/source/api/{name}.rst` page now fails `make test`/`make verify`
  immediately, instead of silently shipping undocumented (as
  `device`/`io`/`utils` did).
- Symbol-level gaps — a page that exists but omits a name a reader would
  reasonably expect — are **not** caught by automation, by design. Only a
  manual/agent-driven audit (like the one that produced this ADR) catches
  those; they're not rare or a small residual risk, they're the *normal*
  state of every hand-curated page in this codebase.
- `docs/source/development/testing.rst` gets a one-line mention so
  contributors learn the requirement before hitting the test failure.
