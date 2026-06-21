# Shell Completion Generation — Design

**Date:** 2026-06-21
**Status:** Approved (brainstorming) — pending implementation plan
**Sub-project:** 1 of 3 (GUI / CLI / shell completion optimization effort)

## Problem

`xpcsjax/runtime/shell/completion.sh` (247 LOC) is a hand-curated bash
completion script that mirrors the CLI argument surface defined across several
argparse parsers (`xpcsjax/cli/args_parser.py` and five secondary entry
points). The two are kept in sync manually. This is a **two-sources-of-truth**
design that drifts: a fix landed on a branch on 2026-06-20, was never merged,
and the drift silently returned.

Three live defects confirmed against `args_parser.py` on 2026-06-21:

| `completion.sh` says | `args_parser.py` reality | Defect |
|---|---|---|
| `--initial-contrast`, `--initial-offset` (line 73) | Do **not** exist — per-angle scaling is explicitly *not* a CLI override (parser comment L272–274) | Phantom completions |
| *(absent)* | `--initial-beta` exists (L207) | Missing completion |
| *(absent)* | `--no-multistart` exists (L133) | Missing completion |

There is also **no parser↔completion *parity* test in the working tree** — the
`test_completion_parser_parity.py` guard referenced in project memory is absent
(verified). Completion-*adjacent* tests do exist and must be kept in step with
this work: `tests/cli/test_post_install.py`, `tests/cli/test_uninstall_scripts.py`,
`tests/runtime/test_runtime_shell.py`, and
`tests/optimization/test_heterodyne_completion_layout.py`. None of them asserts
parser↔completion parity; that gap is what this sub-project closes.

## Goals

Serve all four optimization dimensions for this subsystem:

- **Robustness/correctness:** make completion↔parser drift *structurally
  impossible* (a derived artifact + a CI gate), and fix the three live defects.
- **Design/maintainability:** single source of truth; `completion.sh` becomes a
  generated artifact; per-command parser construction is separated from
  execution (standard argparse idiom, independently testable).
- **UX:** completions are always correct and as rich as the parser allows
  (enums from `choices`, file/dir hints, config-file discovery).
- **Performance:** the shipped artifact stays a static, dependency-free bash
  script — no Python interpreter or JAX import runs on TAB.

## Non-goals (YAGNI)

- Fish **completion**. Shells in scope for *completion* are **bash and zsh only**
  (zsh consumes the bash script via `bashcompinit`). Fish completion is dropped.
  **Scope guard:** this removes fish only from the *completion* surface — the
  separate fish **XLA-activation** machinery in `post_install`
  (`detect_shell_type`, `_install_xla_fish_activation`, `XLA_CONFIG_FISH`, the
  `runtime/shell/__init__.py` fish helpers + their tests) is a different feature
  and is **out of scope** for this sub-project. `--shell fish` therefore stays
  valid for XLA-only installs; only the fish *completion* branch and fish from
  the completion-relevant `--shell` paths are removed. (This avoids ballooning a
  drift fix into an unrelated multi-hundred-line XLA refactor — flagged by both
  reviewers.)
- Runtime dynamic completion (argcomplete). Rejected: runs Python per TAB
  (latency + risk of heavy import), adds a runtime dependency, degrades in bare
  conda/mamba shells.
- Completing positional *values* beyond the curated dynamic hints below.
- zsh-native `compsys`; `bashcompinit` is sufficient.

## Approach

**Commit-time generation + CI parity gate.** A Python generator introspects
every entry point's argparse parser and emits `completion.sh`. A pytest parity
test regenerates into memory and diffs against the committed file; drift fails
the build. The generator owns **all** command blocks.

### Why introspection is feasible

`create_parser()` is a pure function returning a standard
`argparse.ArgumentParser`. Each option exposes everything the generator needs:

- `action.option_strings` → flag names (incl. short aliases).
- `action.choices` → enum completion via `compgen -W`.
- `action.nargs == 0` → the option is a **flag** (covers `store_true`,
  `store_false`, `store_const`, `count`, `version`, `help`); otherwise it takes a
  value. Classifying on `nargs == 0` rather than an action-class allow-list is
  deliberate — see the BLOCKER fix in Component 3.

Two things are **not** introspectable and stay curated in the per-command hint
table (Component 2):

1. **File-vs-directory intent.** `type=Path` covers both `--config`→YAML file and
   `--output`→directory; argparse carries no dir-vs-file marker.
2. **Path-like `str` args.** Some path args use `type=str`, not `Path` (e.g.
   `xpcsjax-config --data/-d`), so a `type`-based "Path→file" rule would miss
   them and silently drop today's file completion. These need an explicit `file`
   hint.

## Components

Each is a single-purpose unit with a defined interface.

### 1. Parser-factory convention

Every entry point exposes `build_parser() -> argparse.ArgumentParser` as a
module-level factory. "Import-light" here means **no Qt or JAX import is
triggered by importing the module**. (Note: importing any `xpcsjax` submodule
runs `xpcsjax/__init__.py`, which only sets JAX-related *environment variables* —
it does not import JAX. That env mutation is harmless at generation time. To keep
the parity test hermetic, it imports the generator in a clean subprocess.)

- Already present: `cli/args_parser.create_parser`,
  `cli/config_generator._build_parser`.
- Five secondary entry points lack a reusable factory: four build the parser
  inline inside `main()` (`cli/xla_config`, `post_install`, `uninstall_scripts`,
  `runtime/utils/system_validator`); the GUI builds its parser inside
  `gui/app._parse_cli_args()`. Extract a module-level `build_parser()` from each.
- **GUI is already import-light** — `gui/app.py` has no top-level Qt/JAX imports
  and lazy-imports PySide6 inside `main()`, so the generator can import it safely
  as-is. The only work is extracting the parser-building lines from
  `_parse_cli_args()` into a `build_parser()` factory (in `gui/app.py`; a
  separate `gui/cli_args.py` is optional and justified only by consistency, *not*
  by import-safety — the earlier "avoid importing PySide6" rationale was wrong).

Behavior of each `main()` is unchanged: it calls its `build_parser()` then
`parse_args()` exactly as before. Refactor is pure extraction.

### 2. Command registry — `xpcsjax/runtime/shell/completion_spec.py`

One declarative list, the single source for *what* to generate and *how* to
register it:

```python
CommandSpec(
    completion_func: str,           # e.g. "_xpcsjax"
    command_names: list[str],       # all names+aliases bound to this func
    parser_factory: Callable[[], argparse.ArgumentParser],
    dynamic_hints: dict[str, str],  # flag -> "configfile" | "file" | "dir" | "threads"
)
```

Covers: `xpcsjax`/`xj` (+ `xjexp`/`xjsim` reuse `_xpcsjax`),
`xpcsjax-config`/`xj-config`, `xpcsjax-config-xla`/`xj-config-xla`,
`xpcsjax-post-install`/`xj-post-install`, `xpcsjax-cleanup`/`xj-cleanup`,
`xpcsjax-validate`/`xj-validate`, `xpcsjax-gui`/`xj-gui`.

`dynamic_hints` is **per-`CommandSpec`** (not global by flag name) — the same
flag means different things in different commands. The four hint kinds:

| Hint | Emits | Example flags |
|---|---|---|
| `configfile` | cached YAML discovery (`_xpcsjax_get_config_files`) | `xpcsjax --config/-c` |
| `dir` | `_filedir -d` | `xpcsjax --output/-o` |
| `file` | `_filedir` | `xpcsjax-config --output/-o`, `xpcsjax-config --data/-d` |
| `threads` | nproc-based numeric list | `--threads` |

Per-command necessity (concrete cases the reviewers flagged):
- `xpcsjax --output/-o` is a **directory**; `xpcsjax-config --output/-o` is a
  **YAML file** (default `xpcsjax_config.yaml`). Same flag, opposite hint —
  hence per-command tables, never a global map.
- `xpcsjax-config --data/-d` is `type=str` (a *path string*, not `Path`), so it
  gets an explicit `file` hint; without it the generator would emit no file
  completion and silently regress today's behavior.

`dynamic_hints` (a few lines per command) is the only hand-maintained surface;
everything else is derived from the parser and unit-tested.

### 3. Generator — `xpcsjax/runtime/shell/generate_completion.py`

`generate() -> str` returns the full completion script text.

- Emits a fixed **preamble**: cache dir vars, the `_init_completion` /
  `_filedir` fallback (for bare conda/mamba shells), and the cached
  `_xpcsjax_get_config_files` helper. These are copied **verbatim** from the
  current `completion.sh` shim block (byte-exact parity depends on it).
- For each `CommandSpec`: walk the parser's options, classify each, emit the
  `case "$prev"` block (enum / file / dir / configfile / threads) and the
  trailing option-name `compgen -W` list.
- Emits the registration **footer** (`complete -F <func> <name>` for every
  `command_name`).
- **Fails loudly** (raises) on: a parser import error, or an option that matches
  no classifier rule. No silent skips — integrity over a quietly-incomplete
  script. Runnable as `python -m xpcsjax.runtime.shell.generate_completion`
  (writes the file) and importable (`generate()` returns text).

Classification rules (evaluated top-to-bottom; first match wins):

| Condition | Completion emitted |
|---|---|
| `dynamic_hints[flag] == "configfile"` | `compgen -W "$(_xpcsjax_get_config_files)"` |
| `dynamic_hints[flag] == "dir"` | `_filedir -d` |
| `dynamic_hints[flag] == "file"` | `_filedir` |
| `dynamic_hints[flag] == "threads"` | nproc-based numeric list |
| `action.choices` set | `compgen -W "<choices>"` |
| `action.type is Path`, no hint | `_filedir` (file) |
| takes a value (any `nargs != 0`), no choices | `return` (no value hint) |
| **flag: `action.nargs == 0`** (`store_true`/`store_false`/`store_const`/`count`/`version`/`help`) | name only, in option list |

**BLOCKER fix (reviewers):** the flag rule keys on `action.nargs == 0`, **not** an
action-class allow-list. The earlier draft listed only
`store_true/store_const/count/version`, which omitted `store_false` — and
`--no-plot` is `action="store_false"` (in the `--plot/--no-plot` mutually
exclusive group). Under the fail-loudly rule that omission would crash the
generator on its first run. `nargs == 0` covers every zero-argument action,
present and future, so the "every option matches a rule" invariant holds.

Edge cases now provably covered:
- **`count` actions** (`-v/--verbose`): completed name-only as `-v`/`--verbose`.
  Repeated short forms (`-vv`) are intentionally **not** synthesized (documented
  decision, not an omission).
- **Multi-value args** (`--phi` is `type=float, nargs="+"`): fall through the
  value-flag rule → no value hint. No special `nargs` handling needed.
- **Mutually exclusive group** (`--plot/--no-plot`): both members are ordinary
  flags; the group adds nothing the per-action rules don't already handle.
- **`--validate`** in `xpcsjax-config` is `store_true`; today's `completion.sh`
  wrongly file-completes it. Generation fixes this for free (covered by a
  regression test).

Implementation note: walking `parser._actions` uses a private attribute, which is
acceptable for a repo-owned generator pinned by tests; if a public traversal
suffices it is preferred.

### 4. `completion.sh` (generated)

Becomes generator output, with a banner:
`# GENERATED by generate_completion.py — do not edit. Run 'make completion'.`

### 5. Tests — `tests/cli/test_completion_parity.py` (+ classifier units)

`tests/cli/` already exists — the new file is *added* to it (confirm no
conftest/`__init__` collision). The basename/location differs from the
never-merged `tests/runtime/test_completion_parser_parity.py` from memory; this
is the new, canonical guard replacing it.

- **Parity gate:** `assert generate() == completion_sh.read_text()`; on failure,
  show a unified diff and the message "run `make completion`". Pure and fast —
  runs in the default suite so `make verify` catches drift pre-push. Imports the
  generator in a clean subprocess to stay hermetic.
- **Syntax smoke:** `bash -n completion.sh` exits 0.
- **Classifier units:** enum→list, `Path`→file, `file`-hint→`_filedir`,
  `dir`-hint→`_filedir -d`, `store_false`/zero-arg→name-only,
  value-arg→`return`, unclassifiable→raises.
- **Regression (the defects this closes):** phantom `--initial-contrast`/
  `--initial-offset` gone; `--initial-beta` + `--no-multistart` present;
  `xpcsjax-config --validate` is name-only (not file-completed);
  `xpcsjax-config --data/-d` **is** file-completed (str-path guard).
- **Existing completion-adjacent tests:** review/update
  `tests/cli/test_post_install.py`, `tests/cli/test_uninstall_scripts.py`, and
  `tests/runtime/test_runtime_shell.py` for any assertions touching fish
  *completion* choices; leave their fish *XLA* assertions intact (per the
  Component 7 scope guard).

### 6. `make completion` target

Regenerates and writes `completion.sh`. One-line note added where post-install
references completion wiring.

### 7. Fish removal (completion only — scoped)

Remove fish from the **completion** surface only:
- No fish branch is emitted by the completion generator (it only targets bash;
  zsh reuses it via `bashcompinit`).
- Where `post_install` wires *completion*, fish is dropped (e.g.
  `install_fish_completion` / `_install_completion_fish_activation` for the
  completion path).

**Out of scope (do NOT touch):** the fish **XLA-activation** machinery —
`detect_shell_type`, `_install_xla_fish_activation`, `XLA_CONFIG_FISH`,
`get_xla_config_source_path("fish")`, the `runtime/shell/__init__.py` fish
helpers, and their tests. These are a separate feature; `--shell fish` remains
valid for XLA-only installs. Removing them would be an unrelated multi-hundred-
line refactor (flagged by both reviewers) and is explicitly deferred.

Implementation must verify, before editing, which `post_install` fish references
belong to completion vs XLA, and remove only the former.

## Data flow

```
build_parser() (per command)
      → completion_spec registry
      → generate_completion.generate()
      → completion.sh  (committed artifact)
      → sourced by bash/zsh shell
parity test: re-run generate(), diff vs committed completion.sh
```

## Error handling

- Generator raises on import failure or unclassifiable option (no silent
  degradation).
- Parity test failure prints a diff and the remediation command.
- Generated script preserves the existing `_init_completion` fallback so
  completion still works without system bash-completion (conda/mamba).

## Risks & mitigations

- **Import side effects** (Qt/JAX) when the generator imports a parser factory →
  no factory module imports Qt/JAX at load (`gui/app.py` already lazy-imports
  PySide6 inside `main()`). Package import runs only `__init__.py`'s env-var
  setup (no JAX import); the parity test imports the generator in a clean
  subprocess to stay hermetic.
- **Byte-exact parity brittleness** (whitespace) → the generator is the sole
  writer of `completion.sh`; the committed file is always its output, so the
  diff is stable. Contributors regenerate via `make completion`.
- **Hidden curated divergence** → `dynamic_hints` is the only hand-maintained
  surface and is unit-tested; everything else is derived.

## Acceptance criteria

1. `completion.sh` is generated; the three live defects are resolved.
2. `python -m xpcsjax.runtime.shell.generate_completion` reproduces the
   committed file byte-for-byte.
3. `tests/cli/test_completion_parity.py` passes and *fails* if any parser option
   is added/removed without regenerating.
4. Every command exposes a `build_parser()` factory (2 existing + 5 extracted);
   each `main()`'s behavior is unchanged.
5. The generator handles `store_false` (`--no-plot`) without raising; generated
   output completes `xpcsjax-config --data/-d` as a file and `--validate` as
   name-only.
6. Fish is removed from the **completion** surface only; fish **XLA** activation
   is untouched and `--shell fish` still works for XLA installs.
7. `bash -n completion.sh` passes; `make verify` is green (including the existing
   `test_post_install`/`test_uninstall_scripts`/`test_runtime_shell` suites).

## Review & validation (2026-06-21)

Reviewed by **codex** and a **Claude** agent (independent passes); **agy** was
attempted twice but stalled on slow/dead network mounts without producing
findings. Both completed reviewers converged; all design-driving claims were then
re-verified directly against the code:

- **BLOCKER (fixed):** `--no-plot` is `store_false` (args_parser.py:320) — the
  classifier now keys on `nargs == 0`, so the fail-loudly generator no longer
  crashes on first run.
- **MAJOR (fixed):** path-like `str` args (`config_generator --data`, str at
  config_generator.py:429) and the `--output` file-vs-dir split now use a
  per-command `file`/`dir` hint kind.
- **MAJOR (fixed):** the GUI is already import-light (PySide6 lazy-imported in
  `gui/app.py:84`); the `gui/cli_args.py` "avoid Qt import" rationale was false
  and is removed.
- **MAJOR (fixed):** fish removal scoped to *completion* only — fish XLA
  activation (`XLA_CONFIG_FISH`, `runtime/shell/__init__.py`) stays.
- **MAJOR (fixed):** the "only completion test" claim was false; existing
  adjacent tests are now named and slated for review.
- **MINOR/NIT (fixed):** `--validate` regression, `-vv` decision, `nargs`,
  mutually-exclusive group, `__init__` env-mutation, `parser._actions` private
  API, and verbatim-preamble all documented.

## Follow-ups (later sub-projects)

- Sub-project 2 (CLI): with completion now drift-proof, CLI arg restructuring is
  safe — completion auto-follows.
- The `build_parser()` factories introduced here make per-command CLI unit tests
  straightforward in sub-project 2.
