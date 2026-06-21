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

There is also **no parity test in the working tree** — the
`test_completion_parser_parity.py` guard referenced in project memory is absent.
The only completion-adjacent test is `tests/optimization/test_heterodyne_completion_layout.py`.

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

- Fish completion. Shells in scope are **bash and zsh only** (zsh consumes the
  bash script via `bashcompinit`). Fish is dropped from `post_install`'s
  `--shell` choices and its emitting branch removed.
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
- `action.type` (`Path` vs `float`/`int`) and action class (`store_true`,
  `store_const`, `count`, `version`) → value-vs-flag classification.

The **one** thing not introspectable is file-vs-directory intent (`type=Path`
covers both `--config`→YAML file and `--output`→directory). That stays curated
in a small explicit hint table (see Component 2).

## Components

Each is a single-purpose unit with a defined interface.

### 1. Parser-factory convention

Every entry point exposes `build_parser() -> argparse.ArgumentParser` in an
**import-light** module (no Qt or JAX imported at module load).

- Already present: `cli/args_parser.create_parser`,
  `cli/config_generator._build_parser`.
- Extract a factory from the 5 inline parsers (parser currently built inside
  `main()`): `cli/xla_config`, `post_install`, `uninstall_scripts`,
  `runtime/utils/system_validator`, and the GUI launcher.
- For the GUI: move its trivial `--version/--help` parser into a new
  import-light `xpcsjax/gui/cli_args.py` (`build_parser()`), imported by
  `gui/app.main`, so the generator never imports PySide6.

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
    dynamic_hints: dict[str, str],  # flag -> "configfile" | "dir" | "threads"
)
```

Covers: `xpcsjax`/`xj` (+ `xjexp`/`xjsim` reuse `_xpcsjax`),
`xpcsjax-config`/`xj-config`, `xpcsjax-config-xla`/`xj-config-xla`,
`xpcsjax-post-install`/`xj-post-install`, `xpcsjax-cleanup`/`xj-cleanup`,
`xpcsjax-validate`/`xj-validate`, `xpcsjax-gui`/`xj-gui`.

`dynamic_hints` is the only hand-maintained data (~a dozen lines), e.g.
`{"--config": "configfile", "-c": "configfile", "--output": "dir",
"--threads": "threads"}`.

### 3. Generator — `xpcsjax/runtime/shell/generate_completion.py`

`generate() -> str` returns the full completion script text.

- Emits a fixed **preamble**: cache dir vars, the `_init_completion` /
  `_filedir` fallback (for bare conda/mamba shells), and the cached
  `_xpcsjax_get_config_files` helper. These are static shell — carried verbatim.
- For each `CommandSpec`: walk `parser._actions`, classify each option, emit the
  `case "$prev"` block (enum / file / dir / configfile / threads) and the
  trailing option-name `compgen -W` list.
- Emits the registration **footer** (`complete -F <func> <name>` for every
  `command_name`).
- **Fails loudly** (raises) on: a parser import error, or an option that matches
  no classifier rule. No silent skips — integrity over a quietly-incomplete
  script. Runnable as `python -m xpcsjax.runtime.shell.generate_completion`
  (writes the file) and importable (`generate()` returns text).

Classification rules:

| Condition | Completion emitted |
|---|---|
| `dynamic_hints[flag] == "configfile"` | `compgen -W "$(_xpcsjax_get_config_files)"` |
| `dynamic_hints[flag] == "dir"` | `_filedir -d` |
| `dynamic_hints[flag] == "threads"` | nproc-based numeric list |
| `action.choices` set | `compgen -W "<choices>"` |
| `type` is `Path` (no hint) | `_filedir` (file) |
| takes a value (`float`/`int`/`str`), no choices | `return` (no value hint) |
| flag (`store_true`/`store_const`/`count`/`version`/`help`) | name only, in option list |

### 4. `completion.sh` (generated)

Becomes generator output, with a banner:
`# GENERATED by generate_completion.py — do not edit. Run 'make completion'.`

### 5. Tests — `tests/cli/test_completion_parity.py` (+ classifier units)

- **Parity gate:** `assert generate() == completion_sh.read_text()`; on failure,
  show a unified diff and the message "run `make completion`". Pure and fast —
  runs in the default suite so `make verify` catches drift pre-push.
- **Syntax smoke:** `bash -n completion.sh` exits 0.
- **Classifier units:** enum→list, `Path`→file, dir-hint→`_filedir -d`,
  flag→name-only, unclassifiable→raises.
- **Regression:** the three live defects are absent/present correctly in
  generated output (phantom contrast/offset gone; `--initial-beta` +
  `--no-multistart` present).

### 6. `make completion` target

Regenerates and writes `completion.sh`. One-line note added where post-install
references completion wiring.

### 7. Fish removal

`post_install` `--shell` choices → `("bash", "zsh")`; remove the fish-emitting
branch and any fish docs/tests. Verify no other module references fish
completion.

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
  mitigated by the import-light factory convention (GUI parser moved to
  `gui/cli_args.py`).
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
4. All six secondary commands expose `build_parser()`; their `main()` behavior is
   unchanged.
5. Fish is removed from `post_install`; `--shell` accepts `bash`/`zsh` only.
6. `bash -n completion.sh` passes; `make verify` is green.

## Follow-ups (later sub-projects)

- Sub-project 2 (CLI): with completion now drift-proof, CLI arg restructuring is
  safe — completion auto-follows.
- The `build_parser()` factories introduced here make per-command CLI unit tests
  straightforward in sub-project 2.
