# Shell Completion Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `xpcsjax/runtime/shell/completion.sh` a generated artifact derived from the CLI argparse parsers, guarded by a CI parity test so it can never silently drift again.

**Architecture:** Each CLI entry point exposes a `build_parser()` factory. A declarative `completion_spec` registry maps every console command to its parser factory, completion function name, and a tiny per-command `dynamic_hints` table. A generator introspects each parser and emits the full bash completion script; a pytest parity test regenerates and diffs against the committed file. The shipped script stays static bash (zsh via `bashcompinit`) — no Python/JAX runs on TAB.

**Tech Stack:** Python 3.12+, stdlib `argparse`, pytest, bash. Package-managed by `uv` (`uv run pytest`). Lint `ruff` (line-length 100, `E,F,W,I,B,UP,N`), type-check `mypy` (non-strict). Tests run via the `make` targets.

## Global Constraints

- **NLSQ-only package.** Do not add Bayesian/MCMC pathways or flags (project CLAUDE.md). This work touches only CLI plumbing.
- **No `from module import *`** (ruff `F`).
- **Float64 / JAX env** is set in `xpcsjax/__init__.py` only — do not add env mutation elsewhere.
- **Docstrings:** NumPy/numpydoc convention (ruff `D` gate; `xpcsjax/` only, tests exempt).
- **`make verify` is the pre-push gate** (lint + advisory mypy + smoke under `-x -n auto`). The new parity test must be in the default (non-gated) suite so it runs there.
- **CI mypy is a HARD gate** (`uv run mypy xpcsjax`). Run it before considering work done — `make verify` runs mypy only advisorily.
- **Commit cadence:** one commit per task (TDD: red → green → commit). Commit messages end with the `Co-Authored-By` trailer used in this repo.
- **`completion.sh` is generated** — after any parser change it must be regenerated via `make completion`; the parity test enforces this.
- **Fish scope:** remove fish only from the *completion* surface; leave fish **XLA activation** (`XLA_CONFIG_FISH`, `_install_xla_fish_activation`, `runtime/shell/__init__.py` fish helpers) untouched. `--shell` keeps `choices=["bash","zsh","fish"]` (fish remains valid for XLA-only installs).

---

## File Structure

**Create:**
- `xpcsjax/runtime/shell/completion_spec.py` — `Hint` type alias, `CommandSpec` dataclass, `COMMAND_SPECS` registry. Single source for what to generate + how to register.
- `xpcsjax/runtime/shell/generate_completion.py` — `classify_option()`, the bash emitters, `generate()`, and a `__main__` that writes `completion.sh`.
- `tests/cli/test_completion_parity.py` — parity gate + classifier units + regression assertions + `bash -n` smoke.

**Modify:**
- `xpcsjax/cli/args_parser.py` — add `build_parser` alias (uniform factory name).
- `xpcsjax/cli/config_generator.py` — expose `build_parser` (alias of existing `_build_parser`).
- `xpcsjax/cli/xla_config.py` — extract `build_parser()` from `main()`.
- `xpcsjax/post_install.py` — extract `build_parser()` from `main()`; remove fish *completion* install (keep fish XLA).
- `xpcsjax/uninstall_scripts.py` — extract `build_parser()` from `main()`.
- `xpcsjax/runtime/utils/system_validator.py` — extract `build_parser()` from `main()`.
- `xpcsjax/gui/app.py` — extract `build_parser()` from `_parse_cli_args()`.
- `xpcsjax/runtime/shell/completion.sh` — becomes generator output.
- `Makefile` — add `completion` target.
- `tests/cli/test_post_install.py` — update fish *completion* assertions (keep XLA).

**Per-command hint map (authoritative — used in Task 3):**

| Completion func | Commands | dynamic_hints |
|---|---|---|
| `_xpcsjax` | `xpcsjax`, `xj`, `xjexp`, `xjsim` | `--config`/`-c`→`configfile`, `--output`/`-o`→`dir`, `--threads`→`threads` |
| `_xpcsjax_config` | `xpcsjax-config`, `xj-config` | `--output`/`-o`→`file`, `--data`/`-d`→`file` |
| `_xpcsjax_config_xla` | `xpcsjax-config-xla`, `xj-config-xla` | `--threads`→`threads` |
| `_xpcsjax_post_install` | `xpcsjax-post-install`, `xj-post-install` | `--xla-mode`→`("auto","nlsq")` |
| `_xpcsjax_cleanup` | `xpcsjax-cleanup`, `xj-cleanup` | *(none)* |
| `_xpcsjax_validate` | `xpcsjax-validate`, `xj-validate` | *(none)* |
| `_xpcsjax_gui` | `xpcsjax-gui`, `xj-gui` | *(none)* |

Enums come from `choices` (e.g. `xpcsjax --mode`, `xpcsjax-config --mode`, `post-install --shell`, `xpcsjax --output-format`, `xpcsjax --plotting-backend`) — never hand-listed.

---

## Task 1: `build_parser()` factories for the four inline CLI/util parsers

**Files:**
- Modify: `xpcsjax/cli/args_parser.py` (add alias)
- Modify: `xpcsjax/cli/config_generator.py` (add alias)
- Modify: `xpcsjax/cli/xla_config.py:105-160` (extract)
- Modify: `xpcsjax/post_install.py:974-1015` (extract — parser construction only this task)
- Modify: `xpcsjax/uninstall_scripts.py:564-600` (extract)
- Modify: `xpcsjax/runtime/utils/system_validator.py:657-690` (extract)
- Test: `tests/cli/test_build_parser_factories.py` (create)

**Interfaces:**
- Produces: `build_parser() -> argparse.ArgumentParser` importable from each of:
  `xpcsjax.cli.args_parser`, `xpcsjax.cli.config_generator`, `xpcsjax.cli.xla_config`,
  `xpcsjax.post_install`, `xpcsjax.uninstall_scripts`, `xpcsjax.runtime.utils.system_validator`.
  Each returns the same parser the command's `main()` parses with.

- [ ] **Step 1: Write the failing test**

Create `tests/cli/test_build_parser_factories.py`:

```python
import argparse

import pytest


@pytest.mark.parametrize(
    "modpath",
    [
        "xpcsjax.cli.args_parser",
        "xpcsjax.cli.config_generator",
        "xpcsjax.cli.xla_config",
        "xpcsjax.post_install",
        "xpcsjax.uninstall_scripts",
        "xpcsjax.runtime.utils.system_validator",
    ],
)
def test_build_parser_returns_parser(modpath):
    mod = __import__(modpath, fromlist=["build_parser"])
    parser = mod.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    # Has at least one real option beyond -h
    opts = {s for a in parser._actions for s in a.option_strings}
    assert opts - {"-h", "--help"}


def test_xla_config_threads_roundtrip():
    from xpcsjax.cli import xla_config

    ns = xla_config.build_parser().parse_args(["--threads", "4"])
    assert ns.threads == 4


def test_post_install_shell_choices_preserved():
    from xpcsjax import post_install

    shell_action = next(
        a for a in post_install.build_parser()._actions if "--shell" in a.option_strings
    )
    assert shell_action.choices == ["bash", "zsh", "fish"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_build_parser_factories.py -q`
Expected: FAIL — `AttributeError: module 'xpcsjax.cli.xla_config' has no attribute 'build_parser'` (and same for the other inline modules).

- [ ] **Step 3: Add factory aliases to the two modules that already have a builder**

In `xpcsjax/cli/args_parser.py`, after `create_parser` is defined, add an alias and export it:

```python
build_parser = create_parser  # uniform factory name used by completion_spec
```
Update `__all__` to include `"build_parser"`.

In `xpcsjax/cli/config_generator.py`, after `_build_parser` is defined, add:

```python
def build_parser() -> argparse.ArgumentParser:
    """Public factory alias for the config-generator parser."""
    return _build_parser()
```

- [ ] **Step 4: Extract `build_parser()` in the four inline modules**

For each of `xpcsjax/cli/xla_config.py`, `xpcsjax/post_install.py`,
`xpcsjax/uninstall_scripts.py`, `xpcsjax/runtime/utils/system_validator.py`:
move the parser-construction block (from `parser = argparse.ArgumentParser(...)`
through the final `parser.add_argument(...)`) out of `main()` into a new
module-level function, and have `main()` call it. Pattern (xla_config shown):

```python
def build_parser() -> argparse.ArgumentParser:
    """Build the xpcsjax-config-xla argument parser."""
    parser = argparse.ArgumentParser(...)   # the existing block, verbatim
    parser.add_argument("--threads", ...)
    parser.add_argument("--no-x64", ...)
    parser.add_argument("--debug", ...)
    parser.add_argument("--info", ...)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # ... rest of main() unchanged ...
```

For `post_install.py`: extract ONLY the `argparse.ArgumentParser(...)` +
`add_argument(...)` block into `build_parser()`; leave the `args = parser.parse_args(argv)`
call and ALL downstream logic (the `_validate_xla_mode` block, dispatch) in `main()`.
`main()` becomes `parser = build_parser(); args = parser.parse_args(argv)`.

Add `build_parser` to each module's `__all__` if it defines one.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_build_parser_factories.py -q`
Expected: PASS (all parametrized cases + roundtrip + shell-choices).

- [ ] **Step 6: Verify `main()` behavior is unchanged**

Run: `uv run pytest tests/cli/test_post_install.py tests/cli/test_uninstall_scripts.py -q`
Expected: PASS (no behavior change from the extraction).

- [ ] **Step 7: Lint + type-check the touched files**

Run: `uv run ruff check xpcsjax/cli/args_parser.py xpcsjax/cli/config_generator.py xpcsjax/cli/xla_config.py xpcsjax/post_install.py xpcsjax/uninstall_scripts.py xpcsjax/runtime/utils/system_validator.py tests/cli/test_build_parser_factories.py`
Run: `uv run mypy xpcsjax/cli/xla_config.py xpcsjax/post_install.py`
Expected: clean (no new errors).

- [ ] **Step 8: Commit**

```bash
git add xpcsjax/cli/args_parser.py xpcsjax/cli/config_generator.py xpcsjax/cli/xla_config.py xpcsjax/post_install.py xpcsjax/uninstall_scripts.py xpcsjax/runtime/utils/system_validator.py tests/cli/test_build_parser_factories.py
git commit -m "refactor(cli): expose build_parser() factory on every CLI entry point

Pure extraction; main() behavior unchanged. Enables parser introspection
for completion generation.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `build_parser()` factory for the GUI launcher

**Files:**
- Modify: `xpcsjax/gui/app.py:45-66` (extract parser from `_parse_cli_args`)
- Test: `tests/cli/test_build_parser_factories.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces: `xpcsjax.gui.app.build_parser() -> argparse.ArgumentParser` (the `--version`/`--help` parser). `gui/app.py` must stay import-light (no top-level Qt/JAX import).

- [ ] **Step 1: Write the failing test**

Append to `tests/cli/test_build_parser_factories.py`:

```python
def test_gui_build_parser_import_light():
    import sys

    sys.modules.pop("PySide6", None)
    from xpcsjax.gui import app

    parser = app.build_parser()
    import argparse

    assert isinstance(parser, argparse.ArgumentParser)
    # Importing the GUI module must NOT pull in PySide6.
    assert "PySide6" not in sys.modules
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_build_parser_factories.py::test_gui_build_parser_import_light -q`
Expected: FAIL — `AttributeError: module 'xpcsjax.gui.app' has no attribute 'build_parser'`.

- [ ] **Step 3: Extract the GUI parser factory**

In `xpcsjax/gui/app.py`, move the `argparse.ArgumentParser(...)` construction out of
`_parse_cli_args()` into a module-level `build_parser()`, and call it from
`_parse_cli_args()`:

```python
def build_parser() -> argparse.ArgumentParser:
    """Build the xpcsjax-gui launcher parser (--version / --help only)."""
    parser = argparse.ArgumentParser(
        prog="xpcsjax-gui",
        description="Launch the xpcsjax analysis workbench (PySide6 GUI).",
    )
    # ... existing add_argument(...) calls, verbatim ...
    return parser


def _parse_cli_args(argv: list[str]) -> list[str]:
    parser = build_parser()
    # ... existing parse_known_args / passthrough logic, unchanged ...
```

Keep `import argparse` at module top (stdlib, import-light). Do not move any Qt import.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_build_parser_factories.py -q`
Expected: PASS (all cases including the new import-light one).

- [ ] **Step 5: Verify GUI launcher arg parsing unchanged**

Run: `uv run pytest tests/gui -q -k "app or launch or cli" `
Expected: PASS (or "no tests ran" if none match — acceptable; the import-light test covers the change).

- [ ] **Step 6: Lint**

Run: `uv run ruff check xpcsjax/gui/app.py tests/cli/test_build_parser_factories.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/gui/app.py tests/cli/test_build_parser_factories.py
git commit -m "refactor(gui): extract build_parser() factory; keep app.py import-light

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Command registry — `completion_spec.py`

**Files:**
- Create: `xpcsjax/runtime/shell/completion_spec.py`
- Test: `tests/cli/test_completion_spec.py` (create)

**Interfaces:**
- Consumes: the seven `build_parser` factories from Tasks 1–2.
- Produces:
  - `Hint = str | tuple[str, ...]` (a hint kind `"configfile"|"file"|"dir"|"threads"` or a literal word tuple).
  - `CommandSpec` dataclass with fields `completion_func: str`, `command_names: tuple[str, ...]`, `parser_factory: Callable[[], argparse.ArgumentParser]`, `dynamic_hints: dict[str, Hint]`.
  - `COMMAND_SPECS: list[CommandSpec]` covering all seven completion functions.

- [ ] **Step 1: Write the failing test**

Create `tests/cli/test_completion_spec.py`:

```python
from xpcsjax.runtime.shell.completion_spec import COMMAND_SPECS


def test_every_console_script_is_covered():
    names = {n for spec in COMMAND_SPECS for n in spec.command_names}
    expected = {
        "xpcsjax", "xj", "xjexp", "xjsim",
        "xpcsjax-config", "xj-config",
        "xpcsjax-config-xla", "xj-config-xla",
        "xpcsjax-post-install", "xj-post-install",
        "xpcsjax-cleanup", "xj-cleanup",
        "xpcsjax-validate", "xj-validate",
        "xpcsjax-gui", "xj-gui",
    }
    assert names == expected


def test_factories_are_callable_and_unique_funcs():
    funcs = [s.completion_func for s in COMMAND_SPECS]
    assert len(funcs) == len(set(funcs))  # no duplicate function names
    for spec in COMMAND_SPECS:
        parser = spec.parser_factory()
        assert parser is not None


def test_hint_flags_exist_on_their_parser():
    for spec in COMMAND_SPECS:
        opts = {s for a in spec.parser_factory()._actions for s in a.option_strings}
        for flag in spec.dynamic_hints:
            assert flag in opts, f"{spec.completion_func}: hint flag {flag} not in parser"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_completion_spec.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'xpcsjax.runtime.shell.completion_spec'`.

- [ ] **Step 3: Write the registry**

Create `xpcsjax/runtime/shell/completion_spec.py`:

```python
"""Declarative registry mapping console commands to their completion data.

Single source of truth for the completion generator: which parser backs each
command, what completion-function name it binds to, and the per-command dynamic
hints that argparse cannot express (file/dir intent, free-form suggestions).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field

from xpcsjax.cli import args_parser, config_generator, xla_config
from xpcsjax import post_install, uninstall_scripts
from xpcsjax.gui import app as gui_app
from xpcsjax.runtime.utils import system_validator

# A hint is either a named kind or an explicit tuple of literal completion words.
Hint = str | tuple[str, ...]


@dataclass(frozen=True)
class CommandSpec:
    """One completion function and the commands bound to it."""

    completion_func: str
    command_names: tuple[str, ...]
    parser_factory: Callable[[], argparse.ArgumentParser]
    dynamic_hints: dict[str, Hint] = field(default_factory=dict)


COMMAND_SPECS: list[CommandSpec] = [
    CommandSpec(
        completion_func="_xpcsjax",
        command_names=("xpcsjax", "xj", "xjexp", "xjsim"),
        parser_factory=args_parser.build_parser,
        dynamic_hints={
            "--config": "configfile", "-c": "configfile",
            "--output": "dir", "-o": "dir",
            "--threads": "threads",
        },
    ),
    CommandSpec(
        completion_func="_xpcsjax_config",
        command_names=("xpcsjax-config", "xj-config"),
        parser_factory=config_generator.build_parser,
        dynamic_hints={
            "--output": "file", "-o": "file",
            "--data": "file", "-d": "file",
        },
    ),
    CommandSpec(
        completion_func="_xpcsjax_config_xla",
        command_names=("xpcsjax-config-xla", "xj-config-xla"),
        parser_factory=xla_config.build_parser,
        dynamic_hints={"--threads": "threads"},
    ),
    CommandSpec(
        completion_func="_xpcsjax_post_install",
        command_names=("xpcsjax-post-install", "xj-post-install"),
        parser_factory=post_install.build_parser,
        dynamic_hints={"--xla-mode": ("auto", "nlsq")},
    ),
    CommandSpec(
        completion_func="_xpcsjax_cleanup",
        command_names=("xpcsjax-cleanup", "xj-cleanup"),
        parser_factory=uninstall_scripts.build_parser,
    ),
    CommandSpec(
        completion_func="_xpcsjax_validate",
        command_names=("xpcsjax-validate", "xj-validate"),
        parser_factory=system_validator.build_parser,
    ),
    CommandSpec(
        completion_func="_xpcsjax_gui",
        command_names=("xpcsjax-gui", "xj-gui"),
        parser_factory=gui_app.build_parser,
    ),
]

__all__ = ["Hint", "CommandSpec", "COMMAND_SPECS"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_completion_spec.py -q`
Expected: PASS. If `test_hint_flags_exist_on_their_parser` fails, the hint table has a typo'd flag — fix the registry (not the test).

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check xpcsjax/runtime/shell/completion_spec.py tests/cli/test_completion_spec.py`
Run: `uv run mypy xpcsjax/runtime/shell/completion_spec.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/runtime/shell/completion_spec.py tests/cli/test_completion_spec.py
git commit -m "feat(completion): declarative command registry for completion generation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Option classifier — `generate_completion.classify_option`

**Files:**
- Create: `xpcsjax/runtime/shell/generate_completion.py` (classifier only this task)
- Test: `tests/cli/test_classify_option.py` (create)

**Interfaces:**
- Consumes: `Hint` from `completion_spec`.
- Produces: `classify_option(action: argparse.Action, hints: dict[str, Hint]) -> Completion | None` where `Completion` is a `dataclass(kind: str, payload: str = "")` describing the value completion for an option that takes a value; returns `None` for pure flags (zero-arg). Raises `ValueError` for an option that takes a value but matches no rule. `kind` ∈ {`"configfile"`, `"file"`, `"dir"`, `"threads"`, `"words"`, `"choices"`, `"none"`}.

- [ ] **Step 1: Write the failing test**

Create `tests/cli/test_classify_option.py`:

```python
import argparse

import pytest

from xpcsjax.runtime.shell.generate_completion import Completion, classify_option


def _action(*flags, **kw):
    p = argparse.ArgumentParser()
    return p.add_argument(*flags, **kw)


def test_store_true_is_flag_returns_none():
    assert classify_option(_action("--plot", action="store_true"), {}) is None


def test_store_false_is_flag_returns_none():
    # --no-plot is store_false: zero-arg, must NOT raise (the BLOCKER fix).
    assert classify_option(_action("--no-plot", action="store_false"), {}) is None


def test_count_action_is_flag_returns_none():
    assert classify_option(_action("-v", "--verbose", action="count"), {}) is None


def test_choices_become_choices_completion():
    c = classify_option(_action("--mode", choices=["a", "b"]), {})
    assert c == Completion(kind="choices", payload="a b")


def test_path_type_defaults_to_file():
    from pathlib import Path

    c = classify_option(_action("--out", type=Path), {})
    assert c == Completion(kind="file")


def test_dir_hint_overrides_path():
    from pathlib import Path

    c = classify_option(_action("--out", type=Path), {"--out": "dir"})
    assert c == Completion(kind="dir")


def test_str_path_needs_explicit_file_hint():
    # type=str with no hint -> plain value, no completion.
    assert classify_option(_action("--data", type=str), {}) == Completion(kind="none")
    # with file hint -> file completion.
    assert classify_option(_action("--data", type=str), {"--data": "file"}) == Completion(
        kind="file"
    )


def test_literal_word_hint():
    c = classify_option(_action("--xla-mode", type=str), {"--xla-mode": ("auto", "nlsq")})
    assert c == Completion(kind="words", payload="auto nlsq")


def test_threads_hint():
    c = classify_option(_action("--threads", type=int), {"--threads": "threads"})
    assert c.kind == "threads"


def test_plain_value_no_hint_no_choices_is_none_kind():
    assert classify_option(_action("--tol", type=float), {}) == Completion(kind="none")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_classify_option.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'xpcsjax.runtime.shell.generate_completion'`.

- [ ] **Step 3: Implement the classifier**

Create `xpcsjax/runtime/shell/generate_completion.py` (classifier portion):

```python
"""Generate xpcsjax/runtime/shell/completion.sh from the CLI parsers.

The completion script is a DERIVED artifact: this module is the single writer.
Run ``python -m xpcsjax.runtime.shell.generate_completion`` to regenerate, or
``make completion``. A parity test fails CI if the committed file drifts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from xpcsjax.runtime.shell.completion_spec import COMMAND_SPECS, CommandSpec, Hint


@dataclass(frozen=True)
class Completion:
    """How to complete the value of one option. kind drives the emitted bash."""

    kind: str  # configfile | file | dir | threads | words | choices | none
    payload: str = ""


def classify_option(action: argparse.Action, hints: dict[str, Hint]) -> Completion | None:
    """Classify one option into a value-completion descriptor.

    Returns ``None`` for a zero-argument flag (``nargs == 0``). Raises
    ``ValueError`` only for a value-taking option that matches no rule — by
    construction that cannot happen (the final fallback is ``none``), so the
    raise guards against future argparse action types.
    """
    # Zero-argument actions are flags: store_true/store_false/store_const/count/
    # version/help all set nargs == 0. Classify on nargs, never an allow-list.
    if action.nargs == 0:
        return None

    # Per-command dynamic hints take precedence over type-based inference.
    flag = action.option_strings[0] if action.option_strings else ""
    hint = _lookup_hint(action.option_strings, hints)
    if hint is not None:
        if isinstance(hint, tuple):
            return Completion(kind="words", payload=" ".join(hint))
        if hint in {"configfile", "file", "dir", "threads"}:
            return Completion(kind=hint)
        raise ValueError(f"unknown hint {hint!r} for {flag}")

    if action.choices:
        return Completion(kind="choices", payload=" ".join(str(c) for c in action.choices))

    if action.type is Path:
        return Completion(kind="file")

    # Takes a value but no completion source: emit nothing.
    return Completion(kind="none")


def _lookup_hint(option_strings: list[str], hints: dict[str, Hint]) -> Hint | None:
    for s in option_strings:
        if s in hints:
            return hints[s]
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_classify_option.py -q`
Expected: PASS (all 10 cases, including `store_false`).

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check xpcsjax/runtime/shell/generate_completion.py tests/cli/test_classify_option.py`
Run: `uv run mypy xpcsjax/runtime/shell/generate_completion.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/runtime/shell/generate_completion.py tests/cli/test_classify_option.py
git commit -m "feat(completion): option classifier (nargs==0 flag rule covers store_false)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Generator assembly + regenerate `completion.sh` + parity gate

**Files:**
- Modify: `xpcsjax/runtime/shell/generate_completion.py` (add emitters + `generate()` + `__main__`)
- Modify: `xpcsjax/runtime/shell/completion.sh` (regenerated)
- Test: `tests/cli/test_completion_parity.py` (create)

**Interfaces:**
- Consumes: `Completion`, `classify_option`, `COMMAND_SPECS`.
- Produces: `generate() -> str` (full script text); `COMPLETION_SH_PATH: Path`; running the module as `__main__` writes `COMPLETION_SH_PATH`.

- [ ] **Step 1: Write the failing test**

Create `tests/cli/test_completion_parity.py`:

```python
import shutil
import subprocess

import pytest

from xpcsjax.runtime.shell.generate_completion import COMPLETION_SH_PATH, generate


def test_committed_completion_matches_generator():
    generated = generate()
    committed = COMPLETION_SH_PATH.read_text(encoding="utf-8")
    if generated != committed:
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                committed.splitlines(), generated.splitlines(),
                "committed completion.sh", "generated", lineterm="",
            )
        )
        pytest.fail(f"completion.sh is stale — run `make completion`.\n{diff}")


def test_generated_script_is_valid_bash():
    assert shutil.which("bash"), "bash required"
    r = subprocess.run(
        ["bash", "-n", str(COMPLETION_SH_PATH)], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


def test_three_drift_defects_resolved():
    text = generate()
    # Phantom flags gone:
    assert "--initial-contrast" not in text
    assert "--initial-offset" not in text
    # Previously-missing flags present:
    assert "--initial-beta" in text
    assert "--no-multistart" in text


def test_config_data_is_file_completed_and_validate_is_not():
    text = generate()
    # --data gets _filedir (file hint); --validate is a flag (store_true), so it
    # must not appear in a `--validate)` value-completion case arm.
    assert "--data|-d" in text or "--data)" in text
    assert "--validate)" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_completion_parity.py -q`
Expected: FAIL — `ImportError: cannot import name 'generate'` (and `COMPLETION_SH_PATH`).

- [ ] **Step 3: Implement the emitters + `generate()`**

Append to `xpcsjax/runtime/shell/generate_completion.py`:

```python
COMPLETION_SH_PATH = Path(__file__).with_name("completion.sh")

_BANNER = (
    "#!/bin/bash\n"
    "# GENERATED by generate_completion.py — DO NOT EDIT.\n"
    "# Regenerate with: make completion  (or python -m "
    "xpcsjax.runtime.shell.generate_completion)\n"
    "# Bash/zsh completion for xpcsjax CLI commands.\n"
)

# Static preamble: cache vars + _init_completion/_filedir fallback (bare
# conda/mamba shells) + cached config-file discovery. Copied verbatim from the
# historical completion.sh; byte-exact parity depends on this block being fixed.
_PREAMBLE = r'''
_XPCSJAX_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/xpcsjax"
_XPCSJAX_CACHE_TTL=300  # 5 minutes

_xpcsjax_ensure_cache() {
    [[ -d "$_XPCSJAX_CACHE_DIR" ]] || mkdir -p "$_XPCSJAX_CACHE_DIR"
}

if ! type _init_completion &>/dev/null; then
    _init_completion() {
        COMPREPLY=()
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
        words=("${COMP_WORDS[@]}")
        cword=$COMP_CWORD
    }

    if ! type _filedir &>/dev/null; then
        _filedir() {
            if [[ "$1" == "-d" ]]; then
                mapfile -t COMPREPLY < <(compgen -d -- "${cur}")
            else
                mapfile -t COMPREPLY < <(compgen -f -- "${cur}")
            fi
        }
    fi
fi

_xpcsjax_get_config_files() {
    _xpcsjax_ensure_cache
    local cache_file="$_XPCSJAX_CACHE_DIR/config_files"
    local now
    now=$(date +%s)

    if [[ -f "$cache_file" ]]; then
        local cache_time
        cache_time=$(stat -f %m "$cache_file" 2>/dev/null || stat -c %Y "$cache_file" 2>/dev/null)
        if [[ $((now - cache_time)) -lt $_XPCSJAX_CACHE_TTL ]]; then
            cat "$cache_file"
            return
        fi
    fi

    {
        find . -maxdepth 2 \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null
        [[ -d "config" ]] && find config \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null
        [[ -d "configs" ]] && find configs \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null
    } | sort -u | tee "$cache_file"
}
'''


def _all_option_strings(parser: argparse.ArgumentParser) -> list[str]:
    """All option strings in declaration order (includes -h/--help)."""
    out: list[str] = []
    for action in parser._actions:
        out.extend(action.option_strings)
    return out


def _value_completion_bash(comp: Completion) -> str:
    """Bash body that fills COMPREPLY for an option that takes a value."""
    if comp.kind == "configfile":
        return 'mapfile -t COMPREPLY < <(compgen -W "$(_xpcsjax_get_config_files)" -- "${cur}")'
    if comp.kind == "dir":
        return "_filedir -d"
    if comp.kind == "file":
        return "_filedir"
    if comp.kind == "words" or comp.kind == "choices":
        return f'mapfile -t COMPREPLY < <(compgen -W "{comp.payload}" -- "${{cur}}")'
    if comp.kind == "threads":
        return (
            "local cpu_count\n"
            "        cpu_count=$(nproc 2>/dev/null || echo 4)\n"
            '        mapfile -t COMPREPLY < <(compgen -W "1 2 4 8 ${cpu_count}" -- "${cur}")'
        )
    return ""  # kind == "none": no value hint


def _emit_function(spec: CommandSpec) -> str:
    parser = spec.parser_factory()
    opts = " ".join(_all_option_strings(parser))

    # Build per-prev value-completion case arms (skip kind == "none").
    arms: list[str] = []
    for action in parser._actions:
        if not action.option_strings:
            continue
        comp = classify_option(action, spec.dynamic_hints)
        if comp is None or comp.kind == "none":
            continue
        pattern = "|".join(action.option_strings)
        body = _value_completion_bash(comp)
        arms.append(f"        {pattern})\n            {body}\n            return\n            ;;")

    case_block = ""
    if arms:
        case_block = '    case "$prev" in\n' + "\n".join(arms) + "\n    esac\n\n"

    return (
        f"{spec.completion_func}() {{\n"
        "    local cur prev words cword\n"
        "    _init_completion -s || return\n\n"
        f'    local opts="{opts}"\n\n'
        f"{case_block}"
        '    if [[ "$cur" == -* ]]; then\n'
        '        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")\n'
        "        return\n"
        "    fi\n"
        "}\n"
    )


def _emit_footer() -> str:
    lines = ["# Register completions"]
    for spec in COMMAND_SPECS:
        for name in spec.command_names:
            lines.append(f"complete -F {spec.completion_func} {name}")
    return "\n".join(lines) + "\n"


def generate() -> str:
    """Return the full completion.sh text."""
    parts = [_BANNER, _PREAMBLE, ""]
    for spec in COMMAND_SPECS:
        parts.append(_emit_function(spec))
        parts.append("")
    parts.append(_emit_footer())
    return "\n".join(parts)


def main() -> None:
    COMPLETION_SH_PATH.write_text(generate(), encoding="utf-8")
    print(f"wrote {COMPLETION_SH_PATH}")


if __name__ == "__main__":
    main()
```

Note: the `_xpcsjax` config-file-discovery on bare word (the historical
`compgen -W "$(_xpcsjax_get_config_files) ${all_opts}"` for non-dash `cur`) is
intentionally dropped — value completion for `--config` already covers config
discovery, and bare-word command completion adds noise. This is a deliberate
behavior simplification, not drift; the parity test pins the new output.

- [ ] **Step 4: Regenerate `completion.sh`**

Run: `uv run python -m xpcsjax.runtime.shell.generate_completion`
Expected: `wrote .../completion.sh`. Inspect the diff: `git diff -- xpcsjax/runtime/shell/completion.sh` and confirm the three defects are fixed (no `--initial-contrast`/`--initial-offset`; `--initial-beta` and `--no-multistart` present) and `--validate` no longer has a value case-arm.

- [ ] **Step 5: Run the parity + smoke + regression tests**

Run: `uv run pytest tests/cli/test_completion_parity.py -q`
Expected: PASS (committed file now equals generator output; `bash -n` clean; defects resolved).

- [ ] **Step 6: Lint + type-check**

Run: `uv run ruff check xpcsjax/runtime/shell/generate_completion.py tests/cli/test_completion_parity.py`
Run: `uv run mypy xpcsjax/runtime/shell/generate_completion.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/runtime/shell/generate_completion.py xpcsjax/runtime/shell/completion.sh tests/cli/test_completion_parity.py
git commit -m "feat(completion): generate completion.sh from parsers + parity gate

Resolves 3 live drift defects (phantom contrast/offset; missing beta,
no-multistart). completion.sh is now a derived artifact; the parity test
fails CI on drift.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Drop fish from the completion surface (keep fish XLA)

**Files:**
- Modify: `xpcsjax/post_install.py` (completion-install path only)
- Modify: `tests/cli/test_post_install.py` (completion-fish assertions)

**Interfaces:**
- Consumes: nothing new.
- Produces: `install_shell_completion(...)` no longer installs a fish *completion* file; fish XLA activation paths unchanged.

- [ ] **Step 1: Identify the fish completion vs fish XLA references**

Run: `grep -n "fish" xpcsjax/post_install.py`
Classify each hit as **completion** (e.g. `install_fish_completion`, `_install_completion_fish_activation`, completion-path dispatch) or **XLA** (`_install_xla_fish_activation`, `XLA_CONFIG_FISH`, `get_xla_config_source_path("fish")`). Only completion hits are in scope.

- [ ] **Step 2: Write/adjust the failing test**

In `tests/cli/test_post_install.py`, add a test asserting fish completion is not installed while fish XLA still is:

```python
def test_fish_gets_no_completion_but_keeps_xla(tmp_path, monkeypatch):
    import xpcsjax.post_install as pi

    # install_shell_completion for fish should be a no-op / unsupported now.
    result = pi.install_shell_completion("fish", verbose=False)
    assert result is False or result is None  # completion not installed for fish
    # XLA activation for fish remains available.
    assert pi.get_xla_config_source_path("fish").name == "xla_config.fish"
```

Adjust any existing test that asserts fish *completion* is installed (search for `install_fish_completion` / fish completion assertions) to expect it removed. Leave fish XLA tests intact.

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_post_install.py -q -k fish`
Expected: FAIL (current code still installs fish completion).

- [ ] **Step 4: Remove fish from the completion path**

In `install_shell_completion()` (and its dispatch), drop the `fish` branch so
fish requests skip completion installation (return `False` with a one-line
log: "shell completion not provided for fish; bash/zsh only"). Delete
`install_fish_completion` / `_install_completion_fish_activation` if they are
solely for completion. Do NOT touch `_install_xla_fish_activation`,
`XLA_CONFIG_FISH`, `get_xla_config_source_path`, or `detect_shell_type`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/cli/test_post_install.py -q`
Expected: PASS.

- [ ] **Step 6: Confirm completion.sh is unaffected**

Run: `uv run python -m xpcsjax.runtime.shell.generate_completion && git diff --exit-code -- xpcsjax/runtime/shell/completion.sh`
Expected: exit 0 (no change — `--shell` keeps `choices=["bash","zsh","fish"]`, so completion output is identical). If it changed, investigate before proceeding.

- [ ] **Step 7: Lint + commit**

Run: `uv run ruff check xpcsjax/post_install.py tests/cli/test_post_install.py`

```bash
git add xpcsjax/post_install.py tests/cli/test_post_install.py
git commit -m "refactor(post-install): drop fish completion install (keep fish XLA)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: `make completion` target + docs note + full verify

**Files:**
- Modify: `Makefile`
- Modify: `xpcsjax/post_install.py` docstring/help note (one line) or `README` where completion wiring is described.

**Interfaces:** none (developer-facing tooling).

- [ ] **Step 1: Add the make target**

In `Makefile`, add (match the existing `uv`-detection idiom used by other targets):

```makefile
.PHONY: completion
completion:  ## Regenerate the shell completion script from the CLI parsers
	$(UVRUN) python -m xpcsjax.runtime.shell.generate_completion
```

(Use whatever variable the Makefile already uses for `uv run`, e.g. `$(UVRUN)` / `uv run`. Match the surrounding style.)

- [ ] **Step 2: Verify the target works and is idempotent**

Run: `make completion`
Expected: `wrote .../completion.sh`.
Run: `git diff --exit-code -- xpcsjax/runtime/shell/completion.sh`
Expected: exit 0 (already up to date from Task 5).

- [ ] **Step 3: Add a one-line maintenance note**

Where completion installation/wiring is documented (post_install module docstring
or README completion section), add: "`completion.sh` is generated — after any CLI
argument change, run `make completion` (a parity test enforces this)."

- [ ] **Step 4: Run the full CLI + runtime test scope**

Run: `uv run pytest tests/cli tests/runtime -q`
Expected: PASS (all completion, post-install, uninstall, runtime-shell tests green).

- [ ] **Step 5: Run the real gates**

Run: `uv run mypy xpcsjax`   (the HARD CI gate — must be clean for the touched modules)
Run: `make verify`
Expected: green (lint + advisory mypy + smoke; the parity test runs in the default suite).

- [ ] **Step 6: Commit**

```bash
git add Makefile xpcsjax/post_install.py
git commit -m "build(completion): add 'make completion' target + regeneration note

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** Generation (Tasks 4–5), CI parity gate (Task 5), all-commands ownership via factories (Tasks 1–3), `store_false` BLOCKER (Task 4 classifier), per-command `file`/`dir`/literal-list hints incl. `--data` str-path and `--xla-mode` (Tasks 3–5), GUI import-light extraction (Task 2), fish-completion-only removal keeping XLA (Task 6), `make completion` + docs (Task 7), `--validate`/`--data` regressions (Task 5 tests), existing adjacent test updates (Task 6). All spec acceptance criteria map to a task.

**Placeholder scan:** No TBD/TODO; every code step shows real code; extraction steps name exact files+line ranges and show the resulting structure.

**Type consistency:** `build_parser` name uniform across Tasks 1–3; `Hint`/`CommandSpec`/`COMMAND_SPECS` defined in Task 3 and consumed unchanged in Tasks 4–5; `Completion(kind, payload)` and `classify_option(action, hints)` defined in Task 4 and used consistently in Task 5; `COMPLETION_SH_PATH`/`generate()` defined in Task 5 and used by its tests.

**Known judgment calls flagged for the implementer:** (a) the historical bare-word config-discovery in `_xpcsjax` is intentionally dropped (Task 5 note) — if byte-parity reveals it's wanted, add it to the emitter and regenerate; (b) Task 6 must classify each `post_install` fish reference as completion vs XLA before deleting — only completion is in scope.
