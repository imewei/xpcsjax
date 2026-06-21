# CLI Decomposition & Error Hardening — Design

**Date:** 2026-06-21
**Status:** Approved (brainstorming) — pending implementation plan
**Sub-project:** 2 of 3 (GUI / CLI / shell completion optimization effort)

## Problem

Two CLI modules have grown into god-modules mixing several unrelated concerns:

- `xpcsjax/cli/plot_dispatch.py` (682 LOC) — backend selection, experimental
  plots, simulated plots (the only JAX-using plot code), post-fit plots, and the
  dispatch orchestrator, all in one file.
- `xpcsjax/cli/config_generator.py` (564 LOC) — YAML template/generation logic
  intermixed with the `xpcsjax-config` console entry point.

Separately, `xpcsjax/cli/config_handling.py` (306 LOC) has three weak error
paths that produce cryptic failures or silently discard user input.

## Goal

Decompose the two god-modules into focused, independently-testable units and
make config-loading errors actionable — **strictly behavior-preserving**, with
all public contracts unchanged so sub-project 1's completion generation and the
GUI (sub-project 3, untouched here) keep working.

## Non-goals (YAGNI)

- **Startup performance / lazy JAX import.** Verified already achieved:
  `xpcsjax --help`, `xpcsjax-config`, and completion generation are all
  JAX-free (`jax not in sys.modules`). `main.py` lazy-imports the parser inside
  `main()`; `__init__.py` does not eagerly import JAX. Only a real fit dispatch
  pays the ~1.5 s JAX import, which it legitimately needs. The single cleanup is
  fixing the stale `main.py:26` comment that claims `__init__` "eagerly imports
  JAX". No code change for perf.
- `args_parser.py` (442 LOC) — cohesive single argparse builder, JAX-free;
  splitting argparse is awkward. Leave as-is.
- `optimization_runner.py` (330 LOC) — type-cohesive around `OptimizationResult`.
  Leave as-is.
- Anything GUI (sub-project 3).

## Contract constraints (verified importers — must not break)

| Symbol | External importer | Constraint |
|---|---|---|
| `plot_dispatch.dispatch_plots` | `cli/__init__.py` lazy export; `commands.py:24` | Stays in `plot_dispatch.py` |
| `plot_dispatch.resolve_plots_dir` | `tests/cli/test_output_resolution.py:18` | Moves to `plot_backend`; update that import |
| `plot_dispatch._generate_post_fit_plots` | `tests/cli/test_output_resolution.py:110,144` | Moves to `plot_families/postfit`; update test |
| `plot_dispatch._plot_simulated_from_config`, `._evaluate_model_c2` | `tests/cli/test_simulated_data_grid.py` (calls + monkeypatch) | Move to `plot_families/simulated`; update test patch target |
| `plot_dispatch` (module) | `tests/cli/test_plot_dispatch_logging.py` | Keep module; update any moved-symbol refs |
| `config_generator.generate_config` | `gui/views/main_window.py:459`, `config_dialogs.py`; `tests/cli/test_config_generator_yaml.py:12` | **Re-exported** from `config_generator` (facade) — GUI untouched |
| `config_generator.main` | `cli/__init__.py` lazy export `config_main` | Stays in `config_generator.py` |
| `config_generator.build_parser` | `completion_spec.py:60`; `tests/cli/test_build_parser_factories.py` | Stays in `config_generator.py` |

## Components

### 1. `plot_dispatch.py` (682) → 5 focused units

- **`xpcsjax/cli/plot_backend.py`** (new) — backend/dir resolution:
  `_current_run_id`, `resolve_plots_dir`, `should_use_datashader`.
- **`xpcsjax/cli/plot_families/`** (new subpackage):
  - `experimental.py` — `_plot_experimental_data`.
  - `simulated.py` — `resolve_phi_angles_for_sim`, `_plot_simulated_from_config`,
    `_evaluate_model_c2`. **This is the only JAX-importing plot code** — the
    `import jax` (currently `plot_dispatch.py:381`) lives here, function-local or
    module-top within this file only.
  - `postfit.py` — `_generate_post_fit_plots`, `_save_fit_comparison_only`.
    (Note: `service/plots.py` is the existing argparse-free worker-side core that
    `_generate_post_fit_plots` mirrors/delegates to — preserve that relationship,
    do not duplicate it.)
- **`plot_dispatch.py`** (refactored) — thin orchestrator: keeps the public
  `dispatch_plots()` with its exact signature, imports from the families/backend
  and delegates. Target ≈ 100 LOC.

Test updates required (behavior-preserving, same assertions): repoint
`test_output_resolution.py` (`resolve_plots_dir` → `plot_backend`;
`_generate_post_fit_plots` → `plot_families.postfit`), `test_simulated_data_grid.py`
(`_plot_simulated_from_config`/`_evaluate_model_c2` → `plot_families.simulated`,
including the monkeypatch target), and any moved-symbol reference in
`test_plot_dispatch_logging.py`.

### 2. `config_generator.py` (564) → split with a facade

- **`xpcsjax/cli/config_template.py`** (new) — template/generation/validation:
  `get_template_path`, `generate_config`, `show_template`, `validate_config`,
  `interactive_builder`, `_prompt`, and the `_MODE_TO_TEMPLATE` map.
- **`config_generator.py`** (refactored) — keeps `build_parser`, `main`, and
  **re-exports** the template API (`generate_config`, etc.) from
  `config_template` so `config_generator.generate_config` stays importable for
  the GUI and tests. This facade is deliberate (preserves the public surface
  without touching GUI sub-project 3). `config_generator` stays JAX-free
  (verified) — importing it imports `config_template` → `ConfigManager`, which is
  already the case today.

### 3. `config_handling.py` (306) — harden 3 error paths (no split)

- **L108** bare `ConfigManager(yaml_path)` with no file context → wrap in
  try/except, re-raise with `f"Failed to load config from {yaml_path}: {e}"`.
- **L149** silent `except AttributeError: pass` around `_normalize_analysis_mode`
  → log a warning and raise `ValueError` with the offending value, OR narrow the
  except to the genuinely-expected case with a comment (decide during
  implementation by reading what raises). No silent pass.
- **L158** silent `if not isinstance(out, dict): out = {}` → keep the reset
  (defensive) but log a warning naming the unexpected type so a malformed
  `output:` block is not silently dropped.

Each fix gets a regression test asserting the message/behavior.

### 4. Keep-as-is + cleanup

`args_parser.py` and `optimization_runner.py` unchanged. Fix the stale
"eagerly imports JAX" comment in `main.py:26` to reflect the lazy `__getattr__`
reality.

## Data flow (unchanged externally)

```
xpcsjax --config x.yaml
  → main() → args_parser → commands.dispatch_command
  → config_handling.load_and_merge_config (hardened errors)
  → optimization_runner.run_nlsq (fit; imports JAX here — correct)
  → plot_dispatch.dispatch_plots → plot_backend + plot_families/*
```

## Error handling

The only behavioral change is in `config_handling`: errors that were cryptic or
silent become actionable (file context) or logged. No change to success paths,
exit codes, or the NLSQ result.

## Testing & parity

- Plotting and config generation are **not** part of the NLSQ parity contract
  (no `rtol=1e-10` risk). The homodyne characterization baselines are untouched.
- Existing tests stay green after import-path updates: `test_plot_dispatch_logging`,
  `test_output_resolution`, `test_simulated_data_grid`, `test_config_generator_yaml`,
  `test_build_parser_factories`, plus completion parity (`test_completion_parity`).
- New regression tests for the 3 config error paths.
- `make verify` + `mypy xpcsjax` (hard CI gate) green before merge.
- Every moved function: confirm no importer outside those listed above before
  moving (grep), to avoid a silent broken import.

## Acceptance criteria

1. `plot_dispatch.py` ≈ 100 LOC orchestrator; `plot_backend.py` +
   `plot_families/{experimental,simulated,postfit}.py` created; `dispatch_plots`
   signature unchanged.
2. JAX import for plotting is isolated to `plot_families/simulated.py`.
3. `config_template.py` holds the template logic; `config_generator.generate_config`
   / `.build_parser` / `.main` all still importable (facade); GUI + completion
   untouched and green.
4. The 3 `config_handling` error paths emit actionable/logged messages, each with
   a regression test.
5. `main.py` JAX-import comment corrected.
6. All pre-existing CLI tests pass after import-path updates; `make verify` +
   `mypy xpcsjax` green.

## Follow-ups

Sub-project 3 (GUI) can later migrate its `config_generator.generate_config`
import to `config_template` directly and drop the facade re-export if desired —
out of scope here.
