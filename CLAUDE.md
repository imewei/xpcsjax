# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project scope and what it is *not*

**xpcsjax is NLSQ-only by design.** v0.1 ports the homodyne + heterodyne XPCS NLSQ pipelines into one JAX-native package. Bayesian sampling — NumPyro, BlackJAX, ArviZ, CMC (Consensus Monte Carlo), NUTS, HMC, parallel tempering — is **permanently out of scope.** Users needing Bayesian XPCS analysis should use the upstream `homodyne` or `heterodyne` packages, not this one.

The architectural rule this implies:

- **Do not wire up `get_cmc_config()` or any MCMC pathway.** Stale references survive from the homodyne port in:
  `xpcsjax/config/manager.py` (`get_cmc_config`, `_get_default_cmc_config`, `"mcmc"` config block), `xpcsjax/core/`, `xpcsjax/data/`, `xpcsjax/utils/logging.py`, and ~20 other files. Treat these as **scheduled-for-removal** dead code; don't add new call sites and don't write tests that exercise them.
- New optimization code goes through `fit_nlsq` (the v0.1 single-entry wrapper) or `fit_nlsq_jax` / `fit_nlsq_multistart`. There is no second optimizer pathway to "fall back to."

### Intentional v0.1 cuts from the homodyne port

Two homodyne modules were deliberately not ported. Don't flag their absence as parity gaps or port them on speculation:

- **`homodyne/optimization/checkpoint_manager.py` is not ported.** Resumable long-running NLSQ jobs are out of scope for v0.1 — re-launch rather than resume. JAX's stateless / JIT-pure-function idiom makes mid-run checkpoint/restore awkward, and the workloads xpcsjax targets fit in a single CPU-bound run.
- **`homodyne/core/scaling_utils.py` is not ported.** Its quantile / contrast helpers are only used by homodyne's CMC / MCMC path, which is permanently out of scope for xpcsjax. The NLSQ path uses the mirrored `compute_quantile_per_angle_scaling()` in `xpcsjax/optimization/nlsq/parameter_utils.py:345`.

## Architecture you need to know before editing

### Public API is lazy-loaded via `__getattr__`

`xpcsjax/__init__.py` does **not** import its public symbols at module top-level. Six names are registered in `_LAZY_EXPORTS` and resolved on first attribute access via a module-level `__getattr__`:

```python
_LAZY_EXPORTS = {
    "load_xpcs_data":     "xpcsjax.data",
    "fit_nlsq":           "xpcsjax.optimization.nlsq",
    "ConfigManager":      "xpcsjax.config",
    "HomodyneModel":      "xpcsjax.core",
    "HeterodyneModel":    "xpcsjax.core",          # public lazy export (Phase 6)
    "OptimizationResult": "xpcsjax.optimization.nlsq.results",
}
```

Adding a new public symbol means: (a) add it to `_LAZY_EXPORTS`, (b) add to literal `__all__`, (c) ensure the target submodule actually exposes the symbol (the runtime `assert` will catch (a)/(b) drift but not (c)). Pyright's `reportUnsupportedDunderAll` requires `__all__` to be a literal list, so don't generate it from `_LAZY_EXPORTS`.

`xpcsjax.viz` is a separate lazy-loaded subpackage (not in the top-level `_LAZY_EXPORTS`). Import directly:
`from xpcsjax.viz import plot_nlsq_fit, plot_residual_map, plot_simulated_data, generate_nlsq_plots, compute_diagonal_overlay_stats, DiagonalOverlayResult`

`xpcsjax/config/parameter_registry.py` is the single source of truth for parameter names, bounds, and physical constraints across all modes. When adding a new physics parameter, register it there first — `ConfigManager` and the NLSQ bounds builder both read from the registry.

### `xpcsjax/__init__.py` sets JAX environment **before any JAX import**

The module top sets:
- `JAX_ENABLE_X64=1` (parameters span 6+ orders of magnitude — float32 is unsafe)
- `XLA_FLAGS` including `--xla_force_host_platform_device_count=4` (parallel paths) and `--xla_disable_hlo_passes=constant_folding` (avoids > 1 s slow-compile warnings on HYBRID_STREAMING with 23M+ points)
- `NLSQ_SKIP_GPU_CHECK=1` (v0.1 is CPU-only; GPU support is v0.2+)

If you need to add or amend these flags, do it **inside `xpcsjax/__init__.py` only** — adding env-mutation elsewhere will race the first JAX import.

### NLSQ engine: xpcsjax owns strategy, NLSQ owns the trust-region solve

The split with the upstream NLSQ library (`nlsq>=0.6.10`) is:

- **NLSQ owns:** `CurveFit` JIT cache, `curve_fit()`, the trust-region (Levenberg-Marquardt) solve. `WorkflowSelector` was removed in NLSQ v0.6.0 — do **not** call it.
- **xpcsjax owns:** memory-aware strategy routing (`select_nlsq_strategy`), the 5-layer anti-degeneracy controller (`anti_degeneracy_controller.py`), CMA-ES escape (auto-triggered above a threshold), LHS multistart, bounds + parameter transforms, angle-stratified chunking for large datasets, and shear-weighting.

When working inside `xpcsjax/optimization/nlsq/`, the convention is: call NLSQ's `CurveFit` directly, never NLSQ's higher-level `fit()` unified API or its `MemoryBudgetSelector`. xpcsjax routes memory itself.

The 5 anti-degeneracy layers, in order:

| Layer | Name | Module | Active modes |
|-------|------|--------|-------------|
| L1 | Fourier/Constant Reparameterization | `fourier_reparam.py` | all |
| L2 | Hierarchical Optimization | `hierarchical.py` | all |
| L3 | Adaptive CV-based Regularization | `adaptive_regularization.py` | all |
| L4 | Gradient Collapse Monitoring | `gradient_monitor.py` | all |
| L5 | Shear-Sensitivity Weighting | `shear_weighting.py` | `laminar_flow` only |

Layer gating is declared in `_LAYER_GATES` at the top of `anti_degeneracy_controller.py`. Layers absent from that dict are default-active for all modes; only L5 is gated. L5 up-weights data near the flow direction φ0 to exploit the shear-sensitivity peak, which exists only when the kernel has a shear rate — so L5 is active for `laminar_flow` **only**. The static modes (`static_anisotropic`, `static_isotropic`) have no flow direction and `two_component` (heterodyne) has no shear rate, so L5 short-circuits for all of them. Note `is_layer_active()` still returns `True` for every layer when `analysis_mode=None` (the homodyne characterization gate's path), so this gating does not affect the rtol=1e-10 parity baselines.

### Analysis modes and config templates

xpcsjax ships four mode-specific YAML templates under `xpcsjax/config/templates/`:

| Mode | Template file |
|------|--------------|
| `static_anisotropic` | `xpcsjax_static_anisotropic.yaml` |
| `static_isotropic` | `xpcsjax_static_isotropic.yaml` |
| `laminar_flow` | `xpcsjax_laminar_flow.yaml` |
| `two_component` | `xpcsjax_two_component.yaml` |

`ConfigManager` validates mode at construction; passing an unknown mode raises immediately.

**`data_type` valid values:** `"aps_old"` (legacy APS format) or `"aps_u"` (unified APS format). No other strings are accepted.

### Homodyne is both a port source and a parity oracle

`scripts/generate_homodyne_baselines.py` runs the upstream homodyne package and serializes results into `tests/characterization/fixtures/`. `tests/characterization/test_homodyne_equivalence.py` then asserts xpcsjax produces bit-comparable output. **When porting any new module from homodyne, generate a fresh baseline before changing behavior.** That's how regressions are caught.

### Heterodyne is a fully public model with per-angle-mode parity

HeterodyneModel is a public lazy export. Phase 6 brought it to full
per-angle-mode parity with homodyne — see
`docs/source/theory/heterodyne_anti_degeneracy.rst` for the 4-layer defense
system (L5 shear-weighting is `laminar_flow`-only by design — heterodyne has
its own, structurally different velocity/flow term, so laminar_flow's
shear-sensitivity weighting does not transfer to it).

### Heterodyne per-angle modes (parity with homodyne)

| Mode | Optimizer params | When to use |
|---|---|---|
| `constant` | `n_physics` | Pre-estimate scaling, freeze |
| `auto` (default) | depends on `n_phi` | Recommended; dispatches by thresholds |
| `fourier` | `n_physics + 2(2K+1)` | Many angles, smooth angular variation |
| `individual` | `n_physics + 2·n_phi` | Many angles, large physical contrast variation |

L5 (shear weighting) is `laminar_flow`-only. Heterodyne's velocity/flow term
(`v0`, `v_offset`, `phi0_het`) is structurally different from laminar_flow's
shear rate (`gamma_dot`), so laminar_flow's shear-sensitivity weighting does
not apply to it. See `docs/source/theory/heterodyne_anti_degeneracy.rst`.

## Commands

Use the project Makefile rather than reinventing pytest/ruff invocations — the targets are tuned for this layout (domain-sharded tests, not pyramid layers).

| Action | Command |
|---|---|
| Install dev deps | `make dev` (= `uv pip install -e ".[dev]"`) |
| Run all tests | `make test` |
| Run a single test file | `uv run pytest tests/optimization/test_nlsq_core.py -v` |
| Run a single test | `uv run pytest tests/optimization/test_nlsq_core.py::test_name -v` |
| Domain-scoped tests | `make test-core` · `make test-optimization` · `make test-heterodyne` · `make test-characterization` · `make test-property` |
| Fast smoke | `make test-smoke` |
| Pre-push gate | `make verify` (lint + advisory mypy + smoke under `-x -n auto`) |
| Lint | `make lint` (ruff, line-length 100, `E,F,W,I,B,UP,N`) |
| Type-check | `make type-check` (mypy non-strict, `ignore_missing_imports=true`) |
| Format | `make format` (ruff format + `ruff check --fix`) |
| Verify NLSQ integration end-to-end | `make verify-nlsq` |
| Generate homodyne baselines | `make run-example` (= `python scripts/generate_homodyne_baselines.py`) |

Notes:
- `pytest` auto-loads `JAX_ENABLE_X64=1` from `[tool.pytest.ini_options]` — no need to set it manually for tests.
- `make type-check` will surface many findings because `strict = false`; `make verify` runs mypy in **advisory** mode (`| tail -1 || true`) so type findings don't block push.
- Python 3.12+ required (per `pyproject.toml`).

## Workflow conventions

- **uv-first.** `uv.lock` is the source of truth; never run bare `pip install`. The Makefile auto-detects `uv` and uses `uv run` to route through `.venv`.
- **Float64 everywhere.** `JAX_ENABLE_X64=1` is mandatory — parameters span 6+ orders of magnitude.
- **No `from module import *`.** Enforced by user CLAUDE.md and by ruff (`F` rule).
- **JIT-safe interpolation only.** Use `interpax`, never `jax.numpy.interp` in JIT'd paths.
- **Characterization tests are the parity contract.** If `tests/characterization/test_homodyne_equivalence.py` starts failing after a port change, do **not** loosen tolerances — regenerate the baseline only if the homodyne package itself changed.
- **Characterization tests are corpus-loading, not corpus-generating.** `make test-characterization` reads serialized fixtures from `tests/characterization/fixtures/{baselines,configs}/` and does **not** need the upstream `homodyne` / `heterodyne` packages — `test_homodyne_equivalence.py` imports only stdlib. Only `make run-example` (the baseline regenerator at `scripts/generate_homodyne_baselines.py`) imports them, and only the maintainer runs that — typically when upstream ships a behavior fix that needs to propagate. Neither package is declared in `pyproject.toml` as a dependency or extra; this is deliberate — xpcsjax is a JAX-native rewrite and shouldn't pull its predecessors into normal installs. To regenerate baselines, manually install the upstream packages into xpcsjax's venv first: `uv pip install -e /path/to/homodyne` (and `/path/to/heterodyne` when its baseline generator lands).

## graphify

This project has a graphify knowledge graph at `graphify-out/`.

- Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` for god nodes and community structure.
- If `graphify-out/wiki/index.md` exists, navigate it instead of reading raw files.
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files.
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost).
