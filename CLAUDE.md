# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project scope and what it is *not*

**xpcsjax is NLSQ-only by design.** v0.1 ports the homodyne + heterodyne XPCS NLSQ pipelines into one JAX-native package. Bayesian sampling — NumPyro, BlackJAX, ArviZ, CMC (Consensus Monte Carlo), NUTS, HMC, parallel tempering — is **permanently out of scope.** Users needing Bayesian XPCS analysis should use the upstream `homodyne` or `heterodyne` packages, not this one.

The architectural rule this implies:

- **Do not wire up any Bayesian / MCMC / CMC pathway.** The homodyne port's CMC/MCMC machinery (`get_cmc_config`, `_get_default_cmc_config`, the `"mcmc"` config block) has been **removed** — those symbols no longer exist anywhere in the package. What remains are a handful of docstrings and **defensive guards** that *name* Bayesian sampling only to state it is out of scope (e.g. the `ValueError` in `xpcsjax/data/optimization.py` that rejects non-NLSQ methods). Keep those — they reject invalid input, they are not dead code. Don't add new Bayesian call sites and don't write tests that exercise one.
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

L4 is a **per-iteration gradient-collapse monitor** (`build_gradient_collapse_callback` feeding `GradientCollapseMonitor`), a **shared mechanism** with behavioral parity between `laminar_flow` and `two_component`. It is **strictly diagnostic** — monitor-on vs monitor-off is bit-identical (the homodyne rtol=1e-10 baselines included). When the solver callback never fires it falls back to a **post-solve covariance-condition** check; the `gradient_monitor` diagnostics block's `mechanism` field reports which path ran (`per_iteration_gradient_ratio` vs `post_solve_fallback`), and `gradient_consecutive_triggers` is now effective. Per-iteration is wired on the standard joint-fit path of both modes; the ≥1 M stratified tier is not yet wired (documented follow-up).

The anti-degeneracy *diagnostics contract* is now symmetric across modes: both `laminar_flow` and `two_component` emit the same top-level `nlsq_diagnostics` activation keys (`hierarchical_active`, `regularization_active`, `shear_weighting`, + `gradient_monitor` when L4 ran) via the shared `assemble_anti_degeneracy_diagnostics` (`xpcsjax/optimization/nlsq/anti_degeneracy_diagnostics.py`). The flat top-level keys are emitted at all dataset sizes — every laminar path (in-memory, HYBRID_STREAMING, stratified-LS ≥1 M, sequential, out-of-core) plus all heterodyne paths (in-memory, STREAMING) — with honest per-path values: both laminar and heterodyne HYBRID_STREAMING report the real active L2/L3 they ran; stratified-LS/sequential/out-of-core report inactive markers (`hierarchical_active=False`, `regularization_active=False`, `shear_weighting="laminar_flow_inactive"`) since those layers don't run there.

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

### Heterodyne joint global escapes (parity gap C closed)

The heterodyne joint CMA-ES (`_fit_joint_cmaes_multi_phi`) and joint multistart
(`_fit_joint_multistart`) escapes in `heterodyne_core.py` are **real global
escapes** (no longer the Phase-6 minimal stubs). Each runs a seed-pinned global
optimizer over the joint `[physics | scaling]` vector, **keeps-better** vs the
plain NLSQ joint fit, and **best-effort falls back** to the plain joint fit on
failure — reusing the shared `fit_with_cmaes` / `run_multistart_nlsq`. This is
the joint-fit global escape; the per-angle escapes were already real. This
closed parity gap **C** between `two_component` and `laminar_flow`. An escape
result is tagged `nlsq_diagnostics["global_escape"]` and, by construction,
carries NaN covariance / uncertainties and `n_iterations=0` (no covariance solve
on the kept vector) — read `global_escape` to detect an escape result.

**The escape honours the resolved per-angle scaling mode — it does NOT force
Fourier.** `fit_nlsq_multi_phi` resolves `effective_mode` (`_resolve_effective_mode`)
*before* the global-escape gate, so enabling CMA-ES / multistart never changes
which scaling layout is used (the consistency invariant: `auto → averaged` for
`n_phi >= constant_scaling_threshold` (3) else `individual`; `constant`/`fourier`
explicit-only — see the `per_angle_mode` templates). Routing by mode:

- **`fourier` / `individual`** escapes use the Fourier-reparam joint problem
  builder (`_fit_joint_cmaes_multi_phi` / `_fit_joint_multistart` →
  `_build_joint_problem` / `_build_joint_fourier`: `fourier` ↔ `independent`).
- **`averaged`** (the `auto` default at `n_phi >= 3`) and explicit **`constant`**
  escapes run the global search over their OWN `[physics | scaling]` data
  residual via the `global_escape_kind=` hook on `_fit_joint_averaged_multi_phi`
  / `_fit_joint_constant_multi_phi` (frozen scaling → physics-only search). The
  shared keep-better + escape-contract machinery lives in `_apply_global_escape`
  (with `_cmaes_joint_candidate` / `_multistart_joint_candidate` /
  `_solve_residual_nlsq`); when `global_escape_kind=None` those solvers are
  byte-identical to the plain path. This mirrors `laminar_flow`'s CMA-ES, which
  honours `use_averaged_scaling` (`core.py`'s `AntiDegeneracyController` path) —
  so a default `auto` fit no longer silently switches to a Fourier scaling tail
  just because a global escape was enabled.

### Heterodyne hybrid-streaming anti-degeneracy (parity gap D closed)

The heterodyne STREAMING path previously froze the quantile-estimated per-angle
scaling inside the JIT closure and ran no anti-degeneracy layers. Gap D is now
**closed**: `fit_with_stratified_hybrid_streaming_heterodyne` optimizes the
scaling tail and runs L1–L4, mirroring `laminar_flow` streaming:

- **`per_angle_mode` dispatch** (driven by `anti_degeneracy_config.per_angle_mode`):
  `auto` is **the default**, including when `anti_degeneracy_config` is absent/`None`
  (mirrors laminar `hybrid_streaming.py:462` — no "freeze when unconfigured" special
  case). `auto` → `auto_averaged` (2 averaged scaling params) when
  `n_phi ≥ constant_scaling_threshold` (default 3), else → `individual`
  (2·n_phi per-angle params, activates L2). `fixed_constant` (frozen scaling) is the
  **explicit opt-out** via `per_angle_mode="constant"`. `fourier` → 2·(2K+1) Fourier
  coeffs (silently falls back to `individual` when n_phi < 1+2K — surfaced via
  `meta["fourier_effective_mode"]`).
- **L1** active for all optimized modes; skipped for `fixed_constant`.
- **L2** (`HierarchicalOptimizer`) gated to `individual`/`fourier` exactly mirroring
  laminar's gate. `auto_averaged`/`fixed_constant` skip L2. On the L2 branch the
  `[physics | scaling]` vector is permuted to `[per_angle | physics]` layout,
  solved, and un-permuted; covariance is an identity placeholder on this branch
  (`info["covariance_is_placeholder"] = True`).
- **L3** adaptive CV regularization active when `regularization.enable=True` and
  there is a scaling tail; group indices are mode-aware.
- **L4** gradient-collapse monitor wired via `callback=` (plain branch) and
  `_hier_grad` (L2 branch); strictly observational.
- **L5** omitted (heterodyne has no shear term); diagnostics report
  `'laminar_flow_inactive'` sentinel.
- **Parity contract:** mechanism + objective (optimized SSR ≤ frozen baseline),
  NOT `rtol=1e-10` (that gate is homodyne-specific).

Diagnostics: streaming emits the symmetric `info["anti_degeneracy"]` block
(`hierarchical_active`, `regularization_active`, `shear_weighting`,
`gradient_monitor`, `per_angle_mode`) via `assemble_anti_degeneracy_diagnostics`.

Remaining optional/structural follow-ups: **A** (routing heterodyne standard path
through the shared `AntiDegeneracyController`) and aligning heterodyne's standard-path
L2 onto `HierarchicalOptimizer` (currently an inline two-stage implementation).
Neither is required for mechanism parity; both are architectural cleanups.

**Gap A is NOT a parity gap — do not "fix" it by routing heterodyne through the
controller.** The parity target is laminar's *standard* path, and that path is
itself inline: `wrapper.py` has **zero** `AntiDegeneracyController` uses. The
controller is instantiated ONLY in laminar's CMA-ES path (`core.py:1846`, inside
`fit_nlsq_cmaes`) and stratified-LS path (`stratified_ls.py:203`) — it is
path-specific, not a universal laminar orchestrator. The laminar standard path
also runs no L2 (`wrapper.py` hard-codes `hierarchical_active=False`; the
in-memory path runs no L2/L3). So heterodyne's inline standard path already
*matches* laminar's inline standard path; routing it through the controller (or
swapping its inline two-stage L2 onto `HierarchicalOptimizer`, which both modes
already share on the streaming paths) would **create** divergence and risk
numeric drift with no rtol gate to catch it. Likewise the `shear_weighting` L5
sentinel split (`not_applicable_heterodyne` on heterodyne public surfaces vs
`laminar_flow_inactive` on the internal streaming mirror-block, translated by
`heterodyne_result_builder.py`) is a deliberate two-value design pinned by ~12
tests, not a cosmetic asymmetry to unify. (Reviewed 2026-06-01.)

### Heterodyne memory strategy and angle stratification

Heterodyne mirrors homodyne's angle-stratification mechanism (mechanism parity, not numerical parity — heterodyne fits a different model). The dispatch inside `_fit_nlsq_heterodyne` is: cmaes → multi_start → hybrid_streaming → stratified-LS (≥ 1 M points) → in-memory joint fit.

Key facts:
- **Stratified-LS activates at ≥ 1 M points only.** Homodyne has an additional 100k–1 M shuffle-only regime; this does NOT transfer to heterodyne because heterodyne's sub-1 M solver is batched by angle (`(n_phi, N, N)`) with no flat point list to shuffle.
- **Default-on seed-42 pre-shuffle** (objective-invariant — reordering residuals does not change SSR).
- **Config:** `optimization.stratification.{enabled="auto", target_chunk_size=100000, max_imbalance_ratio=5.0, ...}` in the mode YAML. `enabled: false` reverts to in-memory joint fit at all sizes.
- New module: `xpcsjax/optimization/nlsq/heterodyne_stratified_ls.py` (`fit_heterodyne_stratified_least_squares`).
- See `docs/source/theory/heterodyne_memory_strategy.rst` for the full decision table and config reference.

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
- **The homodyne characterization gate is a maintainer-local LIVE oracle, env-gated OFF by default — never enable it in CI or expect it in a fresh clone.** `tests/characterization/test_homodyne_equivalence.py` and `tests/parity/test_l4_per_iteration_parity.py::test_homodyne_characterization_bit_identical_with_monitor` run **only** when `XPCSJAX_RUN_CHARACTERIZATION=1`; otherwise they self-skip. When enabled they run a **live fit against the upstream `homodyne` package** and read its configs/datasets at **hardcoded absolute paths outside the repo** (e.g. `/home/wei/Documents/Projects/data/C020/...`), so the gate passes only on a maintainer machine that has both. Forcing it on anywhere else fails loudly with "homodyne not importable" / "registered config path is dead" — this is exactly what broke CI when the flag was set there (see the CI-facts memory). The test module imports only stdlib + numpy at top level, but that does **not** mean the comparison runs without upstream: the gated test bodies pull in `homodyne` and the local datasets at runtime. `make run-example` (`scripts/generate_homodyne_baselines.py`) likewise needs the upstream packages — install with `uv pip install -e /path/to/homodyne` (and `/path/to/heterodyne` when its generator lands). Neither upstream package is declared in `pyproject.toml`; deliberate — xpcsjax is a JAX-native rewrite and shouldn't pull its predecessors into normal installs.

## graphify

This project has a graphify knowledge graph at `graphify-out/`.

- Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` for god nodes and community structure.
- If `graphify-out/wiki/index.md` exists, navigate it instead of reading raw files.
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files.
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost).
