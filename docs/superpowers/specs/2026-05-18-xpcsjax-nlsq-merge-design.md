---
title: "xpcsjax — Homodyne + Heterodyne NLSQ Merge Design"
status: draft
date: 2026-05-18
author: Wei Chen
scope: NLSQ-only foundation (v0.1)
---

# xpcsjax — Homodyne + Heterodyne NLSQ Merge Design

## 1. Summary

Consolidate the `homodyne` and `heterodyne` XPCS analysis packages into a unified `xpcsjax` package. v0.1 covers data loading, diagonal correction, NLSQ fitting, minimal config, and result schema. CMC (Bayesian), visualization, CLI, device-config subsystems are explicitly deferred to later phases.

xpcsjax adopts homodyne's NLSQ engine **verbatim** — same memory-aware strategy routing, same 5-layer anti-degeneracy controller, same CMA-ES escape, same per-angle chi-squared loss, same result schema. The two physics models (homodyne and heterodyne) live as **peer classes** sharing the engine; only the residual function differs. Dispatch is config-driven through a single `analysis_mode` enum and a single `fit_nlsq` entry point.

## 2. Goals

1. **Behavioral fidelity to homodyne** — every homodyne test config must produce bit-equivalent results in xpcsjax (`rtol=1e-10`).
2. **First-class heterodyne support** — `analysis_mode: two_component` produces fits within 1σ of heterodyne's golden-value baselines.
3. **One API for both models** — a user switches between physics models by editing config, not by importing different functions.
4. **No accumulated dead code in v0.1** — code paths for deferred subsystems (CMC, viz, CLI) are not carried into v0.1.

## 3. Non-goals

- Porting heterodyne's separately-developed anti-degeneracy controller. Homodyne's controller covers the relevant cases.
- Porting heterodyne's soft-L1 robust loss. Chi-squared is sufficient with the inherited anti-degeneracy machinery.
- Maintaining backwards-compatible imports from `homodyne.*` or `heterodyne.*`. xpcsjax replaces both.
- CMC, visualization, CLI, datashader, NetworkDynamics — all v0.2+ scope.

## 4. Architectural decisions

| # | Decision |
|---|---|
| D1 | Peer models, shared infrastructure |
| D2 | NLSQ-only v0.1 foundation |
| D3 | NLSQ engine verbatim from homodyne; physics models swap at the residual boundary |
| D4 | Config-driven single-entry `fit_nlsq(data, config)` |
| D5 | Strict mirror layout + v0.1 dead-code trim |
| D6 | Heterodyne parameter bounds verbatim from heterodyne docs (symmetric ref/sample bounds) |
| D7 | Anti-degeneracy controller gated by model: Homodyne→5 layers, Heterodyne→4 (no ShearSensitivityWeighting) |
| D8 | Trust-region LM is **JAX-native** via `nlsq.CurveFit` (`nlsq/core/trf.py`, `trf_jit.py`); end-to-end on-device, GPU/TPU-capable; **never use `scipy.optimize.least_squares`** |

## 5. Package layout

```
xpcsjax/
├── pyproject.toml
├── xpcsjax/
│   ├── __init__.py                       # lazy-import top-level API
│   ├── config/
│   │   ├── manager.py                    # ConfigManager (verbatim from homodyne)
│   │   ├── parameter_registry.py         # 3+7 homodyne entries + 14 heterodyne entries
│   │   └── parameter_manager.py          # reads from registry + applies user overrides
│   ├── data/                             # verbatim from homodyne/data/
│   │   ├── xpcs_loader.py
│   │   ├── filtering_utils.py
│   │   ├── preprocessing.py
│   │   ├── quality_controller.py
│   │   └── performance_engine.py         # MultiLevelCache
│   ├── core/
│   │   ├── models.py                     # PhysicsModelBase + HomodyneModel variants
│   │   ├── heterodyne_model.py           # HeterodyneModel (new file)
│   │   ├── diagonal_correction.py        # verbatim from homodyne
│   │   └── kernels.py                    # JAX g₁/g₂ kernels; NLSQ paths only
│   ├── optimization/
│   │   └── nlsq/                         # verbatim from homodyne (full controller)
│   │       ├── core.py                   # fit_nlsq entry
│   │       ├── adapter.py
│   │       ├── memory.py                 # select_nlsq_strategy + adaptive thresholds
│   │       ├── anti_degeneracy_controller.py
│   │       ├── gradient_monitor.py
│   │       ├── cmaes_wrapper.py
│   │       └── strategies/
│   │           ├── chunking.py
│   │           ├── residual.py
│   │           └── executors.py
│   ├── io/
│   │   └── results_nlsq.py               # JSON + NPZ serialization
│   ├── utils/                            # verbatim from homodyne
│   └── device/                           # verbatim from homodyne (JAX/XLA config)
├── tests/
│   ├── characterization/                 # rtol=1e-10 vs homodyne baselines
│   ├── heterodyne/                       # within-1σ vs heterodyne baselines
│   └── property/                         # Hypothesis invariants
└── docs/
    ├── architecture/                     # ported from homodyne + heterodyne
    └── superpowers/specs/                # this doc lives here
```

## 6. Dependencies

| Package | Pin | Purpose |
|---|---|---|
| Python | ≥ 3.12 | per CLAUDE.md |
| `jax`, `jaxlib` | latest | float64 mandatory at import time; backs nlsq's on-device LM step |
| `nlsq` | ≥ 0.6.4 | JAX-native trust-region reflective solver; CurveFit entry point |
| `optimistix`, `optax` | latest | compatibility with homodyne shims |
| `evosax` | latest | CMA-ES backend (transitive via `nlsq.CMAESOptimizer`) |
| `h5py` | latest | XPCS HDF5 I/O |
| `numpy` | latest | host-side array utilities (data loader, result serialization) |
| `scipy` | latest | **not** used for the LM step; available only for non-hot-path utilities (covariance post-processing, statistical tests in result diagnostics) |
| `interpax` | latest | JIT-safe interpolation (per CLAUDE.md prohibition) |
| `pyyaml` | latest | config files |
| `pytest`, `hypothesis` | dev | testing |

`[tool.pytest.ini_options] env = ["JAX_ENABLE_X64=1"]` ensures tests run float64 from import time, matching homodyne's `device/` side-effect behavior.

## 7. Data layer

Verbatim port of `homodyne/data/`. Pipeline:

```
load_xpcs_data(config)
   → _detect_format (old APS vs APS-U HDF5)
   → load + half-triangle reconstruction → full symmetric C₂
   → validate (shape, float64, NaN%, monotonic time arrays)
   → filter (Q-range, phi-range wrapped OR, frame-range)
   → apply_diagonal_correction (mandatory; methods: basic/statistical/interpolation)
   → DataQualityController
   → MultiLevelCache (LRU + disk NPZ in XDG_CACHE_HOME)
   → {wavevector_q_list, phi_angles_list, t1, t2, c2_exp, dt, metadata}
```

Diagonal correction is **mandatory for both physics models** (matching homodyne v2.14.2 enforcement). The autocorrelation peak on C₂(t,t) is a physics artifact in both contexts.

## 8. Physics models

Peer classes living under `core/`:

### 8.1 `PhysicsModelBase` interface (ABC)

| Member | Type | Purpose |
|---|---|---|
| `param_names` | `tuple[str, ...]` | Ordered parameter labels |
| `param_bounds` | `tuple[tuple[float, float], ...]` | `(lo, hi)` per parameter |
| `param_transforms` | `tuple[Literal["linear","log"], ...]` | Per-parameter transform |
| `analysis_mode` | `AnalysisMode` | Enum used by anti-degeneracy gating |
| `initial_guess(data)` | `→ jnp.ndarray` | Model-specific warm-start |
| `compute_residual(params, data, ctx)` | `→ jnp.ndarray` | Flattened residual vector |
| `compute_jacobian(...)` *(opt.)* | `→ jnp.ndarray` | Otherwise `jax.jacfwd` |

### 8.2 HomodyneModel

- **`DiffusionModel`** — `static_diffusion` mode, 3 params: `D₀, α, D_offset`. `c₂ = offset + contrast·|g₁_diff|²`.
- **`CombinedModel`** — `laminar_flow` mode, 7 params: above + `γ̇₀, β, γ̇_offset, φ₀`. Includes **sinc(γ̇·q·L·t)** shear modulation in `g₁_shear`.
- Per-angle scaling `(contrast, offset)` solved **post-fit**, not part of the parameter vector.
- All kernels JAX-jit; NumPy fallback for finite-diff Jacobian.

### 8.3 HeterodyneModel

- **`two_component` mode**, 14 physics params + 2 per-angle scaling.
- `c₂ = c₁_ref(τ) + c₂_sample(φ, τ)` (PNAS Eq. S-95 form).
- **No sinc-shear term** — velocity enters as phase contribution to the cross/sample dynamics.
- Time-varying sample fraction `f_s(t) = clip(f0·exp(f1·(t−f2)) + f3, 0, 1)`.
- Per-angle scaling follows homodyne convention exactly.

## 9. Configuration system

### 9.1 `analysis_mode` enum

```
static_diffusion  → DiffusionModel       (3 params)
laminar_flow      → CombinedModel        (7 params, sinc-shear)
two_component     → HeterodyneModel      (14 physics + 2 scaling)
```

`ConfigManager.get_model()` reads the enum and instantiates the right class.

### 9.2 Parameter registry — heterodyne entries

Verbatim from [heterodyne.readthedocs.io configuration/options](https://heterodyne.readthedocs.io/en/latest/configuration/options.html):

| Parameter | Default | Bounds | Transform |
|---|---|---|---|
| `D0_ref` | 1e4 | [0, 1e6] | log |
| `alpha_ref` | 0.0 | [-2.0, 2.0] | linear |
| `D_offset_ref` | 0.0 | [-1e4, 1e4] | linear |
| `D0_sample` | 1e4 | [0, 1e6] | log |
| `alpha_sample` | 0.0 | [-2.0, 2.0] | linear |
| `D_offset_sample` | 0.0 | [-1e4, 1e4] | linear |
| `v0` | 1e3 | [0, 1e6] | log |
| `beta` | 1.0 | [0, 2.0] | linear |
| `v_offset` | 0.0 | [-100, 100] | linear |
| `f0` | 0.5 | [0, 1.0] | linear |
| `f1` | 0.0 | [-1.0, 1.0] | linear |
| `f2` | 0.0 | [-1.0, 1.0] | linear |
| `f3` | 0.0 | [-1.0, 1.0] | linear |
| `phi0` | 0.0 | [-π, π] rad | linear |
| `contrast` (scaling) | 1.0 | [0, 10.0] | linear |
| `offset` (scaling) | 0.0 | [-1.0, 1.0] | linear |

Homodyne entries lifted verbatim from `homodyne/config/parameter_registry.py` during the port (no hand-typing).

### 9.3 Identifiability for heterodyne (bounds are symmetric)

`D0_ref` and `D0_sample` share bounds `[0, 1e6]` and default `1e4`. Ref↔sample label-swap is broken by:

1. **The physics formula** — `c₁_ref` and `c₂_sample` contribute differently at different lag times under the time-varying `f_s(t)`; they aren't interchangeable in the loss landscape away from the trivial swap.
2. **User-supplied initial guesses** in YAML — the documented workflow for heterodyne fits.
3. **Narrow LHS multistart window** `[0.3·init, 3·init]` (homodyne default) keeps multistart inside the user-chosen basin.
4. **Inherited anti-degeneracy controller** — 4 of 5 layers apply (see §10).

### 9.4 Consistency fix during port

The flagged `parameter_registry.py` / `parameter_manager.py` inconsistency on contrast bounds resolves with the registry as the single source of truth. `parameter_manager.py` becomes a thin layer that reads from the registry and overlays user overrides.

## 10. NLSQ engine

**Verbatim port from `homodyne/optimization/nlsq/`.** The engine doesn't depend on physics; it consumes `PhysicsModelBase.compute_residual` and parameter bounds.

### 10.1 Solver substrate — JAX-native end-to-end

- `nlsq.CurveFit` uses nlsq's **JAX-native** trust-region reflective LM solver, implemented in `nlsq/core/trf.py` and JIT-compiled in `nlsq/core/trf_jit.py`. nlsq's own description: "JAX-accelerated nonlinear least squares curve fitting … GPU/TPU acceleration via JAX … Drop-in replacement for `scipy.optimize.curve_fit`."
- **The entire LM loop — residual, Jacobian, trust-region step, parameter update — stays on device.** No per-iteration Host↔Device transfers. GPU/TPU-capable end-to-end (CPU fallback automatic on non-Linux platforms via `NLSQ_FORCE_CPU`).
- **Never use `scipy.optimize.least_squares`.** xpcsjax does not import scipy's trust-region solver. The CLAUDE.md "JAX-first, minimize Host↔Device transfers" principle is enforced at the engine layer.
- Jacobian: `jax.jacfwd(model.compute_residual)`, JIT-compiled, vmap-vectorized. nlsq computes it internally when the residual function is JAX-compatible.
- **Finite-difference Jacobian fallback** (via nlsq's own `common_scipy.py` numerical utilities — these are JAX-ported scipy reference routines, not calls to scipy at runtime) is available for non-JAX residuals but is not used by xpcsjax models.

### 10.2 Memory-aware strategy routing

`select_nlsq_strategy(n_points, n_params, memory_fraction=0.75)` chooses per data size with **adaptive thresholds** based on system RAM (via `psutil`); overridable via `NLSQ_MEMORY_FRACTION` env var, clamped to `[0.1, 0.9]`.

| Strategy | Trigger | Behavior |
|---|---|---|
| `STANDARD` | Index + peak Jacobian memory < threshold | In-memory least squares |
| `OUT_OF_CORE` | Peak Jacobian exceeds threshold | Chunked J^T J accumulation, shared arrays |
| `HYBRID_STREAMING` | Index array exceeds threshold | L-BFGS warmup + streaming Gauss-Newton |

Default `THRESHOLD_GB = 16.0` if RAM detection fails.

### 10.3 Anti-degeneracy controller — model-gated

5-layer system from `homodyne/optimization/nlsq/anti_degeneracy_controller.py`:

| # | Layer | Homodyne | Heterodyne |
|---|---|---|---|
| 1 | `FourierReparameterizer` | active | active |
| 2 | `HierarchicalOptimizer` | active | active |
| 3 | `AdaptiveRegularizer` | active | active |
| 4 | `GradientCollapseMonitor` (ratio_threshold=0.01, consecutive=5, λ×10) | active | active |
| 5 | `ShearSensitivityWeighting` | active | **disabled (homodyne-specific)** |

Gating is by `model.analysis_mode`:
- `static_diffusion`, `laminar_flow` → 5 layers
- `two_component` → 4 layers (Layer 5 short-circuits to identity)

### 10.4 LHS multistart + CMA-ES escape

- **LHS multistart** auto-triggers when `scale_ratio = max(bounds_hi/bounds_lo) > 1000`.
- LHS window is `[0.3·init, 3·init]` around the user-supplied initial guess (homodyne default), **not** the full bounds. This is load-bearing for heterodyne identifiability.
- **CMA-ES** auto-enables when LHS trigger fires. Implementation: `nlsq.CMAESOptimizer` with `evosax` JAX backend, BIPOP restart strategy (alternating large/small populations), auto-configured `population_batch_size` and `data_chunk_size` for large datasets.

### 10.5 Bounds + parameter transforms

- Bounds enforced natively by nlsq's JAX trf solver (`bounds=(lo, hi)`).
- Log transforms applied at the **model boundary**: engine sees flat transformed parameter vector; `model.compute_residual` un-transforms before computing physics.
- `ParameterManager` reads transform table from registry.

## 11. Public API

```python
# typical workflow — identical regardless of model
from xpcsjax import load_xpcs_data, fit_nlsq

data   = load_xpcs_data("config.yaml")
result = fit_nlsq(data, "config.yaml")
print(result.parameters)
print(result.r_squared)
result.save("output_dir/")
```

```python
# advanced — direct ConfigManager access
from xpcsjax import ConfigManager, fit_nlsq, load_xpcs_data

cfg    = ConfigManager("config.yaml")
data   = load_xpcs_data(cfg)
result = fit_nlsq(data, cfg)         # accepts ConfigManager or path
```

`fit_nlsq` signature: `fit_nlsq(data: XPCSData, config: str | Path | ConfigManager) → OptimizationResult`.

### 11.1 `OptimizationResult` schema (verbatim from homodyne)

| Field | Type | Notes |
|---|---|---|
| `parameters` | `dict[str, float]` | Keyed by param_name |
| `parameter_errors` | `dict[str, float]` | 1σ from covariance diagonal |
| `covariance` | `np.ndarray (n, n)` | Full param covariance |
| `chi_squared` | `float` | Final loss |
| `dof` | `int` | n_residuals − n_params |
| `r_squared` | `float` | Per-angle + global |
| `residuals` | `np.ndarray` | Flattened |
| `convergence_status` | `Literal["converged","max_iter","trust_region_collapse"]` | |
| `n_iterations` | `int` | |
| `model_metadata` | `dict` | analysis_mode, q/φ ranges, multistart info, strategy used, initials |

Serialization: JSON for dict/scalar fields, NPZ for arrays.

### 11.2 Top-level lazy exports

`load_xpcs_data`, `fit_nlsq`, `ConfigManager`, `HomodyneModel`, `HeterodyneModel`, `OptimizationResult`.

## 12. Testing strategy

### 12.1 Layer 1 — Homodyne characterization (port-correctness gate)

`tests/characterization/`: every homodyne test config produces bit-equivalent xpcsjax results.

| Field | Tolerance |
|---|---|
| Fitted parameters | `rtol=1e-10` |
| chi_squared, r_squared | `rtol=1e-10` |
| convergence_status | exact match |
| Strategy used (STANDARD/OUT_OF_CORE/HYBRID_STREAMING) | exact match |
| Iteration count | ±1 |

A failure here is a port regression. **Phase 6 cannot start until this passes.**

### 12.2 Layer 2 — Heterodyne golden-value tests

`tests/heterodyne/`: canonical fits within heterodyne-baseline 1σ `parameter_errors`. Baselines generated by running source heterodyne package on the test configs.

If parameters fall outside 1σ on >5% of golden-value configs, the v0.2 backlog adds heterodyne's separate anti-degeneracy controller as an opt-in feature.

### 12.3 Layer 3 — Property tests (Hypothesis)

Invariants:
- Diagonal correction reduces matrix diagonal to interpolated neighbors (any method, any width).
- `inverse(transform(x)) == x` to float64 precision.
- Returned parameters always satisfy bounds.
- Result serialization round-trip preserves equality.

## 13. Phased migration plan

| Phase | Scope | Effort |
|---|---|---|
| 1 | Skeleton + deps + lazy `__init__.py` + CI scaffold | ~1 day |
| 2 | Data layer (verbatim from `homodyne/data/`, `utils/`, `device/`) | ~2 days |
| 3 | Config + HomodyneModel + parameter registry (with consistency fix) | ~2 days |
| 4 | NLSQ engine verbatim (incl. anti_degeneracy_controller, cmaes_wrapper, memory) | ~2 days |
| 5 | **GATE: homodyne characterization passes at rtol=1e-10** | ~1 day |
| 6 | HeterodyneModel + heterodyne registry entries + `two_component` enum | ~3 days |
| 7 | Heterodyne golden-value validation; gate Layer-5 disable for two_component | ~2 days |
| 8 | Architecture docs + migration guide + v0.1.0 tag | ~1 day |

**Total: ~14 days serial.** Phases 2–4 partially parallelize with multiple developers; the Phase 5 gate is strictly sequential.

## 14. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Homodyne characterization fails at `rtol=1e-10` | Med | High (Phase 5 gate) | Allow `rtol=1e-8` for trust-region late-iteration drift; any failure at `rtol=1e-6` is a real port bug |
| R2 | Heterodyne fits diverge under homodyne's 4-layer anti-degeneracy | **Low** (reduced from Med) | Med | Layer 2 tests catch it; v0.2 escape hatch defined |
| R3 | Memory-strategy thresholds miscalibrated for heterodyne data | Low | Low | Same c₂ data sizes; only parameter vector grows |
| R4 | Parameter naming collision | Low | Low | Registry keyed by `(analysis_mode, name)` |
| R5 | `nlsq` library version drift between source packages | Med | Low | Pin to homodyne's exact `nlsq` version in Phase 1 |
| R6 | Layer 5 (`ShearSensitivityWeighting`) disable for `two_component` regresses homodyne `laminar_flow` | Low | High | Layer 1 characterization tests include laminar_flow configs explicitly |

## 15. Out of scope for v0.1 (v0.2+ backlog)

- CMC (NumPyro Bayesian sampler, ArviZ diagnostics, parallel chain MCMC)
- Visualization (matplotlib + pyqtgraph, datashader, MCMC diagnostic plots)
- CLI (single binary with subcommands `fit`, `plot`, `convert`)
- PyQt6 desktop UI
- NetworkDynamics, advanced HPC scheduling
- Heterodyne's separately-developed anti-degeneracy controller (added if heterodyne golden-value tests fail >5%)
- soft-L1 robust loss as an opt-in alternative to chi-squared

---

## Appendix A — Cross-reference to source packages

| xpcsjax module | Source | Treatment |
|---|---|---|
| `xpcsjax/data/` | `homodyne/data/` | Verbatim |
| `xpcsjax/core/diagonal_correction.py` | `homodyne/core/diagonal_correction.py` | Verbatim |
| `xpcsjax/core/models.py` | `homodyne/core/models.py` | Verbatim minus CMC-only g₂ paths |
| `xpcsjax/core/heterodyne_model.py` | `heterodyne/core/heterodyne_model.py` + `heterodyne/core/physics.py` | Adapted to `PhysicsModelBase` interface |
| `xpcsjax/core/kernels.py` | `homodyne/core/kernels.py` | NLSQ-path entries only |
| `xpcsjax/optimization/nlsq/` | `homodyne/optimization/nlsq/` | Verbatim |
| `xpcsjax/config/` | `homodyne/config/` | Verbatim + consistency fix + heterodyne registry entries |
| `xpcsjax/io/results_nlsq.py` | `homodyne/io/results_nlsq.py` | Verbatim |
| `xpcsjax/utils/` | `homodyne/utils/` | Verbatim |
| `xpcsjax/device/` | `homodyne/device/` | Verbatim |
