# Codex review — shard core
## Files read
xpcsjax/core/__init__.py — 21 — full
xpcsjax/core/diagonal_correction.py — 521 — full
xpcsjax/core/fitting.py — 62 — full
xpcsjax/core/heterodyne_jax_backend.py — 476 — full
xpcsjax/core/heterodyne_model.py — 334 — full
xpcsjax/core/heterodyne_model_stateful.py — 419 — full
xpcsjax/core/heterodyne_models.py — 300 — full
xpcsjax/core/heterodyne_physics_factors.py — 99 — full
xpcsjax/core/heterodyne_physics_kernel.py — 442 — full
xpcsjax/core/heterodyne_physics_utils.py — 283 — full
xpcsjax/core/heterodyne_scaling_utils.py — 767 — full
xpcsjax/core/homodyne_model.py — 411 — full
xpcsjax/core/jax_backend.py — 1588 — full
xpcsjax/core/math_primitives.py — 69 — full
xpcsjax/core/models.py — 748 — full
xpcsjax/core/physics.py — 271 — full
xpcsjax/core/physics_factors.py — 365 — full
xpcsjax/core/physics_nlsq.py — 547 — full
xpcsjax/core/physics_utils.py — 312 — full

## Bugs

### core-B1: Heterodyne fits shift the entire time axis when the loader clamps the end frame
- Location: xpcsjax/core/heterodyne_model_stateful.py:234
- Severity: Required
- Failure scenario: A 100-frame request against an 80-frame acquisition at `dt=0.1` loads frames 1–80. The fit assigns them times 2.1–10.0 s instead of 0.1–8.0 s, changing the evaluation of time-dependent transport, velocity, and fraction.
- Root cause: The loader clamps the end of the window locally and leaves the configured end frame intact. `sync_time_axis` interprets every decrease in length as a leading-frame exclusion and retains the tail of the configured axis.
- Checked: Both HDF5 readers use the same clamping helper. The heterodyne fit dispatch passes only the observed length to the sync method. The short-grid evaluator regression compares with a model using the same sync method, so it cannot detect a shared time-origin error.
- Red test:
  ```python
  import numpy as np
  from xpcsjax import HeterodyneModel
  from xpcsjax.data.xpcs_loader import XPCSDataLoader

  ap = {"dt": 0.1, "start_frame": 1, "end_frame": 100,
        "scattering": {"wavevector_q": 0.01}}
  loader = XPCSDataLoader.__new__(XPCSDataLoader)
  loader.analyzer_config = ap
  c2 = loader._apply_frame_slicing_to_selected_q(np.ones((1, 80, 80)))
  model = HeterodyneModel.from_config({"analyzer_parameters": ap})
  model.sync_time_axis(loader._calculate_time_arrays(c2.shape[-1]))
  np.testing.assert_allclose(model.t, np.arange(1, 81) * 0.1)
  ```
- Confidence: high

### core-B2: The scattered homodyne kernel changes the modeled correlation on the same frame grid
- Location: xpcsjax/core/physics_nlsq.py:140
- Severity: Required
- Failure scenario: At `dt=0.1`, `q=0.01`, and diffusion parameters `[100, -2, 0]`, the pair `(0.1, 0.2)` produces scaled `g2 ≈ 0.951226` through the scattered branch and `g2 ≈ 0.939413` on the original meshgrid. Enabling stratification can therefore change the fitted objective for the same observations and parameters.
- Root cause: The singleton branch replaces the experiment’s trapezoidal intervals with 64 subdivisions per pair. The fitted meshgrid branch uses the original frame intervals. The shear branch makes the same substitution at line 270.
- Checked: `NLSQWrapper._apply_stratification_if_needed` accepts `enabled=true` below one million points, where `_create_residual_function` calls the singleton branch for each pair. Its diagonal mask leaves this off-diagonal pair active. `tests/core/test_physics_nlsq_degenerate_grid.py` checks milder exponents and lags with a loose tolerance.
- Red test:
  ```python
  import numpy as np
  import jax.numpy as jnp
  from xpcsjax.core.physics_nlsq import compute_g2_scaled

  p, t = jnp.array([100., -2., 0.]), jnp.array([0.1, 0.2])

  def g(a, b):
      return compute_g2_scaled(p, a, b, 0., 0.01, 1e6, 1., 0., 0.1)

  np.testing.assert_allclose(g(t[:1], t[1:])[0, 0, 0], g(t, t)[0, 0, 1],
                             rtol=1e-3)
  ```
- Confidence: high

### core-B3: The opt-in homodyne adapter flattens correlations beyond 10,000 frame intervals
- Location: xpcsjax/core/jax_backend.py:377
- Severity: Required
- Failure scenario: With `dt=0.01`, a constant diffusion rate of 100, and `q=0.01`, the adapter predicts the same `g2` of 1.5 for `(100 s, 100 s)` and `(100 s, 101 s)`. The latter should be about 1.495025.
- Root cause: The cached and uncached adapter closures call `CombinedModel.compute_g1_batch` without `time_grid`. The element-wise kernel builds exactly 10,001 points and clips both requested indices to the last index. The shear branch uses the same limit.
- Checked: `fit_nlsq_jax(use_adapter=True)` selects these closures without a 10,000-interval guard. The separate CMA-ES closure supplies the full grid. `tests/core/test_debug_audit_2026_07_22.py` explicitly demonstrates the plateau without a grid, while `tests/optimization/test_adapter_xdata_cache.py` stubs the physics kernel.
- Red test:
  ```python
  import numpy as np
  from xpcsjax.config.parameter_registry import AnalysisMode
  from xpcsjax.optimization.nlsq.adapter import get_or_create_model

  _, predict, _ = get_or_create_model(
      AnalysisMode.STATIC_ISOTROPIC, np.array([0.]), q=0.01,
      per_angle_scaling=True, dt=0.01,
  )
  x = np.array([[100., 100., 0.], [100., 101., 0.]])
  np.testing.assert_allclose(predict(x, 0.5, 1., 100., 0., 0.),
                             [1.5, 1. + 0.5 * np.exp(-0.01)])
  ```
- Confidence: high

### core-B4: Distinct time axes can share a cache key and return stale meshgrids
- Location: xpcsjax/core/jax_backend.py:132
- Severity: Required
- Failure scenario: Cache `[0, 1, 2, 3, 4, 5, 6, 7]`, then request `[0, 0.5, 2, 3, 4, 5, 6, 7]`. The second meshgrid still contains `1.0` in row 1. The homodyne callers then evaluate the rate at the old time.
- Root cause: The key records endpoints and three interior samples. A change at an unsampled index leaves the key identical, and a hit returns the stored grids without checking the contents.
- Checked: The cache accepts these one-dimensional arrays. The `compute_g1_*` and `compute_g2_scaled` wrappers reuse its output. `TestMeshgridCacheCollision` in `tests/core/test_jax_backend.py` changes at least one sampled index in each collision case.
- Red test:
  ```python
  import jax.numpy as jnp
  from xpcsjax.core.jax_backend import clear_meshgrid_cache, get_cached_meshgrid

  clear_meshgrid_cache()
  a = jnp.arange(8.)
  b = a.at[1].set(0.5)
  get_cached_meshgrid(a, a)
  assert float(get_cached_meshgrid(b, b)[0][1, 0]) == 0.5
  ```
- Confidence: high

### core-B5: A masked homodyne observation can make chi-squared NaN
- Location: xpcsjax/core/jax_backend.py:1269
- Severity: Required
- Failure scenario: A data pixel has `NaN` and `sigma=0` to mark it as excluded. `compute_chi_squared` returns `NaN` even when every included observation and prediction is finite.
- Root cause: The kernel substitutes infinity for the masked denominator and still subtracts the masked observation. Dividing a `NaN` numerator by infinity leaves `NaN` in the reduction.
- Checked: `CombinedModel.compute_chi_squared` forwards both arrays with only a `dt` guard. `TestChiSquaredZeroSigma` in `tests/core/test_jax_backend.py` uses finite data at every masked pixel. The public chi-squared function documents zero contribution from nonpositive sigma.
- Red test:
  ```python
  import jax.numpy as jnp
  from xpcsjax.core.jax_backend import compute_chi_squared

  t = jnp.arange(3.)
  data = jnp.ones((3, 3)).at[0, 0].set(jnp.nan)
  sigma = jnp.ones_like(data).at[0, 0].set(0.)

  def chi2(obs):
      return compute_chi_squared(jnp.array([100., 0., 0.]), obs, sigma,
                                 t, t, jnp.array([0.]), 0.01, 1e6, 0.5, 1., 1.)

  assert jnp.isclose(chi2(data), chi2(data.at[0, 0].set(1.)))
  ```
- Confidence: high

### core-B6: The public homodyne wrapper builds a laminar model from the shipped static templates
- Location: xpcsjax/core/homodyne_model.py:352
- Severity: Required
- Failure scenario: Constructing `HomodyneModel` from the shipped static-isotropic template yields `analysis_mode == "laminar_flow"` and a seven-parameter underlying model. Using that model’s defaults introduces a shear term into a static simulation.
- Root cause: Any nonempty `analysis_settings` activates legacy flag resolution. Both static templates populate `analysis_settings.model_description` and omit the legacy flags, so the missing flags default to false and bypass the explicit top-level mode.
- Checked: The documented public constructor accepts the YAML mapping directly. `ConfigManager.get_model()` uses the separate `make_model` factory. `tests/core/test_core_gaps.py` omits `analysis_settings` in its public-constructor fixture, and `tests/core/test_homodyne_models.py` constructs the static `CombinedModel` directly.
- Red test:
  ```python
  from pathlib import Path
  import yaml
  from xpcsjax import HomodyneModel

  path = Path("xpcsjax/config/templates/xpcsjax_static_isotropic.yaml")
  model = HomodyneModel(yaml.safe_load(path.read_text()))
  assert (model.analysis_mode, model.model.n_params) == ("static_isotropic", 3)
  ```
- Confidence: high

### core-B7: Stateful heterodyne residuals silently broadcast incomplete correlation matrices
- Location: xpcsjax/core/heterodyne_jax_backend.py:361
- Severity: Required
- Failure scenario: A four-frame public model given an experimental array shaped `(4, 1)` returns six ordinary off-diagonal residuals by reusing the single observed column. The residual vector has the expected length for a valid four-frame square matrix.
- Root cause: Subtraction broadcasts the input against the modeled `(4, 4)` surface before the off-diagonal gather. `HeterodyneModel.compute_residuals` converts the input to JAX and delegates without checking its shape.
- Checked: The stratified fit builder enforces square inputs on its own route. The separate `HeterodynePhysicsAdapter.compute_residual` has an explicit shape guard. The backend tests in `tests/core/test_debug_audit_2026_07_22.py` use well-shaped matrices and exercise masking.
- Red test:
  ```python
  import numpy as np
  import pytest
  from xpcsjax import HeterodyneModel

  model = HeterodyneModel.from_config({
      "temporal": {"dt": 0.1, "time_length": 4},
      "scattering": {"wavevector_q": 0.01},
  })
  with pytest.raises(ValueError):
      model.compute_residuals(np.ones((4, 1)))
  ```
- Confidence: high

### core-B8: JAX diagonal correction truncates fractional interpolants for integer arrays
- Location: xpcsjax/core/diagonal_correction.py:269
- Severity: Minor
- Failure scenario: For the integer JAX matrix `[[0, 1], [2, 0]]`, the correction computes 1.5 and writes 1 to each diagonal. Batch correction inherits the same behavior.
- Root cause: The JAX wrappers preserve the input dtype and scatter the floating-point diagonal back into the integer array. The NumPy wrappers explicitly promote integer inputs before writing.
- Checked: The public API auto-selects JAX for a JAX array and imposes no floating-dtype guard. The property tests and the preprocessing integer regression exercise NumPy. The optimizer normally supplies floating-point theory arrays.
- Red test:
  ```python
  import jax.numpy as jnp
  from xpcsjax.core.diagonal_correction import apply_diagonal_correction

  c2 = jnp.array([[0, 1], [2, 0]])
  assert jnp.all(jnp.diag(apply_diagonal_correction(c2)) == 1.5)
  ```
- Confidence: high

### core-B9: The reported heterodyne sample fraction differs from the fraction used in correlation
- Location: xpcsjax/core/heterodyne_models.py:296
- Severity: Minor
- Failure scenario: With valid coefficients `f0=f1=f2=f3=0`, the public `compute_fraction()` reports zero sample fraction at every frame. The correlation kernel for those same parameters uses approximately `0.01386294`.
- Root cause: The public helper implements a separate hard-clipped exponential. The documented correlation model uses the smooth clip in `heterodyne_physics_kernel._fraction`.
- Checked: `HeterodyneModel.compute_fraction` delegates directly to the hard-clipped helper, and `compute_correlation` delegates to the smooth-clipped kernel. The registry permits the zero coefficients. The meshgrid/pointwise parity tests share the smooth fraction function and never compare it with the public accessor.
- Red test:
  ```python
  import numpy as np
  from xpcsjax import HeterodyneModel
  from xpcsjax.core.heterodyne_physics_kernel import _fraction

  model = HeterodyneModel.from_config({
      "temporal": {"dt": 0.1, "time_length": 4},
      "scattering": {"wavevector_q": 0.01},
  })
  p = model.get_params()
  p[9:13] = 0.
  np.testing.assert_allclose(model.compute_fraction(p), _fraction(model.t, *p[9:13]))
  ```
- Confidence: high

### core-B10: Parameter validation succeeds when it cannot convert a concrete parameter
- Location: xpcsjax/core/physics.py:203
- Severity: Minor
- Failure scenario: Validating `np.array([[999., 999.]])` against `[(0., 1.)]` returns `valid=True` with `parameters_checked=0`. A concrete row containing two out-of-bounds values passes validation.
- Root cause: The count check uses the leading dimension. After that check passes, scalar conversion of the row fails. The exception path logs and skips it, leaving the violations list empty.
- Checked: `ParameterManager.validate_parameters` and `PhysicsModelBase.validate_parameters` pass through the caller’s parameter array without a one-dimensional shape check. Their separate optimizer-bound checks do not protect these validation entry points.
- Red test:
  ```python
  import numpy as np
  from xpcsjax.core.physics import validate_parameters_detailed

  result = validate_parameters_detailed(np.array([[999., 999.]]), [(0., 1.)])
  assert not result.valid
  ```
- Confidence: high

## Non-blocking notes

- Reviewed bd0a549 and used standalone read-only probes. The test suite was left unrun as requested.

## Verdict
BUGS_FOUND
END_OF_REPORT