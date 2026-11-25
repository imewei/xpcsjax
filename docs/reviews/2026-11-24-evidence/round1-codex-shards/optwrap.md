# Codex review — shard optwrap

## Files read

xpcsjax/optimization/nlsq/wrapper.py — 4054 — full  
xpcsjax/optimization/nlsq/wrapper_stratified_route.py — 639 — full  
xpcsjax/optimization/nlsq/core.py — 3032 — full  
xpcsjax/optimization/nlsq/adapter.py — 1671 — full

## Bugs

### optwrap-B1: The standard wrapper fits unsorted observations against the wrong angles

- Location: xpcsjax/optimization/nlsq/wrapper.py:3759
- Severity: Required
- Failure scenario: A small laminar fit receives angle blocks in `[90°, 0°]` order. Predictions for `0°` are compared with observations for `90°` and vice versa. The fit can change the shear parameters to compensate.
- Root cause: `_prepare_xy_data` flattens observations in input order. The grid model evaluates `np.unique(phi)` in sorted order, then indexes its output by the original flat positions.
- Checked: Public `fit_nlsq` passes the data through `core._normalize_data_to_object` without sorting. The APS-old reader preserves detector row order. `test_phase5_standard_resolver.py` uses ascending angles, and the unsorted adapter regression exercises a different model.
- Red test:
  ```python
  from types import SimpleNamespace

  import numpy as np
  from xpcsjax.config.parameter_registry import AnalysisMode
  from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper

  w = NLSQWrapper()
  t = np.arange(1, 9) * .1
  d = SimpleNamespace(phi=np.array([0., 90.]), t1=t, t2=t,
                      g2=np.zeros((2, 8, 8)), sigma=None, q=.01, L=2e6, dt=.1)
  p = [.3, 1., 1000., .5, 10., .1, 0., 0., 0.]
  x = np.arange(d.g2.size)
  mode = AnalysisMode.LAMINAR_FLOW
  y = np.asarray(w._create_residual_function(d, mode, True, "averaged")(x, *p))
  d.phi, d.g2 = d.phi[::-1], d.g2[::-1]
  z = np.asarray(w._create_residual_function(d, mode, True, "averaged")(x, *p))
  np.testing.assert_allclose(z.reshape(2, -1), y.reshape(2, -1)[::-1])
  ```
- Confidence: high

### optwrap-B2: Unsorted angle input swaps per-angle uncertainty weights

- Location: xpcsjax/optimization/nlsq/wrapper.py:2347
- Severity: Required
- Failure scenario: Stratified data arrives in `[90°, 0°]` order with sigma `0.1` for `90°` and `1.0` for `0°`. The wrapper gives the 100-fold squared-residual weight to `0°`. The sequential route makes the same row-indexing error at line 3141.
- Root cause: `StratifiedData` preserves the source sigma grid in raw angle order while `.phi` is sorted. `_prepare_sigma_data` uses sorted indices directly on the raw grid, ignoring `.phi_raw`. The sequential residual likewise indexes raw sigma with indices derived from sorted angles.
- Checked: Core checks that sigma is positive and finite without changing row order. `test_prepare_sigma_data_phi_index.py` tests rounding with ascending sigma rows. Hybrid streaming explicitly remaps raw rows for its solver, confirming the two orders can differ.
- Red test:
  ```python
  from types import SimpleNamespace

  import numpy as np
  from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper

  d = SimpleNamespace(
      phi=np.array([0., 90.]), phi_raw=np.array([90., 0.]),
      phi_flat=np.array([90., 0.]), t1=np.array([0.]), t2=np.array([1.]),
      t1_flat=np.zeros(2), t2_flat=np.ones(2), sigma=np.array([[[.1]], [[1.]]]),
  )
  np.testing.assert_allclose(NLSQWrapper._prepare_sigma_data(None, d, 2), [.1, 1.])
  ```
- Confidence: high

### optwrap-B3: Sequential fitting crashes after solving with a restricted active-parameter list

- Location: xpcsjax/optimization/nlsq/wrapper.py:3218
- Severity: Required
- Failure scenario: A three-angle laminar fit forces the sequential route and excludes `D_offset` using `active_parameters`. The per-angle solver restores a 13-slot result. The wrapper indexes an eight-slot diagnostic bounds array as if it had 13 slots and raises `IndexError`.
- Root cause: The solve receives full bounds from `core.py`, with excluded positions narrowed by `fit`. `_run_sequential_optimization` separately builds diagnostic bounds from `ParameterManager.get_all_parameter_names()`, which filters out inactive names. With eight entries the array matches neither the compact nor the dense layout, so `_expand_compact_layout` leaves it short.
- Checked: `get_active_parameters()` preserves valid explicit subsets. `fit_nlsq_jax` does not catch this `IndexError`. The sequential integration tests use `fixed_parameters` alone, and the restricted-active integration test uses the standard route.
- Red test:
  ```python
  from tests.optimization.test_fixed_parameters_integration import (
      _ALL_PHYSICAL_NAMES, _config, _synthetic_data,
  )
  from xpcsjax.config import ConfigManager
  from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

  cfg = ConfigManager(config_override=_config(
      active_parameters=[p for p in _ALL_PHYSICAL_NAMES if p != "D_offset"],
      extra_top={"optimization": {"stratification": {"force_sequential_fallback": True}}},
  ))
  r = fit_nlsq_jax(_synthetic_data(), cfg)
  assert r.device_info["strategy"] == "sequential_per_angle"
  ```
- Confidence: high

### optwrap-B4: Averaged-mode covariance uses the compressed solver denominator

- Location: xpcsjax/optimization/nlsq/wrapper.py:2119
- Severity: Required
- Failure scenario: A standard laminar fit with three angles and 300 points solves nine parameters. The documented covariance convention uses 13 parameters in the variance denominator. The published covariance keeps the solver's denominator of 291 instead of 287.
- Root cause: NLSQ scales covariance by the optimizer vector length. The wrapper computes the expanded parameter count for reduced chi-squared, then broadcasts the solver covariance without adjusting its scale. The optional sampled covariance uses a separate diagnostics field and cannot correct the public covariance.
- Checked: `test_phase5_dof_semantics.py` pins the diagnostic count and basis. It does not check the public covariance scale. The installed NLSQ `CurveFit` computes relative-sigma covariance using `ysize - p0.size`.
- Red test:
  ```python
  import numpy as np
  import pytest
  from tests.optimization.test_fixed_parameters_integration import _config, _synthetic_data
  from xpcsjax.config import ConfigManager
  from xpcsjax.optimization.nlsq.core import fit_nlsq_jax
  from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper

  def test_averaged_covariance_dof(monkeypatch):
      def solve(self, **kw):
          p = kw["validated_params"]
          assert len(p) == 9
          return p, 2 * np.eye(9), {}, [], "converged"

      monkeypatch.setattr(NLSQWrapper, "_execute_optimization_with_fallback", solve)
      r = fit_nlsq_jax(_synthetic_data(n_t=10, n_phi=3),
                       ConfigManager(config_override=_config()))
      assert r.covariance[-1, -1] == pytest.approx(2 * (300 - 9) / (300 - 13))
  ```
- Confidence: high

### optwrap-B5: Restored fixed physical parameters are counted as fitted in reduced chi-squared

- Location: xpcsjax/optimization/nlsq/wrapper.py:1759
- Severity: Required
- Failure scenario: A 300-point, three-angle averaged laminar fit fixes `D_offset`. The denominator is `300 - 13` instead of `300 - 12`, biasing reduced chi-squared upward. The same fixed-slot overcount occurs in the sequential wrapper, stratified-LS result assembly, CMA-ES result assembly, and adapter result conversion.
- Root cause: The solvers remove fixed slots, then restore them for reporting. The affected result builders count the restored vector or use the full physical count. See also `wrapper.py:3211`, `wrapper_stratified_route.py:578`, `core.py:2906`, and `adapter.py:1234`.
- Checked: `test_fixed_parameters_integration.py` checks restored values and uncertainties on these routes without asserting the chi-squared denominator. The stratified-LS solver itself uses the free count when computing the variance scale. The hybrid-streaming result branch already accounts for fixed slots.
- Red test: In `test_fixed_parameter_survives_real_fit`, after the existing parameter checks, add:
  ```python
  assert np.isfinite(result.chi_squared) and result.chi_squared > 0
  assert result.reduced_chi_squared == pytest.approx(
      result.chi_squared / (data["g2"].size - (len(params) - 1))
  )
  ```
- Confidence: high

### optwrap-B6: The stratified route can report zero chi-squared for an imperfect unweighted fit

- Location: xpcsjax/optimization/nlsq/wrapper_stratified_route.py:535
- Severity: Required
- Failure scenario: The stratified-LS route fits noisy observations with `stratified_data.sigma=None`. The solver sees unit weights, but result assembly reports zero chi-squared and a `good` quality flag even when off-diagonal residuals remain. The streaming result branch uses the same reporting evaluator at line 385.
- Root cause: Result assembly rebuilds residuals with `StratifiedResidualFunction`. Unlike the JIT solver evaluator, it converts `None` to a floating-point NaN and masks every sigma comparison as invalid. The resulting zero residual vector is used to compute the published statistics.
- Checked: `create_stratified_chunks` preserves `None`, and `StratifiedResidualFunctionJIT` explicitly supports the unit-sigma sentinel. `test_fixed_parameter_survives_stratified_ls_fit` supplies that sentinel and noisy data, then checks the route and the fixed value without checking chi-squared.
- Red test: Add this assertion to `test_fixed_parameter_survives_stratified_ls_fit` immediately after its existing route assertion:
  ```python
  assert np.isfinite(result.chi_squared) and result.chi_squared > 0
  ```
- Confidence: high

### optwrap-B7: Adapter flattening misclassifies shared time axes when the frame count is divisible by the angle count

- Location: xpcsjax/optimization/nlsq/adapter.py:1116
- Severity: Required
- Failure scenario: Three angles and six common time samples yield a valid `(3, 6, 6)` observation cube. The adapter generates six predictor rows for 108 observations. CurveFit rejects the lengths, and `fit_nlsq_jax(use_adapter=True)` falls back to the wrapper.
- Root cause: The divisibility shortcut assumes the time vectors are already flattened across angles without comparing their length to `g2`. It runs before the grid reconstruction branch even when the observation shape identifies a shared grid.
- Checked: `_normalize_data_to_object` gives the adapter one-dimensional time marginals. CurveFit checks predictor and observation lengths. `test_adapter_flatten_phi_order.py` tests already-flat inputs, and the three-angle fixed-parameter integration fixture uses ten frames, which bypass the faulty condition.
- Red test:
  ```python
  import numpy as np
  from xpcsjax.optimization.nlsq.adapter import NLSQAdapter

  t = np.arange(6.)
  data = {"phi": np.array([0., 45., 90.]), "t1": t, "t2": t,
          "g2": np.zeros((3, 6, 6))}
  x, y, _ = NLSQAdapter()._flatten_xpcs_data(data)
  assert x.shape == (y.size, 3)
  ```
- Confidence: high

### optwrap-B8: The adapter replaces the laminar shear gap with 1 Å

- Location: xpcsjax/optimization/nlsq/adapter.py:611
- Severity: Required
- Failure scenario: A laminar fit supplies a 2,000,000 Å stator–rotor gap. The adapter uses 1 Å in the shear term, so it can return materially biased shear parameters even when the solve converges.
- Root cause: `_build_model_function` extracts `q` and `dt` without extracting `L`. The cached closure passes `1.0` into `compute_g1_batch`, as does the uncached closure at line 1018. `compute_g1_batch` multiplies `L` directly into the shear prefactor.
- Checked: Core resolves `L` from the input or config before calling the adapter. `test_adapter_xdata_cache.py` stubs the physics and ignores `L`. The fixed-parameter adapter tests assert restoration and backend selection, without checking the shear predictions.
- Red test:
  ```python
  import jax.numpy as jnp
  import numpy as np
  from xpcsjax.config.parameter_registry import AnalysisMode
  from xpcsjax.core.models import CombinedModel
  from xpcsjax.optimization.nlsq.adapter import NLSQAdapter

  def test_adapter_passes_gap(monkeypatch):
      gaps = []

      def g1(self, p, t1, t2, phi, q, L, dt):
          gaps.append(L)
          return jnp.ones_like(t1)

      monkeypatch.setattr(CombinedModel, "compute_g1_batch", g1)
      data = {"phi": np.array([0.]), "q": .005, "L": 2e6, "dt": .001}
      f, *_ = NLSQAdapter()._build_model_function(
          data, {}, AnalysisMode.LAMINAR_FLOW, True, 1
      )
      f(np.array([[0., .001, 0.]]), .3, .8, 8000., -1.2, 50., .01, .1, 0., 0.)
      assert gaps == [2e6]
  ```
- Confidence: high

### optwrap-B9: Constant-mode CMA-ES freezes scaling on the wrong detector rows

- Location: xpcsjax/optimization/nlsq/core.py:2516
- Severity: Required
- Failure scenario: A constant-mode CMA-ES fit receives angles in descending order with different offsets. The first observation block gets the offset estimated from the last block. The optimizer cannot correct the swap because the scaling values are frozen.
- Root cause: `compute_quantile_per_angle_scaling` returns scaling arrays in sorted angle order. `fit_nlsq_cmaes` builds `phi_indices` in incoming row order and uses them to index those sorted arrays. The same sorted arrays are prepended to the result at line 2812.
- Checked: `_validate_data` checks required keys and nonempty observations without sorting. The loader preserves positional order. `test_cmaes_trigger.py` exercises constant mode with ascending angles and verifies convergence and output shape.
- Red test:
  ```python
  import numpy as np
  from tests.optimization.test_cmaes_trigger import _laminar_cmaes_config, _tiny_laminar_data
  from xpcsjax.optimization.nlsq import parameter_utils
  from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESResult, CMAESWrapper
  from xpcsjax.optimization.nlsq.core import fit_nlsq_cmaes

  def test_constant_cmaes_keeps_row_alignment(monkeypatch):
      d = _tiny_laminar_data(n_phi=3)
      for key in ("phi_angles_list", "c2_exp"):
          d[key] = d[key][::-1]
      cfg = _laminar_cmaes_config("constant")
      cfg.config["optimization"]["nlsq"]["cmaes"]["nlsq_warmstart"] = False
      monkeypatch.setattr(parameter_utils, "compute_quantile_per_angle_scaling",
                          lambda **_: (np.zeros(3), np.array([.8, .9, 1.])))

      def fit(self, model_func, xdata, p0, **kw):
          y = np.asarray(model_func(xdata, *p0)).reshape(3, -1)
          np.testing.assert_allclose(y[:, 0], [1., .9, .8])
          return CMAESResult(np.asarray(p0), None, 1., True)

      monkeypatch.setattr(CMAESWrapper, "fit", fit)
      fit_nlsq_cmaes(d, cfg)
  ```
- Confidence: high

### optwrap-B10: Homodyne routing ignores the configured memory budget

- Location: xpcsjax/optimization/nlsq/wrapper_stratified_route.py:215
- Severity: Required
- Failure scenario: A user caps fitting at 25% of available RAM. A fit with an estimated peak of 50% can still select the in-memory tier under the 75% default, exceeding the configured budget.
- Root cause: Both `NLSQWrapper.fit` strategy calls and the stratified recheck omit the configured fraction. The route reads `memory_fraction` and `memory_threshold_gb` after the strategy decision, where the computed threshold only reaches logging.
- Checked: The homodyne YAML templates document both settings. `select_nlsq_strategy` accepts a fraction, and the memory tests exercise that helper directly. The wrapper's routing tests force decisions through monkeypatches and do not assert that the configured budget reaches the selector.
- Red test:
  ```python
  from tests.optimization.test_fixed_parameters_integration import _config, _synthetic_data
  from xpcsjax.config import ConfigManager
  from xpcsjax.optimization.nlsq import wrapper
  from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

  def test_wrapper_routes_with_configured_budget(monkeypatch):
      actual = wrapper.select_nlsq_strategy
      seen = []

      def select(n_points, n_params, memory_fraction=.75, **kw):
          seen.append(memory_fraction)
          return actual(n_points, n_params, memory_fraction, **kw)

      monkeypatch.setattr(wrapper, "select_nlsq_strategy", select)
      cfg = ConfigManager(config_override=_config(
          extra_top={"optimization": {"nlsq": {"memory_fraction": .25}}},
      ))
      fit_nlsq_jax(_synthetic_data(), cfg)
      assert seen and all(fraction == .25 for fraction in seen)
  ```
- Confidence: high

## Non-blocking notes

- Static review of `bd0a549`. The red-test snippets are unexecuted.

## Verdict

BUGS_FOUND

END_OF_REPORT