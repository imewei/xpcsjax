# Codex round 4 report
## A. Fix review
### R4-A1: R3-A3 — Stratified covariance counts index diagonals instead of intersecting the masks
- Location: xpcsjax/optimization/nlsq/strategies/stratified_ls.py:172
- Severity: Required
- Must-fix: yes
- Failure scenario: The solver accepts separate `t1` and `t2` axes. With one axis shifted by a frame, cells on the array diagonal can be real observations. If such a cell has masked sigma, the new counter skips subtracting it. In the real stratified solver probe with five 12-frame slabs, the covariance builder received 665 observations instead of 664. The residual uses the intersection of the time and sigma masks. The count assumes a common axis even though the public homodyne fit accepts separate delay arrays.
- Red test: Extend `test_r3_a3_row_count_uses_engine_thresholds` using `t1, t2 = (a.ravel() for a in np.meshgrid(np.arange(4.), np.arange(1., 5.), indexing="ij"))` and set `sigma[0,0,0] = 1e-12` as the only masked sigma. Assert `_n_contributing_rows(data) == np.count_nonzero((np.abs(t1-t2) > 1e-15) & (sigma.ravel() > 1e-10))`. Actual is 13, expected 12. To verify the covariance seam, shift `d.t2_flat` and `d.t2` by 1 in `tests.parity.test_laminar_execute_layers._build_laminar_stratified_data` and run `_fit(False)` with a spy on `gauss_newton_covariance`.
- Confidence: high

### R4-A2: R3-A5 — Accepted NPZ phi columns still break plotting
- Location: xpcsjax/data/xpcs_loader.py:273, xpcsjax/data/npz_cache.py:206
- Severity: Required
- Must-fix: yes
- Failure scenario: Two correlation matrices with phi `[[0.], [45.]]` pass the loader's intended column-vector compatibility case. The loader returns shape `(2,1)` unchanged. Current NumPy raises `TypeError` at `xpcsjax/viz/nlsq_plots.py:1839` when the public plotter tries to turn a row into a scalar. The standalone simulated plot skips both angles for the same reason. A flat vector makes both valid angle values available.
- Red test: Adapt `tests/review_r1/test_codex_r2.py::test_r3_a5_phi_row_vector_is_rejected` to write `phi_angles_list=np.array([[0.], [45.]])` with the same two surfaces. Assert `np.testing.assert_array_equal(load_from_cache(_bare_loader(), buf)["phi_angles_list"], [0., 45.])`. The loaded values have shape `(2,1)`. Passing that load to `generate_nlsq_plots` reproduces the scalar-conversion failure.
- Confidence: high

### R4-A3: R3-A8 — Zero-weight observations still make a zero-DOF fit report covariance
- Location: xpcsjax/optimization/nlsq/heterodyne_core.py:1682, xpcsjax/optimization/nlsq/heterodyne_core.py:3819
- Severity: Required
- Must-fix: yes
- Failure scenario: The residual vector retains zero entries for observations excluded by zero weights, so its size can still overstate data DOF. Three angles, four frames and four fixed physics parameters give 16 optimizer coordinates. Masking one symmetric off-diagonal pair leaves 16 actual data rows. Real adaptive and unregularized fits both converge with the same SSR and publish 256 finite covariance entries, even though no residual variance can be estimated with zero data DOF. The averaged branch uses the same residual-size assumption.
- Red test:
  ```python
  import numpy as np
  from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
  from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
  from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

  def run(mode):
      m, c2, phi = make_synthetic_two_component(n_phi=3, n_t=4)
      for name in ("D_offset_ref", "D_offset_sample", "v0", "v_offset"):
          m.param_manager.set_vary(name, False)
      w = np.ones_like(c2)
      w[0, 1, 2] = w[0, 2, 1] = 0.
      cfg = NLSQConfig(per_angle_mode="individual", regularization_mode=mode,
                       group_variance_lambda=0., loss="linear", max_nfev=100)
      return fit_nlsq_multi_phi(m, c2, phi, cfg, weights=w)

  a, b = run("adaptive"), run("none")
  np.testing.assert_allclose(a.chi_squared, b.chi_squared, rtol=1e-9)
  assert a.nlsq_diagnostics["covariance_is_placeholder"]
  assert b.nlsq_diagnostics["covariance_is_placeholder"]
  ```
- Confidence: high

### R4-A4: R3-C2 — The standard executor still declares a failed fit converged via its top-level status
- Location: xpcsjax/optimization/nlsq/strategies/executors.py:210
- Severity: Required
- Must-fix: yes
- Failure scenario: Once a real backend fit exhausts its evaluation budget, the exported executor now returns `info["success"] = False` but also `convergence_status="converged"`. A consumer gating on the explicit convergence status accepts an unfinished parameter vector. The sibling large-dataset executor already derives this status from the backend verdict.
- Red test: Extend `tests/review_r1/test_codex_r2.py::test_r3_c2_standard_executor_reports_backend_verdict` by asserting `r.convergence_status != "converged"` after its existing successful `r.info["success"] is False` assertion. The one-evaluation solve still returns `"converged"`.
- Confidence: high

### R4-A5: R3-A2 — Shared non-unit weights now reach an engine that fits the wrong robust objective
- Location: xpcsjax/optimization/nlsq/heterodyne_engine_route.py:453
- Severity: Required
- Must-fix: yes
- Failure scenario: Before the shared-slab broadcast repair, a 2-D slab on a multi-angle fit reached the weighted fallback when engine scoring failed. The repaired path now fits without weights and applies the supplied weights only during scoring. Under the default `soft_l1` loss, uniform non-unit weights change the residual scale relative to the robust transition, so they can change the minimizer. With a shared weight of 1000 and one symmetric outlier pair, the public fit reports failure and a recomputed weighted objective of 345.09. The weighted fallback converges at 242.89 on the same data. This is a routing regression for a broadcast form, separate from changing the older engine policy for inputs it already handled.
- Red test:
  ```python
  from unittest.mock import patch
  import numpy as np
  from tests.optimization._heterodyne_fixtures import _build_cfgmgr
  from tests.review_r1.test_codex_opthet import _unsorted_two_angle
  from xpcsjax import fit_nlsq
  from xpcsjax.core.heterodyne_jax_backend import compute_multi_angle_residuals

  m, c2, phi = _unsorted_two_angle()
  c2[:, 1, 2] += 1
  c2[:, 2, 1] += 1
  weights = np.full(c2.shape[1:], 1000.)
  cfg = _build_cfgmgr(2, 12)
  cfg.config["optimization"]["nlsq"].update(
      per_angle_mode="individual", loss="soft_l1", max_nfev=100)
  data = dict(c2_exp=c2, phi_angles_list=phi, weights=weights)
  actual = fit_nlsq(data, cfg)
  with patch("xpcsjax.optimization.nlsq.heterodyne_engine_route.fit_two_component_via_engine",
             side_effect=NotImplementedError):
      weighted = fit_nlsq(data, cfg)

  def objective(result):
      p = result.parameters
      e = np.asarray(compute_multi_angle_residuals(
          p[4:], m.t, m.q, m.dt, phi, c2, np.broadcast_to(weights, c2.shape),
          p[:2], p[2:4]))
      return 2 * np.sum(np.sqrt(1 + e**2) - 1)

  assert objective(actual) <= 1.001 * objective(weighted)
  ```
- Confidence: high

The original shared uniform weight, legacy cache, YAML escaping, degradation reset, overflow validation, same-shape HDF dtype budget, floating-point reconstruction and measured-q plotting cases passed. The reported fixes for zero sigma on aligned axes and zero data DOF without an observation mask also passed their original cases. Verified `a0b6540` with 25 passing tests in the single serial regression file, plus focused probes for the legacy cache and HDF cases. The full suite and golden parity checks were skipped.

## B. Disputes
| id | Ruling | Must-fix | Evidence or disposition |
| --- | --- | --- | --- |
| R3-A1 | CONCEDE | no | Maintainer decision. The current duplicate layout predates the review. Changing one label list would leave the engine's bounds and grouped scaling contract unresolved. Choose and apply one duplicate-angle policy across both solver routes and their public diagnostics. |

## C. New findings
### R4-C1: Stateful homodyne plotting — Each measured-q replacement keeps the previous cached model alive
- Location: xpcsjax/viz/nlsq_plots.py:557
- Severity: Minor
- Must-fix: no
- Failure scenario: Reusing one model across different measured q values grows a chain of old twins. `copy.copy(model)` preserves the previous `_fit_q_twin` on the copy. When the original replaces its cache tuple, the new twin keeps the old one reachable. The numerical surface is correct. The retention matters for long-lived consumers that cycle through many distinct q values.
- Red test:
  ```python
  import gc
  import weakref
  from xpcsjax.core.homodyne_model import HomodyneModel
  from xpcsjax.viz.nlsq_plots import _stateful_model_at_fit_q

  cfg = {"analysis_mode": "static_isotropic",
         "analyzer_parameters": {"dt": .1, "start_frame": 1, "end_frame": 4,
                                 "scattering": {"wavevector_q": .0054},
                                 "geometry": {"stator_rotor_gap": 1.}}}
  m = HomodyneModel(cfg)
  old = weakref.ref(_stateful_model_at_fit_q(m, {"wavevector_q_list": [.006]}))
  _stateful_model_at_fit_q(m, {"wavevector_q_list": [.007]})
  gc.collect()
  assert old() is None
  ```
- Confidence: high

### R4-C2: HDF5 quality preflight — A larger later bin can still exceed the byte cap before selection
- Location: xpcsjax/data/hdf5_readers.py:129
- Severity: Required
- Must-fix: no
- Failure scenario: The widest dtype now controls the estimate, but the guard still multiplies by the first bin's dimensions. A malformed APS-U file with 4×4 and 8×8 float64 candidates passes a 256-byte preflight. The real quality reader reconstructs 640 bytes before selection. The same shared guard serves APS-old. This is a mixed-frame-shape input, which is outside the supported same-shape contract.
- Red test:
  ```python
  import io
  import h5py
  import pytest
  from tests.review_r1.test_codex_data import _half, _write_aps_u
  from xpcsjax.data import xpcs_loader
  from xpcsjax.data.hdf5_readers import guard_aps_u_intermediate_allocation

  buf = io.BytesIO()
  _write_aps_u(buf, [_half(1.), _half(1., n_t=8)], [0., 90.])
  buf.seek(0)
  with h5py.File(buf, "r") as f, pytest.MonkeyPatch.context() as mp:
      groups = f["xpcs/twotime/correlation_map"]
      mp.setattr(xpcs_loader, "MAX_CORRELATION_ALLOC_BYTES", 256)
      with pytest.raises(xpcs_loader.XPCSDataFormatError):
          guard_aps_u_intermediate_allocation(groups, sorted(groups), [0, 1],
                                              source="candidate probe")
  ```
- Confidence: high

### R4-C3: Weighted heterodyne fallback — A nonuniform shared singleton slab crashes the public fit
- Location: xpcsjax/optimization/nlsq/heterodyne_core.py:3300
- Severity: Required
- Must-fix: yes
- Failure scenario: A shared mask of shape `(1, N, N)` is accepted as a broadcast form by the stratified builder and the new engine normalization. Nonuniform weights deliberately select the weighted fallback. The engine expands only its local variable, so the public dispatcher hands the original singleton slab to the fallback. The individual builder then indexes the second angle's nonexistent weight slab. The equivalent 2-D mask succeeds.
- Red test:
  ```python
  import numpy as np
  from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data
  from xpcsjax import fit_nlsq

  cfg, d = make_cfgmgr_and_data(2, 8)
  w = np.ones_like(d["c2_exp"][0])
  w[1, 2] = w[2, 1] = .25
  a = fit_nlsq(dict(d, weights=w), cfg)
  assert np.isfinite(a.chi_squared)
  b = fit_nlsq(dict(d, weights=w[None]), cfg)
  np.testing.assert_allclose(a.chi_squared, b.chi_squared)
  ```
- Confidence: high

## Verdict
BUGS_FOUND
END_OF_REPORT
