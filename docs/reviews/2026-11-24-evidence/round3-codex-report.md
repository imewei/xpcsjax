# Codex round 3 report
## A. Fix review
### R3-A1: R2-A1 — Duplicate-angle diagnostics still mix raw rows and shared slots
- Location: xpcsjax/optimization/nlsq/heterodyne_stratified_ls.py:1301, xpcsjax/optimization/nlsq/heterodyne_stratified_ls.py:1460, xpcsjax/optimization/nlsq/heterodyne_engine_route.py:574
- Severity: Required
- Failure scenario: Skipping the permutation keeps the physics tail intact. However, on `[0, 50, 100, 0]`, stratified LS still reports four angle labels for three per-angle chi-squared values and three fitted scaling slots. For input `[100, 50, 0, 100]`, the first reported angle is 100 even though the first shared chi-squared slot is for 0. The direct engine also sizes its initial vector for three distinct angles and its bounds for four raw rows. A bounds spy sees shapes `(20,)` and `(22,)` before the solver. The public dispatcher can fall back from the engine, but the large stratified path builds its own inconsistent result.
- Red test: Extend `tests/review_r1/test_codex_r2.py::test_r2_a1_duplicate_angles_keep_physics` by asserting `np.testing.assert_array_equal(a.nlsq_diagnostics["phi_angles"], np.unique(phi))` and `len(a.nlsq_diagnostics["chi2_per_angle"]) == len(a.nlsq_diagnostics["phi_angles"])`. The physics assertions pass. The reported angle length is four and the chi-squared length is three. To check the direct engine with the same fixture, run `fit_two_component_via_engine` with an adapter spy that asserts `initial_params.shape == bounds[0].shape`.
- Confidence: high

### R3-A2: R2-A2 — Shared weight slabs still fail during engine scoring
- Location: xpcsjax/optimization/nlsq/heterodyne_engine_route.py:293, xpcsjax/optimization/nlsq/heterodyne_engine_route.py:450
- Severity: Required
- Failure scenario: The stratified-LS fix preserves shared slabs and passes. The engine now reaches scoring but supplies singleton angle weights to a residual that vmaps over two angles. This also affects a 2-D shared matrix because the entry point adds its singleton angle dimension before scoring. The unweighted control succeeds. The uniform weighted engine path throws, so the production dispatcher takes a different solver route.
- Red test:
  ```python
  import numpy as np
  import pytest
  from tests.review_r1.test_codex_opthet import _patch_exact_fit, _unsorted_two_angle
  from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
  from xpcsjax.optimization.nlsq.heterodyne_engine_route import fit_two_component_via_engine

  @pytest.mark.parametrize("singleton", [False, True])
  def test_engine_broadcasts_shared_weight_slab(monkeypatch, singleton):
      model, c2, phi = _unsorted_two_angle()
      _patch_exact_fit(monkeypatch, {})
      cfg = NLSQConfig(per_angle_mode="individual")
      assert fit_two_component_via_engine(model, c2, phi, cfg).chi_squared < 1e-12
      weights = np.ones_like(c2[:1] if singleton else c2[0])
      result = fit_two_component_via_engine(model, c2, phi, cfg, weights=weights)
      assert result.chi_squared < 1e-12
  ```
- Confidence: high

### R3-A3: R2-A7 — The covariance count diverges from the actual residual mask
- Location: xpcsjax/optimization/nlsq/strategies/stratified_ls.py:172, xpcsjax/optimization/nlsq/strategies/stratified_ls.py:176
- Severity: Required
- Failure scenario: The JIT engine masks sigma `<= 1e-10` and time differences `<= 1e-15` at `xpcsjax/optimization/nlsq/strategies/residual_jit.py:506–516`. The new helper counts strictly positive sigma and compares time values for exact equality. An off-diagonal sigma of `1e-12` is masked in the solve but is still counted in the covariance denominator. The regression test covers exact zero only.
- Red test:
  ```python
  import numpy as np
  from tests.parity import test_laminar_execute_layers as fixture
  from xpcsjax.optimization.nlsq.strategies import stratified_ls

  def test_covariance_count_uses_solver_sigma_threshold(monkeypatch):
      build = fixture._build_laminar_stratified_data
      def masked_input():
          data, *rest = build()
          data.sigma[0, 0, 1] = 1e-12
          return data, *rest

      seen = []
      def covariance_spy(f, J, *, n_valid, n_params, loss):
          seen.append(n_valid)
          return np.eye(np.asarray(J).shape[1])

      monkeypatch.setattr(fixture, "_build_laminar_stratified_data", masked_input)
      monkeypatch.setattr(stratified_ls, "gauss_newton_covariance", covariance_spy)
      fixture._fit(False)
      assert seen == [659]
  ```
- Confidence: high

### R3-A4: R2-A9 — Legacy cache validation misidentifies direct NPZ sources and misses the all-frames request
- Location: xpcsjax/data/npz_cache.py:510, xpcsjax/data/npz_cache.py:518
- Severity: Required
- Failure scenario: A user points `file_path` at a standalone pre-fingerprint NPZ with applied frames 1–4 and requests an end frame beyond 4. The existence test sees the NPZ itself and raises `CacheStaleError` as if a raw HDF5 reload were possible. The direct NPZ path propagates the error. Conversely, a four-frame HDF5 source and a pre-fingerprint automatic cache for frames 1–2 still return two frames when the public loader is asked for `end_frame=-1`. The loader treats `-1` as all frames but the cache comparison never tests that case.
- Red test: For the direct case, write `_qc_data()` from `tests.review_r1.test_codex_data` to a temporary NPZ with `cache_metadata_json=json.dumps({"start_frame":1,"end_frame":4})`. Call the real `load_xpcs_data` with `experimental_data.file_path` pointing there and requested frames 1–8. Assert it returns the four cached frames. The fixture raises `CacheStaleError`. For the automatic case, extend `test_data_b7`: remove only the two requested-frame keys from the first cache's JSON and set `end_frame=-1` on the second load. Keep the original 4×4 assertion. The actual result is 2×2.
- Confidence: high

### R3-A5: R2-A10 — Size-only phi validation admits row vectors that downstream treats as a single angle
- Location: xpcsjax/data/xpcs_loader.py:273, xpcsjax/data/npz_cache.py:206
- Severity: Required
- Failure scenario: A plain NPZ with two surfaces and phi shaped `(1, 2)` now passes validation. The cache returns the same two-dimensional array. The standalone simulated plot iterates it and calls `float` on a two-element row. The heterodyne dispatcher also uses the leading length as the angle count. The previous first-axis validation rejected this input. Previously allowed column vectors can remain supported by canonicalizing them at the boundary.
- Red test: In `tests/review_r1/test_codex_r2.py::test_r2_a10_npz_has_one_phi_per_surface`, use `phi_angles_list=np.array([[0., 45.]])` and keep the two surfaces. Assert that `load_from_cache` raises `XPCSDataFormatError` or returns a flat angle vector `[0., 45.]`. The actual cache result has shape `(1, 2)`. `[float(phi) for phi in resolve_phi_angles_for_sim(None, result)]` then raises `TypeError`.
- Confidence: high

### R3-A6: R2-A13 — Raw Unicode in the newline workaround still corrupts valid paths
- Location: xpcsjax/cli/config_template.py:154
- Severity: Minor
- Failure scenario: An emoji and newline now round-trip. For a filename containing U+0085 and a newline, PyYAML treats U+0085 as a line break and folds it into a space. A filename with U+007F and a newline raises `yaml.reader.ReaderError`. Both characters were escaped by the previous ASCII JSON branch. The failure occurs in the output of the real config generator.
- Red test: Parameterize `test_r2_a13_newline_path_with_non_bmp_char_round_trips` over `["scan\u0085test\n.h5", "scan\x7ftest\n.h5"]`, keeping the generated-YAML round-trip assertion.
- Confidence: high

### R3-A7: R2-C4 — Cache replay still leaks degradation across loader calls
- Location: xpcsjax/data/npz_cache.py:152, xpcsjax/data/xpcs_loader.py:1012
- Severity: Required
- Failure scenario: Deduplication only tests the replay-formatted string against instance history. A cold load records the bare reason and the first warm hit adds the same reason with the replay suffix. If the next loaded dataset is clean, the same public loader still reports the previous reason and `_degraded=True`. A caller deciding whether to trust the current data gets a false rejection.
- Red test: Adapt `test_data_b8` to instantiate one `XPCSDataLoader(config_dict=cfg)` and call `load_experimental_data()` twice. Assert the warm result has one degradation, like the cold result. It has two. To exercise clean reload without changing the loader object, feed two direct NPZ inputs sequentially through a spy on `npz_cache.np.load`, the first with `load_degradations=["fallback"]` and the second with an empty list. Assert the second public load has `_degraded is False`. It reports the first cache's reason.
- Confidence: high

### R3-A8: R2-A3 — Zero data degrees of freedom still yield a finite covariance
- Location: xpcsjax/optimization/nlsq/heterodyne_core.py:3812
- Severity: Required
- Failure scenario: Three angles with four frames and two fixed physics parameters yield 18 contributing rows and 18 optimizer coordinates. Two identically zero penalty rows give the backend a denominator of two. At zero data DOF the new guard skips scaling and retains 324 finite covariance entries with the placeholder flag false. The same unregularized solve converges with the same SSR and correctly reports a placeholder.
- Red test:
  ```python
  import numpy as np
  from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
  from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
  from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

  def run(mode):
      m, c2, phi = make_synthetic_two_component(n_phi=3, n_t=4)
      m.param_manager.set_vary("D_offset_ref", False)
      m.param_manager.set_vary("D_offset_sample", False)
      cfg = NLSQConfig(per_angle_mode="individual", regularization_mode=mode,
                       group_variance_lambda=0., loss="linear", max_nfev=100)
      return fit_nlsq_multi_phi(m, c2, phi, cfg)

  a, b = run("adaptive"), run("none")
  np.testing.assert_allclose(a.chi_squared, b.chi_squared, rtol=1e-9)
  assert b.nlsq_diagnostics["covariance_is_placeholder"]
  assert a.nlsq_diagnostics["covariance_is_placeholder"]
  ```
- Confidence: high

### R3-A9: data-B17 — A numeric config value can still escape schema validation as an exception
- Location: xpcsjax/data/config.py:346, xpcsjax/data/validators.py:119
- Severity: Minor
- Failure scenario: YAML accepts an integer timestep too large to convert to a float. The schema accepts the integer type and the semantic check raises `OverflowError` in `math.isfinite`. The new exception guard covers the malformed-string case and leaves the public error-report contract broken for this input.
- Red test: In `test_r2_data_b17_schema_validation_reports_instead_of_raising`, replace `"dt": "bad"` with `"dt": 10**400`. The same assertion should be reached on the returned report. The call raises `OverflowError`.
- Confidence: high

### R3-A10: R2-C2 — Mixed-width HDF quality candidates can still exceed preflight allocation limits
- Location: xpcsjax/data/hdf5_readers.py:127
- Severity: Required
- Failure scenario: With two 4×4 quality candidates, first uint8 and second float64, the shared preflight now assumes two promoted float32 matrices and accepts a 128-byte cap. APS-U then retains 64-byte and 128-byte reconstructed candidates before selecting one, exceeding the cap by 64 bytes without a later rejection. The APS-old quality path calls the same guard. The single-dtype preallocated read paths now use the corrected width.
- Red test:
  ```python
  import io
  import numpy as np
  import pytest
  from tests.review_r1.test_codex_data import _bare_loader, _half, _write_aps_u
  from xpcsjax.data import xpcs_loader

  def test_hdf_guard_counts_all_candidate_widths(monkeypatch):
      hdf = io.BytesIO()
      _write_aps_u(hdf, [_half(100, dtype=np.uint8), _half(1.2)], [0., 90.])
      hdf.seek(0)
      loader = _bare_loader({"data_filtering": {"enabled": True, "quality_threshold": .9}})
      monkeypatch.setattr(xpcs_loader, "MAX_CORRELATION_ALLOC_BYTES", 128)
      with pytest.raises(xpcs_loader.XPCSDataFormatError):
          loader._load_from_hdf(hdf)
  ```
- Confidence: high

### R3-A11: R2-A8 — Mixed floating-point bins no longer use their source arithmetic on selective reads
- Location: xpcsjax/data/hdf5_readers.py:94
- Severity: Minor
- Failure scenario: The readers document source-dtype arithmetic for float files. A float32 second bin in a float64 output buffer is now added in float64 on the selective path. A 4×4 APS-U half with upper pair value 1 and lower pair value `2**-25` is accepted by both paths. The selective path now returns `1.0000000298023224`, whereas the quality reconstructor and the previous selective path round the same float32 addition to 1. The integer overflow fix does not require widening floating operands before addition.
- Red test: Extend the in-memory APS-U fixture from R3-A10 with a float64 first bin and `half = _half(1., dtype=np.float32); half[1, 0] = 2**-25` as the second. Use no quality filtering or patched allocation limit. Assert `out["c2_exp"][1, 0, 1] == (half + half.T)[0, 1]` after `loader._load_from_hdf(hdf)`. The two values differ.
- Confidence: high

R2-A6 passed at all four sites, including separate CMA-ES and stratified reporter probes. The integer reconstruction, per-key q fallback, grouped heterodyne preview, unbounded standard executor, canonical CLI parameter order, and symlink save fixes satisfy the original cases. The explicit Datashader singleton validation error is acceptable. I accept the streaming refinement reclassification in R2-A5 as a maintainer decision.

## B. Disputes
| id | Ruling | Evidence or disposition |
| --- | --- | --- |
| R2-A4 | CONCEDE | Maintainer decision. Existing callers use `CMAESResult.success` for candidate eligibility, so changing the flag in isolation can change the selected fit. The refinement diagnostic reports local convergence already. |
| core-B1 | CONCEDE | Maintainer decision. Tail-clamp provenance must survive both HDF5 and cache loading. The dispatcher receives a length that cannot distinguish a tail clamp from leading-frame exclusion. |
| optstrat-B12 | CONCEDE | Maintainer decision with optstrat-B10/B11. The exported executor's defect is real. Its repair depends on the same chunk-local model callback contract and on whether the streaming tier will be retained. |

## C. New findings
### R3-C1: Stateful homodyne plots — The measured-q rebuild adds mandatory frame fields to plotting config
- Location: xpcsjax/viz/nlsq_plots.py:555
- Severity: Required
- Failure scenario: The public plot interface accepts a fully built `HomodyneModel` separately from a config that supplies mode, q, geometry and dt. With that documented config and matching q, evaluating one 4×4 surface works. When measured q differs, rebuilding a model from plotting config raises `KeyError` on the missing frame bounds. The public plot loop catches the evaluation error, leaving no fitted surface. With a complete but different plotting config, the same rebuild can change the supplied model's time grid and other physics along with q.
- Red test:
  ```python
  from copy import deepcopy
  from types import SimpleNamespace
  import numpy as np
  from xpcsjax.core.homodyne_model import HomodyneModel
  from xpcsjax.viz.nlsq_plots import _evaluate_c2_per_angle

  cfg = {"analysis_mode": "static_isotropic",
         "analyzer_parameters": {"dt": .1, "start_frame": 1, "end_frame": 4,
                                 "scattering": {"wavevector_q": .0054},
                                 "geometry": {"stator_rotor_gap": 1.}}}
  model = HomodyneModel(cfg)
  plot_cfg = deepcopy(cfg)
  del plot_cfg["analyzer_parameters"]["start_frame"]
  del plot_cfg["analyzer_parameters"]["end_frame"]
  result = SimpleNamespace(parameters=np.array([.2, 1., 100., .8, .25]))
  control = _evaluate_c2_per_angle(model, result, {"wavevector_q_list": [.0054]}, plot_cfg, 0.)
  assert control.shape == (4, 4)
  surface = _evaluate_c2_per_angle(model, result, {"wavevector_q_list": [.005]}, plot_cfg, 0.)
  assert surface.shape == (4, 4) and np.isfinite(surface).all()
  ```
- Confidence: high

### R3-C2: Standard executor — An exhausted backend fit is published as converged
- Location: xpcsjax/optimization/nlsq/strategies/executors.py:173, xpcsjax/optimization/nlsq/strategies/executors.py:188
- Severity: Required
- Failure scenario: The exported standard executor now reaches the backend for an unbounded fit. It unpacks the returned parameters and covariance and fabricates convergence even when the backend returned `success=False` and status 0. A real one-evaluation solve of `y=2*x` reports failure with the unchanged seed `[1.]`; the executor labels that seed converged.
- Red test:
  ```python
  import logging
  from unittest.mock import patch
  import numpy as np
  from xpcsjax.optimization.nlsq.strategies import executors

  real = executors.curve_fit
  seen = []
  def short_budget(*args, **kwargs):
      kwargs["max_nfev"] = 1
      result = real(*args, **kwargs)
      seen.append(bool(result.success))
      return result

  x = np.arange(1., 5.)
  with patch.object(executors, "curve_fit", short_budget):
      r = executors.get_executor("standard").execute(
          lambda xb, a: a*xb, x, 2*x, np.array([1.]), None,
          "linear", "jac", logging.getLogger())
  assert seen == [False]
  assert r.info["success"] is False
  ```
- Confidence: high

Reviewed commit `14ac0ac`. The dedicated round-2 regression file passed all 15 tests in serial. The probes above cover additional cases. Full-suite and golden parity checks were outside this round's memory budget.

## Verdict
BUGS_FOUND
END_OF_REPORT
