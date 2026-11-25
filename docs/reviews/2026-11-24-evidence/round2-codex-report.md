# Codex round 2 report
## A. Fix review
### R2-A1: opthet-B2 — Restoring duplicate phi rows permutes physics parameters
- Location: xpcsjax/optimization/nlsq/heterodyne_result_builder.py:36
- Severity: Critical
- Failure scenario: For input phi `[0, 50, 100, 0]`, stratified LS fits six scaling slots for three distinct angles. The restore helper assumes eight slots and moves two physics entries into the scaling block. It also leaves the three-entry chi-squared array untouched while reordering the four-entry angle array. The sorted-input control skips the broken permutation. Existing duplicate-angle coverage checks finite chi-squared, which can survive this corruption.
- Red test:
  ```python
  import numpy as np
  from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
  from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
  from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
  from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult
  from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
      fit_heterodyne_stratified_least_squares as fit,
  )

  def test_duplicate_angles_keep_physics(monkeypatch):
      model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=12)
      phi = np.array(phi, copy=True)
      phi[-1] = phi[0]

      def solve(self, *, initial_params, **kwargs):
          p = np.array(initial_params, copy=True)
          p[:6] = [.2, .3, .4, .9, 1., 1.1]
          return NLSQResult(p, self._parameter_names, True, "ok")

      monkeypatch.setattr(NLSQAdapter, "fit", solve)
      cfg = NLSQConfig(per_angle_mode="individual")
      order = np.argsort(phi, kind="stable")
      a = fit(model=model, c2=c2, phi=phi, weights=None, config=cfg)
      b = fit(model=model, c2=c2[order], phi=phi[order], weights=None, config=cfg)
      np.testing.assert_allclose(a.parameters[-14:], b.parameters[-14:])
  ```
- Confidence: high

### R2-A2: opthet-B1/B2 — Sorting an individual fit indexes through a shared weight slab
- Location: xpcsjax/optimization/nlsq/heterodyne_stratified_ls.py:616, xpcsjax/optimization/nlsq/heterodyne_engine_route.py:419
- Severity: Required
- Failure scenario: Two unsorted angle rows and weights of shape `(1, n_t, n_t)` raise `IndexError` before the solver. The stratified builder explicitly broadcasts a shared slab. The same slab works on sorted input. Both new sort branches assume that every three-dimensional weight input has one slab per angle. The public dispatcher forwards weights and falls back on an engine exception, so the direct engine and stratified seams both matter.
- Red test:
  ```python
  import numpy as np
  from tests.review_r1.test_codex_opthet import _unsorted_two_angle, _patch_exact_fit
  from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
  from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
      fit_heterodyne_stratified_least_squares as fit,
  )

  def test_shared_weights_survive_phi_sort(monkeypatch):
      model, c2, phi = _unsorted_two_angle()
      seen = {}
      _patch_exact_fit(monkeypatch, seen)
      kw = dict(model=model, weights=np.ones((1, *c2.shape[1:])),
                config=NLSQConfig(per_angle_mode="individual"))
      fit(c2=c2[::-1], phi=phi[::-1], **kw)
      assert seen["ssr"] < 1e-20
      result = fit(c2=c2, phi=phi, **kw)
      np.testing.assert_allclose(result.parameters[2:4], [1.1, .9])
  ```
- Confidence: high

### R2-A3: opthet-B9 — The direct individual covariance still counts L3 penalty rows
- Location: xpcsjax/optimization/nlsq/heterodyne_core.py:3802, xpcsjax/optimization/nlsq/heterodyne_core.py:3947
- Severity: Required
- Failure scenario: The direct individual solve with zero-strength adaptive regularization adds two zero rows to the solver while reporting the same data SSR. On the 3-angle synthetic fixture the median covariance ratio against the unregularized result is 0.9935903, and the diagnostic reports 330 solver rows although the backend saw 332. This is the individual twin of the adjusted averaged branch.
- Red test:
  ```python
  import numpy as np
  from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
  from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
  from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

  def solve(regularization_mode):
      m, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
      return fit_nlsq_multi_phi(m, c2, phi, NLSQConfig(
          per_angle_mode="individual", regularization_mode=regularization_mode,
          group_variance_lambda=0., loss="linear"))
  a, b = solve("adaptive"), solve("none")
  np.testing.assert_allclose(a.chi_squared, b.chi_squared, rtol=1e-9)
  assert abs(np.median(np.diag(a.covariance) / np.diag(b.covariance)) - 1.) < 1e-3
  ```
- Confidence: high

### R2-A4: optescape-B2 — CMA-ES still reports convergence after accepting an unfinished refinement
- Location: xpcsjax/optimization/nlsq/cmaes_wrapper.py:1287
- Severity: Required
- Failure scenario: CMA-ES reaches its generation cap and the post-refinement reaches `max_nfev=1`, leaving SSR at 30. The wrapper adopts the tied point, correctly records `refinement_converged=False`, and returns `success=True`. The public result documents success as convergence, and homodyne and heterodyne callers use it for gating or reporting at `core.py:2997` and `heterodyne_core.py:4376`. Retain the candidate independently of the convergence verdict when changing those callers.
- Red test:
  ```python
  from unittest.mock import patch
  import numpy as np
  from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapper, CMAESWrapperConfig

  class Search:
      def __init__(self, config):
          pass
      def fit(self, **kw):
          return {"popt": kw["p0"],
                  "cmaes_diagnostics": {"convergence_reason": "max_generations"}}

  x = np.arange(1., 5.)
  with patch("nlsq.global_optimization.CMAESOptimizer", Search):
      r = CMAESWrapper(CMAESWrapperConfig(
          refinement_workflow="standard", refinement_max_nfev=1, auto_memory=False
      )).fit(lambda x, a: a*x, x, 2*x, np.array([1.]),
             (np.array([0.]), np.array([4.])))
  assert r.nlsq_refined and r.diagnostics["refinement_converged"] is False
  assert not r.success
  ```
- Confidence: high

### R2-A5: optescape-B3 — Weighted streaming refinement still never runs
- Location: xpcsjax/optimization/nlsq/cmaes_wrapper.py:703
- Severity: Required
- Failure scenario: The fix lets the unweighted regression exercise streaming by removing TRF kwargs, but it still passes sigma. The locked backend rejects every non-None sigma, including unit weights. Both homodyne CMA-ES seams pass `sigma_flat` at `xpcsjax/optimization/nlsq/core.py:2602` and `:2698`. Preserving actual weights requires a compatible residual representation or an explicit supported fallback. The unweighted branch also discards configured refinement budget and loss without reporting it.
- Red test:
  ```python
  import numpy as np
  from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapper, CMAESWrapperConfig
  x = np.arange(1., 5.)
  r = CMAESWrapper(CMAESWrapperConfig(refinement_workflow="streaming"))._run_nlsq_refinement(
      lambda x, a: a*x, x, 2*x, np.array([1.]), (np.array([0.]), np.array([4.])),
      sigma=np.array([1., 2., 3., 4.]))
  assert r["adoptable"], r["mesg"]
  ```
- Confidence: high

### R2-A6: optwrap-B5 — Sequential, adapter, stratified and CMA-ES diagnostics still charge fixed slots
- Location: xpcsjax/optimization/nlsq/wrapper.py:3232, xpcsjax/optimization/nlsq/adapter.py:1252, xpcsjax/optimization/nlsq/wrapper_stratified_route.py:579, xpcsjax/optimization/nlsq/core.py:2912
- Severity: Required
- Failure scenario: With `D_offset` fixed at 37.5, the sequential route divides by 287 instead of 288 (300 rows, 13 reported slots, 12 fitted). The opt-in static adapter divides by 295 instead of 296 (300 rows, five reported slots, four fitted). In laminar individual CMA-ES, a backend spy records 270 rows and 12 optimized slots, yet the restored 13-vector causes a divisor of 257 instead of 258. The stratified reporter uses 703 instead of 704 when holding its 720-row counting convention fixed (17 reported slots, 16 fitted). All four change reported reduced chi-squared even when the fitted point and raw SSR are held fixed. The fixed-slot mask is already available in these routes.
- Red test:
  ```python
  import numpy as np
  from tests.optimization.test_fixed_parameters_integration import _config, _synthetic_data
  from xpcsjax.config.manager import ConfigManager
  from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

  def test_fixed_slot_does_not_consume_dof():
      for mode, adapter, extra in [
          ("laminar_flow", False, {"optimization": {"stratification": {
              "force_sequential_fallback": True}}}),
          ("static_isotropic", True, {}),
      ]:
          data = _synthetic_data(mode)
          cfg = ConfigManager(config_override=_config(
              mode, fixed_parameters={"D_offset": 37.5}, extra_top=extra))
          r = fit_nlsq_jax(data, cfg, use_adapter=adapter)
          expected_dof = data["g2"].size - len(r.parameters) + 1
          np.testing.assert_allclose(r.reduced_chi_squared,
                                     r.chi_squared / expected_dof, rtol=1e-13, atol=0)
  ```
  The CMA-ES variant uses the same laminar fixture and fixed parameter with individual mode and `nlsq_warmstart=False`. Spy on `CMAESWrapper.fit` and return `CMAESResult(p0, np.eye(len(p0)), 30., True)`. For `fit_nlsq_cmaes`, assert `r.reduced_chi_squared == 30 / (len(ydata) - len(p0))` using the spy's optimizer-space vectors.

  The stratified variant uses `tests.parity.test_laminar_execute_layers._build_laminar_stratified_data()` (five angles, 12 frames) in `run_stratified_ls_route`. Set `D_offset=37.5`, supply a `ResolvedPhysicalParameters` mask with only `D_offset` fixed, select individual mode, and stub only `NLSQWrapper._fit_with_stratified_least_squares` to return the expanded fit seed, an identity covariance and `{"success": True}`. Assert `r.reduced_chi_squared == pytest.approx(r.chi_squared / 704, rel=1e-12, abs=0)` under this reporter's existing row convention.
- Confidence: high

### R2-A7: optstrat-B4 — Stratified covariance still counts rows masked by sigma
- Location: xpcsjax/optimization/nlsq/strategies/stratified_ls.py:961
- Severity: Required
- Failure scenario: Five 12-frame angle slabs have 660 off-diagonal cells. Setting one off-diagonal sigma to zero makes the JIT objective discard that cell at `strategies/residual_jit.py:506–516`, but covariance still uses 660 observations instead of 659. The current regression uses unit sigma everywhere.
- Red test: Extend `tests/review_r1/test_codex_optstrat.py::test_optstrat_b4`. Build with `_build_laminar_stratified_data()` from `tests.parity.test_laminar_execute_layers`, set `strat.sigma[0, 0, 1] = 0`, run the same stratified solve with the existing `gauss_newton_covariance` spy, and assert `seen["n_valid"] == 659`. The spy receives 660.
- Confidence: high

### R2-A8: data-B4 — Integer and Boolean reconstruction still corrupts the diagonal
- Location: xpcsjax/data/hdf5_readers.py:94, xpcsjax/data/xpcs_loader.py:1092
- Severity: Required
- Failure scenario: A stored `uint8` diagonal of 200 is doubled in `uint8`, wraps to 144, then halved in the promoted output to 72. A true Boolean diagonal becomes 0.5 due to Boolean addition. The APS-U selective and quality-filtered probes both reproduced 72. The APS-old reader uses the same two reconstructors.
- Red test:
  ```python
  import numpy as np
  from tests.review_r1.test_codex_data import _bare_loader, _half, _write_aps_u

  def test_integer_symmetrization_promotes_before_addition(tmp_path):
      path = tmp_path / "integer.h5"
      _write_aps_u(path, [_half(200, dtype=np.uint8)], [0.])
      for config in ({}, {"data_filtering": {"enabled": True, "quality_threshold": .9}}):
          r = _bare_loader(config)._load_from_hdf(str(path))
          assert np.asarray(r["c2_exp"])[0, 0, 0] == 200
  ```
- Confidence: high

### R2-A9: data-B7 — A pre-upgrade cache still wins over a changed requested frame window
- Location: xpcsjax/data/npz_cache.py:495
- Severity: Required
- Failure scenario: Upgrade with an existing JSON-metadata cache built for frames 1–2 under a fixed cache filename. Request frames 1–4 from the same four-frame HDF5 source. The missing requested-window keys trigger only a warning and the automatic cache hit returns a 2×2 surface. The older cache already stores the applied frame bounds. An automatic cache with an available HDF5 source can be regenerated when compatibility cannot be verified.
- Red test: In `tests/review_r1/test_codex_data.py::test_data_b7`, after the first load, read the arrays from `fixed.npz` with `allow_pickle=False`, decode `cache_metadata_json`, remove only `requested_start_frame` and `requested_end_frame`, and rewrite the NPZ. Keep the original second-load assertion `second.c2.shape[-2:] == (4, 4)`. The second load returns `(2, 2)`.
- Confidence: high

### R2-A10: data-B11 — NPZ validation accepts multiple phi coordinates per matrix
- Location: xpcsjax/data/xpcs_loader.py:270
- Severity: Required
- Failure scenario: A `(2, 2)` phi array passes the new axis-0 count against two matrices. The public service flattens it to four coordinates and skips the requested phi subset when the sizes disagree. The I/O boundary promises one coordinate per matrix. This was reproduced on the actual NPZ read path.
- Red test:
  ```python
  import io
  import numpy as np
  import pytest
  from tests.review_r1.test_codex_data import _bare_loader, _qc_data
  from xpcsjax.data.npz_cache import load_from_cache
  from xpcsjax.data.xpcs_loader import XPCSDataFormatError

  def test_npz_has_one_phi_per_surface():
      data = _qc_data()
      data.update(c2_exp=np.repeat(data["c2_exp"], 2, axis=0),
                  wavevector_q_list=np.array([.01, .01]),
                  phi_angles_list=np.array([[0., 45.], [90., 135.]]))
      buf = io.BytesIO()
      np.savez(buf, **data)
      buf.seek(0)
      with pytest.raises(XPCSDataFormatError):
          load_from_cache(_bare_loader(), buf)
  ```
- Confidence: high

### R2-A11: gui-B4/ioviz-B2 — The documented stateful plotting path still evaluates the wrong q
- Location: xpcsjax/viz/nlsq_plots.py:587
- Severity: Required
- Failure scenario: The public visualization example supplies `HomodyneModel(config)` with its result. When config q is .0054 and measured q is .005, the new annotation uses .005 but the evaluator returns through the model's stored q before calling `_fit_q`. The fitted surface was wrong at 4032 of 4096 cells in the focused probe. The existing regression covers the other evaluator branch. See `docs/source/user_guide/visualization.rst:15–25`.
- Red test: Extend `tests/review_r1/test_codex_ioviz.py::test_ioviz_b2` with the stateful model. Evaluate `_evaluate_c2_per_angle(HomodyneModel(config), result, data, config, 0.)` and compare with `HomodyneModel(config_with_measured_q).compute_c2_single_angle(result.parameters[2:], 0., .2, 1.)`. Keep config q at .0054 and set only the comparison model's q to .005.
- Confidence: high

### R2-A12: cli-B4 — Partial nested scattering hides the top-level q
- Location: xpcsjax/cli/plot_families/simulated.py:115
- Severity: Required
- Failure scenario: A nested scattering mapping containing only `phi_angles` is truthy and hides top-level `wavevector_q=.037`. The standalone preview uses .01 while `HeterodyneModel.from_config` resolves q per key. This is a common split between the angle list and the physical q.
- Red test:
  ```python
  from pathlib import Path
  from unittest.mock import patch
  import numpy as np
  from xpcsjax.cli.plot_families import simulated
  from xpcsjax.config import ConfigManager

  cfg = {"analysis_mode": "two_component", "scattering": {"wavevector_q": .037},
         "analyzer_parameters": {"dt": .1, "start_frame": 1, "end_frame": 4,
                                 "scattering": {"phi_angles": [0.]}}}
  with patch.object(simulated, "_evaluate_model_c2", return_value=np.ones((4, 4))) as spy, \
       patch("xpcsjax.viz.plot_simulated_data"):
      simulated._plot_simulated_from_config(ConfigManager(config_override=cfg),
                                            .3, 1., "0", Path("."), None)
      assert spy.call_args.kwargs["q"] == .037
  ```
- Confidence: high

### R2-A13: cli-B8 — Newline paths containing a non-BMP character do not round-trip
- Location: xpcsjax/cli/config_template.py:154
- Severity: Minor
- Failure scenario: The newline branch writes the default ASCII form of a JSON string. PyYAML decodes an emoji in that form as two UTF-16 surrogate code points, so the parsed filename differs from the supplied path and cannot be opened.
- Red test: In the existing newline path regression in `tests/review_r1/test_codex_cli.py`, set `data_path = "scan 😀\n.h5"`. Generate the config and assert the parsed `experimental_data.file_path` equals `data_path`. The generated scalar is valid YAML, but the value is different.
- Confidence: high

## B. Disputes
| id | Ruling | Evidence or disposition |
| --- | --- | --- |
| optrest-B3 | CONCEDE | Leave the out-of-core constant-mode accounting contract to the maintainer. |
| optrest-B4 | CONCEDE | Changing the recovery scale changes a pinned solver trajectory. |
| config-B2 | CONCEDE | Equal bounds in the heterodyne optimizer vector are an explicit current convention. |
| config-B5 | COUNTER | The documented standalone CLI previews config parameters at `docs/source/user_guide/cli.rst:127–129`. The changed preview in `xpcsjax/cli/plot_families/simulated.py:102,161–168` still passed `v0=500000` to its evaluator when grouped YAML set 42. The optimizer reads the grouped values. Use a preview-specific source so the generic midpoint accessor and all fit results keep their current meaning. Red assertion on the real preview seam in R2-A12, after adding `"parameters": {"velocity": {"v0": {"value": 42}}}` to its config: `assert spy.call_args.args[1][6] == 42`. |
| config-B9 | CONCEDE | The generic midpoint accessor has no supported consumer of the per-angle slots. |
| config-B11 | CONCEDE | No supported caller or finite-time contract for that helper was established. |
| core-B1 | COUNTER | The dispatcher creates a dummy `arange` at `xpcsjax/optimization/nlsq/__init__.py:792–798` even though raw HDF readers generate real axes and the cache stores the applied start/end at `xpcsjax/data/npz_cache.py:300–307`. On a request beyond EOF the HDF path keeps the first frame and clamps only the end, so a proven tail clamp can preserve the leading model time without changing a legitimate leading exclusion or any matching-length fit. Preserve that provenance on cold and warm loads, or reject the ambiguous short window explicitly. This changes affected invalid/inconsistent fits only. |
| core-B2 | CONCEDE | The scattered quadrature rule and tolerance are deliberate. |
| core-B3 | CONCEDE | The time-grid wiring was explicitly deferred. |
| core-B4 | CONCEDE | No supported nonuniform matrix-mode path was found. |
| core-B9 | CONCEDE | No supported consumer requiring identical clip functions was found. |
| optescape-B1 | CONCEDE | Leave the L2 iteration and convergence policy to the maintainer because additional sweeps change the selected fit. |
| optescape-B5 | CONCEDE | Worker lifetime after the multistart fallback is explicitly addressed by the existing policy. |
| optescape-B7 | CONCEDE | The NumPy gradient helper lacks a supported caller. |
| gui-B2 | CONCEDE | Maintainer decision on whether filtering changes the experiment-level q. |
| gui-B14 | CONCEDE | No supported production consumer was found for this validator. |
| cli-B9 | CONCEDE | Retaining caller-supplied host-device count matches the package policy. |
| cli-B10 | CONCEDE | Changing the published CLI import contract needs a maintainer decision. |
| optwrap-B4 | CONCEDE | The standard-fit covariance and expanded diagnostic covariance are a maintainer reporting-policy decision. |
| optwrap-B6 | CONCEDE | The normal path validates or supplies sigma before this branch. |
| ioviz-B7 | CONCEDE | Existing logging tests explicitly pin propagation handling. |
| ioviz-B8 | CONCEDE | The annotation's scalar mean is documented as a summary. |
| ioviz-B9 | CONCEDE | The panel is documented in terms of the matrix diagonal and current production axes agree. |
| ioviz-B10 | COUNTER | The singleton case is reachable directly through the documented `DatashaderRenderer.rasterize_heatmap` API at `docs/source/api/viz.rst:89–90`. A 3×2 control rendered 16×16 pixels. The same public call with matrix `np.full((3,1),1.2)`, x `np.array([0.])`, y `np.arange(3.)`, and limits 1 and 1.5 raises `IndexError` at `xpcsjax/viz/datashader_backend.py:129`. The renderer accepts coordinate vectors matching each matrix dimension and describes no two-sample minimum. |
| data-B1 | CONCEDE | Same experiment-level q decision as gui-B2. |
| data-B9 | CONCEDE | Cache-directory half only. Moving existing template users' caches is a maintainer compatibility decision. |
| data-B14 | CONCEDE | The loader rejects nonfinite data before the cited auto-repair path. |
| data-B15 | CONCEDE | The loader creates a fresh quality controller per load. No supported cross-load reuse was established. |
| data-B17 | COUNTER | `validate_config_schema` is explicitly in the module's `__all__` at `xpcsjax/data/config.py:541–547`. Its public contract returns an error report. Calling it with a valid file key and `analyzer_parameters={"start_frame":1,"end_frame":4,"dt":"bad"}` reproduced `TypeError` at semantic validation (`:345`). An in-package caller is unnecessary under the lead's public-surface rule. |
| opthet-B6 | CONCEDE | Leave adapter termination routing to the maintainer because changing it can cause a second solve. |
| opthet-B8 | CONCEDE | Leave the heterodyne setup preflight policy and solver effects to the maintainer. |
| opthet-B10 | CONCEDE | Honoring the published enable flag changes the shipped objective and needs a maintainer choice. |
| opthet-B11 | CONCEDE | Uniform weights under a robust loss need an explicit objective and routing decision. |
| opthet-B13 | CONCEDE | Leave the pinned sequential heterodyne reporting convention to the maintainer. |
| optstrat-B7 | CONCEDE | Per-angle pseudoinverse behavior has a separate documented convention. |
| optstrat-B9 | CONCEDE | Same regularization-policy decision as opthet-B10. |
| optstrat-B10 | CONCEDE | Maintainer choice whether to make the fallback streaming tier functional or remove it. |
| optstrat-B11 | CONCEDE | Must be resolved with optstrat-B10 before the streaming tier can be enabled. |
| optstrat-B12 | COUNTER | `get_executor("streaming")` is documented at `xpcsjax/optimization/nlsq/strategies/executors.py:450–457` and exported at `xpcsjax/optimization/nlsq/strategies/__init__.py:63–64`. The real backend, without the round-1 spy, reproduced Phase 2 shapes `(12,)` and `(4,)` at `executors.py:403` with `x=np.arange(12.)`, `y=2*x`, `p0=[1.]`, bounds `[0.,4.]`, callback `lambda x,a:(2-a)*x`, and `chunk_size=4`. It returned 1.7 and failed convergence; expected parameter 2. |
| optstrat-B13 | CONCEDE | Full out-of-core passes and in-memory iteration limits have distinct cost and semantics. |
| optstrat-B16 | COUNTER | The exported standard executor is the factory's own example at `xpcsjax/optimization/nlsq/strategies/executors.py:469–474`. Its public signature permits `bounds=None`. The real backend raised `TypeError: 'NoneType' object is not iterable` at the forward call (`:173`) for a four-point `y=2*x` line. The public large-dataset sibling already implements the unbounded convention. |

The ledger names 12 rejected items and 29 design items, counting the design half of data-B9 and including the explicit reclassifications. The stated tally of 31 design items has two additional slots without IDs. The explicitly deferred parts of optwrap-B5 and opthet-B9 are addressed in section A.

## C. New findings
### R2-C1: CLI standalone simulation — Reordered complete active lists scramble physics
- Location: xpcsjax/cli/plot_families/simulated.py:157
- Severity: Required
- Failure scenario: The fixed-parameter change switches to registry order only for a strict subset. With the complete static active list in the order `[alpha, D0, D_offset]`, the evaluator receives `[.8, 100., .25]` in positional slots for `[D0, alpha, D_offset]`. The real ConfigManager and the preview seam reproduced this. The persistence change already recognizes that active list order differs from solver order.
- Red test:
  ```python
  from pathlib import Path
  from unittest.mock import patch
  import numpy as np
  from xpcsjax.cli.plot_families import simulated
  from xpcsjax.config import ConfigManager

  cfg = {"analysis_mode": "static_isotropic",
         "analyzer_parameters": {"dt": .1, "start_frame": 1, "end_frame": 2},
         "initial_parameters": {"parameter_names": ["D0", "alpha", "D_offset"],
                                "values": [100., .8, .25],
                                "active_parameters": ["alpha", "D0", "D_offset"]}}
  with patch.object(simulated, "_evaluate_model_c2", return_value=np.ones((2, 2))) as spy, \
       patch("xpcsjax.viz.plot_simulated_data"):
      simulated._plot_simulated_from_config(ConfigManager(config_override=cfg),
                                            .3, 1., "0", Path("."), None)
      np.testing.assert_allclose(spy.call_args.args[1], [100., .8, .25])
  ```
- Confidence: high

### R2-C2: HDF5 allocation — Guards budget the source dtype before allocating the promoted buffer
- Location: xpcsjax/data/hdf5_readers.py:127, xpcsjax/data/hdf5_readers.py:366, xpcsjax/data/hdf5_readers.py:694
- Severity: Required
- Failure scenario: A 4×4 uint8 slab is budgeted for 16 bytes. With a 32-byte test cap it passes preflight and allocates a 64-byte float32 destination. The later validation rejects after allocation. Large inputs can exceed the process's available memory before the final check. The APS-U allocation was observed using an allocation spy.
- Red test: Use the APS-U fixture from R2-A8. Monkeypatch `xpcsjax.data.xpcs_loader.MAX_CORRELATION_ALLOC_BYTES` to 32 and spy on `hdf5_readers.np.empty`, forwarding to the original. Assert an `XPCSDataFormatError` is raised and no call allocates `(1,4,4)` with a dtype larger than two bytes per item. The second assertion fails on the float32 allocation. The quality-path preflight also uses the on-disk itemsize.
- Confidence: high

### R2-C3: GUI config save — Atomic replacement breaks shared YAML symlinks
- Location: xpcsjax/gui/views/config_dialogs.py:306
- Severity: Required
- Failure scenario: Open `dataset.yaml` as a symlink to a shared YAML. The editor reads the target. Save replaces the symlink with a regular file and leaves the target unchanged. The previous save followed the symlink. A no-write spy on the actual save seam confirmed the replacement is directed at the link.
- Red test:
  ```python
  from types import SimpleNamespace
  from xpcsjax.gui.views.config_dialogs import ConfigTextEditorDialog

  def test_preserve_config_symlink(tmp_path):
      target = tmp_path / "shared.yaml"
      target.write_text("before: true\n")
      link = tmp_path / "dataset.yaml"
      link.symlink_to(target)
      editor = SimpleNamespace(_path=link, _load_error=None,
                               _editor=SimpleNamespace(toPlainText=lambda: "after: true\n"))
      ConfigTextEditorDialog.save(editor)
      assert link.is_symlink()
      assert target.read_text() == "after: true\n"
  ```
- Confidence: high

### R2-C4: NPZ cache replay — A reused loader accumulates the same degradation on every warm hit
- Location: xpcsjax/data/npz_cache.py:150, xpcsjax/data/xpcs_loader.py:1012
- Severity: Minor
- Failure scenario: A consumer reuses one loader to reload the same cached filtered dataset. Each warm hit appends the same cached reason to an instance list retained across calls. The result exposes accumulated reasons for all prior loads as current reasons and the list grows indefinitely. Two reads of the same in-memory NPZ reproduced two identical entries.
- Red test:
  ```python
  import io
  import json
  import numpy as np
  from tests.review_r1.test_codex_data import _bare_loader, _qc_data
  from xpcsjax.data.npz_cache import load_from_cache

  loader = _bare_loader()
  buf = io.BytesIO()
  np.savez(buf, **_qc_data(),
           cache_metadata_json=json.dumps({"load_degradations": ["fallback"]}))
  raw = buf.getvalue()
  for _ in range(2):
      load_from_cache(loader, io.BytesIO(raw))
  assert len(loader.load_degradations) == 1
  ```
- Confidence: high

Reviewed the source diff at `b55206d` against the ledger and checked the adjacent callers and regression cases. The focused serial files `test_codex_optwrap.py` and `test_codex_data.py` passed 8 and 14 tests respectively. The red probes above exercise additional cases. Full-suite and final golden parity verification were outside this round's memory budget.

## Verdict
BUGS_FOUND
END_OF_REPORT
