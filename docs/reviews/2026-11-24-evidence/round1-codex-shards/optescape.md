# Codex review — shard optescape

## Files read

xpcsjax/optimization/nlsq/cmaes_wrapper.py — 1336 — full  
xpcsjax/optimization/nlsq/multistart.py — 1523 — full  
xpcsjax/optimization/nlsq/anti_degeneracy_controller.py — 1105 — full  
xpcsjax/optimization/nlsq/anti_degeneracy_diagnostics.py — 62 — full  
xpcsjax/optimization/nlsq/hierarchical.py — 733 — full  
xpcsjax/optimization/nlsq/adaptive_regularization.py — 560 — full  
xpcsjax/optimization/nlsq/gradient_monitor.py — 642 — full  
xpcsjax/optimization/nlsq/shear_weighting.py — 470 — full  
xpcsjax/optimization/nlsq/per_angle_mode.py — 402 — full  
xpcsjax/optimization/nlsq/config.py — 1037 — full  
xpcsjax/optimization/nlsq/__init__.py — 1203 — full

## Bugs

### optescape-B1: Hierarchical fitting can declare convergence immediately after scaling invalidates the physical solution

- Location: xpcsjax/optimization/nlsq/hierarchical.py:511
- Severity: Required
- Failure scenario: For loss `(h - s)**2 + (s - 1)**2 + o**2` and initial `[s, o, h] = [0, 0, 0]`, Stage 1 leaves `h = 0` and Stage 2 moves `s` to `0.5`. The fit returns `success=True` after one sweep, with loss `0.5` and physical gradient `-1`. The true minimum has loss zero.
- Root cause: The outer convergence check measures only physical displacement from before Stage 1. Stage 2 can change the physical optimum without moving the physical coordinates.
- Checked: Laminar stratified-LS and heterodyne streaming pass coupled loss functions to this optimizer. The laminar keep-better guard checks objective deterioration and can accept this partial improvement. `test_fit_converges_on_separable_quadratic` uses independent blocks and cannot expose the failure.
- Red test:
  ```python
  import numpy as np
  from xpcsjax.optimization.nlsq.hierarchical import HierarchicalConfig, HierarchicalOptimizer

  def loss(p):
      s, o, h = p
      return (h - s)**2 + (s - 1)**2 + o**2

  def grad(p):
      s, o, h = p
      return np.array([2 * (s - h) + 2 * (s - 1), 2 * o, 2 * (h - s)])

  opt = HierarchicalOptimizer(HierarchicalConfig(max_outer_iterations=30), 1, 1)
  result = opt.fit(loss, grad, np.zeros(3), (np.full(3, -3.), np.full(3, 3.)))
  assert result.fun < 1e-8
  ```
- Confidence: high

### optescape-B2: CMA-ES refinement reports convergence when the local solver exhausts its evaluation budget

- Location: xpcsjax/optimization/nlsq/cmaes_wrapper.py:703
- Severity: Required
- Failure scenario: A bounded fit of `y = 2*x` starting at slope `1` with `refinement_max_nfev=1` returns the unchanged slope. The refinement reports `success=True` with chi-squared `30`. The equal-cost keep-better check can then mark an unconverged global search successful.
- Root cause: Tuple unpacking keeps only parameters and covariance from NLSQ’s `CurveFitResult`. The wrapper discards the solver’s status and hardcodes `success=True` and `ier=1` whenever the call returns.
- Checked: Locked NLSQ 0.7.6 returns a result on budget exhaustion. `CMAESWrapper.fit` accepts equal chi-squared and uses the refinement flag in its final success decision. The homodyne CMA-ES warm-start also trusts this refinement status before considering auto-skip. Heterodyne warm-start success-gate tests stub the fit result and do not exercise this conversion.
- Red test:
  ```python
  import numpy as np
  from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapper, CMAESWrapperConfig

  x = np.arange(1., 5.)
  wrapper = CMAESWrapper(CMAESWrapperConfig(
      refinement_workflow="standard", refinement_max_nfev=1,
  ))
  result = wrapper._run_nlsq_refinement(
      lambda x, a: a * x, x, 2 * x, np.array([1.]),
      (np.array([0.]), np.array([4.])),
  )
  assert not result["success"]
  ```
- Confidence: high

### optescape-B3: Selecting streaming CMA-ES refinement prevents refinement from running

- Location: xpcsjax/optimization/nlsq/cmaes_wrapper.py:711
- Severity: Required
- Failure scenario: `cmaes.refinement_workflow: streaming` causes refinement to return its original point immediately, even for a finite, bounded one-parameter linear fit.
- Root cause: The wrapper selects `method="hybrid_streaming"` and still passes the TRF-only `ftol`, `xtol`, `gtol`, `max_nfev`, and `loss` keywords. Locked NLSQ 0.7.6 rejects them before the solve. The wrapper catches the `TypeError` and returns its fallback.
- Checked: The homodyne config parser accepts `streaming` and forwards it to the wrapper. The backend’s `_curve_fit_hybrid_streaming` rejects keywords outside `HybridStreamingConfig`. Existing CMA-ES trigger and memory-sizing tests do not exercise this refinement mode.
- Red test:
  ```python
  import numpy as np
  from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapper, CMAESWrapperConfig

  x = np.arange(1., 5.)
  wrapper = CMAESWrapper(CMAESWrapperConfig(refinement_workflow="streaming"))
  result = wrapper._run_nlsq_refinement(
      lambda x, a: a * x, x, 2 * x, np.array([1.]),
      (np.array([0.]), np.array([4.])),
  )
  assert result["chi_squared"] is not None, result["mesg"]
  ```
- Confidence: high

### optescape-B4: Adaptive population sizing overrides an explicit CMA-ES generation cap

- Location: xpcsjax/optimization/nlsq/cmaes_wrapper.py:995
- Severity: Required
- Failure scenario: `max_generations: 3` with `popsize: null` and bound widths `1` and `1e-5` passes a budget of 300 generations to the backend. With restarts disabled, this can still run up to 100 times the requested number of generations.
- Root cause: `to_cmaes_config` honors the explicit cap initially. The later adaptive branch checks only whether population size was supplied and overwrites both population size and generation count.
- Checked: `fit_nlsq_cmaes` forwards the parsed settings and the wrapper docs reserve adaptive generation sizing for `max_generations=None`. The backend uses the overwritten value as its loop limit. `test_cmaes_memory_sizing_order` checks sizing order and leaves explicit generation caps untested.
- Red test:
  ```python
  def test_explicit_generation_cap(monkeypatch):
      import numpy as np
      import nlsq.global_optimization as backend
      from xpcsjax.optimization.nlsq import cmaes_wrapper as cw

      class Capture:
          def __init__(self, config):
              assert config.max_generations == 3

          def fit(self, **kwargs):
              return {"popt": kwargs["p0"]}

      monkeypatch.setattr(cw, "CMAES_AVAILABLE", True)
      monkeypatch.setattr(backend, "CMAESOptimizer", Capture)
      wrapper = cw.CMAESWrapper(cw.CMAESWrapperConfig(
          max_generations=3, auto_memory=False, refine_with_nlsq=False,
          restart_strategy="none",
      ))
      x = np.arange(3.)
      upper = np.array([1., 1e-5])
      wrapper.fit(lambda x, a, b: a * x + b, x, x, upper / 2, (np.zeros(2), upper))
  ```
- Confidence: high

### optescape-B5: A timed-out multistart worker survives fallback and can block process exit

- Location: xpcsjax/optimization/nlsq/multistart.py:1479
- Severity: Required
- Failure scenario: A spawned fit stalls while the same start succeeds when retried in the parent. The sequential fallback returns a fit, but the stalled worker stays alive and the CLI can hang at interpreter shutdown.
- Root cause: `shutdown(wait=False, cancel_futures=True)` cancels pending work only. It leaves running workers alive, and the executor’s interpreter-exit hook joins the manager thread that is waiting for them.
- Checked: Homodyne `fit_nlsq_multistart` uses this process pool for small datasets with multiple workers. The dataset-size gate does not prevent stalls. The CPython 3.12 shutdown implementation confirms the join, and the existing multistart tests exercise sequential full fits and threaded screening.
- Red test:
  ```python
  # Keep the worker at module scope so spawn can import it.
  def _stall_in_child(idx, start, ready):
      import multiprocessing as mp
      import time
      from xpcsjax.optimization.nlsq.multistart import SingleStartResult

      if mp.parent_process() is not None:
          ready.set()
          time.sleep(60)
      return SingleStartResult(idx, start, start, 0., success=True)

  def test_timeout_reaps_worker(monkeypatch):
      import multiprocessing as mp
      from functools import partial
      from types import SimpleNamespace
      import numpy as np
      from xpcsjax.optimization.nlsq import multistart as ms

      before = {p.pid for p in mp.active_children()}
      with mp.get_context("spawn").Manager() as manager:
          ready = manager.Event()

          def forced_timeout(futures, timeout):
              assert ready.wait(30)
              raise TimeoutError

          monkeypatch.setattr(ms, "as_completed", forced_timeout)
          try:
              ms._run_parallel_with_progress(
                  partial(_stall_in_child, ready=ready), np.array([[0.], [1.]]),
                  2, SimpleNamespace(update=lambda **_: None),
              )
              assert not [p for p in mp.active_children()
                          if p.pid not in before and "Pool" in p.name]
          finally:
              for p in mp.active_children():
                  if p.pid not in before and "Pool" in p.name:
                      p.terminate()
                      p.join(5)
  ```
- Confidence: high

### optescape-B6: Multistart drops fixed parameters from the reported initial vectors

- Location: xpcsjax/optimization/nlsq/multistart.py:1187
- Severity: Minor
- Failure scenario: A laminar multistart fit with `fixed_parameters: {D_offset: 37.5}` returns `SingleStartResult.initial_params` without the fixed coordinate. Reading the vector in canonical order then attributes the next parameter’s value to `D_offset`.
- Root cause: `core._SingleFitWorker` restores the full parameter vector before solving and returns it as `initial_params`. `_OptimizeWorker` overwrites that field with the reduced sampling vector.
- Checked: `fit_nlsq_multistart` intentionally samples only free physical coordinates and routes both sequential and parallel runs through `_OptimizeWorker`. `test_fixed_parameter_survives_multistart_fit` checks the final result only.
- Red test: Reuse `data` and `cm` from `test_fixed_parameter_survives_multistart_fit` and make the following assertions.
  ```python
  from xpcsjax.optimization.nlsq.core import fit_nlsq_multistart

  result = fit_nlsq_multistart(data, cm)
  for start in result.all_results:
      assert len(start.initial_params) == 9
      assert start.initial_params[4] == 37.5
  ```
- Confidence: high

### optescape-B7: The NumPy regularization gradient is zero where the variance fallback has a nonzero derivative

- Location: xpcsjax/optimization/nlsq/adaptive_regularization.py:437
- Severity: Minor
- Failure scenario: A relative-mode group `[-1, 1]` with unit lambda, MSE, and point count receives a penalty of `1` and a reported gradient of `[0, 0]`. Its derivative is `[-1, 1]`.
- Root cause: The loss falls back to absolute variance when the mean is near zero. The analytic NumPy gradient skips that case entirely and leaves the initialized zeros in place.
- Checked: The JAX implementation differentiates the variance fallback correctly, and the in-tree optimization callers use that implementation. Existing zero-mean tests cover the JAX path and NumPy loss. The defect is in the directly callable NumPy gradient method.
- Red test:
  ```python
  import numpy as np
  from xpcsjax.optimization.nlsq.adaptive_regularization import (
      AdaptiveRegularizationConfig, AdaptiveRegularizer,
  )

  reg = AdaptiveRegularizer(AdaptiveRegularizationConfig(
      group_indices=[(0, 2)], auto_tune_lambda=False,
  ), n_phi=2)
  np.testing.assert_allclose(
      reg.compute_regularization_gradient(np.array([-1., 1.]), 1., 1),
      [-1., 1.],
  )
  ```
- Confidence: high

### optescape-B8: Homodyne config validation rejects the supported explicit `averaged` mode

- Location: xpcsjax/optimization/nlsq/config.py:749
- Severity: Minor
- Failure scenario: A configuration with `anti_degeneracy.per_angle_mode: averaged` emits a validation warning and returns `False` from `NLSQConfig.is_valid()`, even though it selects a documented solver mode.
- Root cause: The validation allowlist omits `averaged`, which the shared resolver and controller accept directly.
- Checked: Homodyne multistart and CMA-ES call `NLSQConfig.from_dict`, which logs the false validation error. The anti-degeneracy documentation lists explicit `averaged`. Resolver tests exercise explicit averaging, while `test_config_no_fourier.py` checks rejection of the removed token.
- Red test:
  ```python
  from xpcsjax.optimization.nlsq.config import NLSQConfig

  cfg = NLSQConfig.from_dict({"anti_degeneracy": {"per_angle_mode": "averaged"}})
  assert cfg.is_valid(), cfg.validate()
  ```
- Confidence: high

## Non-blocking notes

- Reviewed at `bd0a549`. Standalone probes reproduced B1–B3 and B7–B8. The remaining findings were checked by inspection. The test suite was skipped as requested.

## Verdict

BUGS_FOUND

END_OF_REPORT